import os
import json
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

from evoagent.auth import AuthManager, hash_password, verify_password
from evoagent.harness import ReviewHarness
from evoagent.reviewer import LocalRuleReviewer
from evoagent.rollout import ReleaseManager
from evoagent.service import ReviewService
from evoagent.store import TaskStore
from evoagent.task_queue import TaskQueue
from evoagent.verifier import RepairVerifier


DIFF = "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+eval(data)\n"


class PasswordTests(unittest.TestCase):
    def test_password_accepts_six_characters(self):
        password_hash = hash_password("123456")

        self.assertTrue(verify_password("123456", password_hash))
        with self.assertRaisesRegex(ValueError, "at least 6"):
            hash_password("12345")


class ProductionFeatureTests(unittest.TestCase):
    def setUp(self):
        handle, self.path = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        self.store = TaskStore(self.path)

    def tearDown(self):
        os.unlink(self.path)

    def test_login_rbac_and_tenant_task_isolation(self):
        auth = AuthManager(
            self.store, "a" * 32, bootstrap_username="alice",
            bootstrap_password="correct-horse", default_tenant_id="tenant-a",
        )
        token = auth.login("alice", "correct-horse")["access_token"]
        principal = auth.authenticate("Bearer " + token)
        self.assertTrue(principal.can("manage"))
        self.store.create("a", "org/a", 1, {}, "tenant-a")
        self.store.create("b", "org/b", 2, {}, "tenant-b")
        self.assertIsNotNone(self.store.get("a", principal.tenant_id))
        self.assertIsNone(self.store.get("b", principal.tenant_id))
        self.assertEqual(["a"], [item["id"] for item in self.store.list_tasks(10, "tenant-a")])

    def test_webhook_delivery_is_idempotent_and_payload_bound(self):
        self.assertTrue(self.store.claim_webhook("delivery-1", "t", "pull_request", "aaa"))
        self.assertFalse(self.store.claim_webhook("delivery-1", "t", "pull_request", "aaa"))
        with self.assertRaisesRegex(ValueError, "different payload"):
            self.store.claim_webhook("delivery-1", "t", "pull_request", "bbb")

    def test_failure_cases_are_filtered_by_tenant(self):
        self.store.create("a", "org/a", 1, {}, "tenant-a")
        self.store.create("b", "org/b", 2, {}, "tenant-b")
        self.store.record_failure_case("a", "false_positive", {"note": "a"})
        self.store.record_failure_case("b", "missed_issue", {"note": "b"})

        cases = self.store.list_failure_cases(tenant_id="tenant-a")

        self.assertEqual(["a"], [item["task_id"] for item in cases])

    def test_failed_graph_resumes_after_last_completed_checkpoint(self):
        class BrokenReviewer:
            name = "broken"

            def review(self, _diff, _parsed):
                raise RuntimeError("temporary provider failure")

        self.store.create("task", "org/repo", 1, {})
        with self.assertRaises(RuntimeError):
            ReviewHarness(
                self.store, BrokenReviewer(), node_retries=0
            ).run("task", "org/repo", 1, DIFF)
        checkpoints = self.store.load_checkpoints("task")
        self.assertEqual("completed", checkpoints["planning"]["status"])
        self.assertEqual("failed", checkpoints["executing"]["status"])

        report = ReviewHarness(
            self.store, LocalRuleReviewer(), node_retries=0
        ).resume("task", "org/repo", 1, DIFF)
        self.assertEqual("high", report.risk)
        planning_events = [
            item for item in self.store.get("task")["trace"] if item["state"] == "PLANNING"
        ]
        self.assertEqual(1, len(planning_events))

    def test_queue_moves_terminal_failure_to_dlq(self):
        def broken(_payload):
            raise RuntimeError("boom")

        queue = TaskQueue(broken, workers=1, max_attempts=1)
        queue.submit({"task_id": "dead"})
        for _ in range(100):
            if queue.dead_letters():
                break
            time.sleep(.01)
        letters = queue.dead_letters()
        queue.close()
        self.assertEqual("dead", letters[0]["message_id"])
        self.assertIn("boom", letters[0]["error"])

    def test_dead_letter_marks_pending_task_failed(self):
        self.store.create("dead", "org/repo", 1, {}, "tenant")
        service = ReviewService.__new__(ReviewService)
        service.store = self.store

        service._on_dead_letter({"task_id": "dead", "tenant_id": "tenant"}, "boom")

        task = self.store.get("dead", "tenant")
        self.assertEqual("FAILED", task["state"])
        self.assertEqual("boom", task["error"])
        self.assertEqual("dead", self.store.list_dead_letters()[0]["message_id"])

    def test_dead_letter_can_be_replayed_after_queue_restart(self):
        self.store.record_dead_letter("dead", {"task_id": "dead"}, "boom")

        item = self.store.get_dead_letter("dead")

        self.assertEqual({"task_id": "dead"}, item["payload"])
        self.store.remove_dead_letter("dead")
        self.assertIsNone(self.store.get_dead_letter("dead"))

    def test_rocketmq_retries_then_publishes_terminal_failure_to_dlq(self):
        class ConsumeStatus:
            CONSUME_SUCCESS = "ack"
            RECONSUME_LATER = "retry"

        class Message:
            def __init__(self, topic):
                self.topic = topic
                self.keys = ""
                self.body = b""

            def set_keys(self, keys):
                self.keys = keys

            def set_body(self, body):
                self.body = body

        class Producer:
            sent = []

            def __init__(self, _group):
                pass

            def set_name_server_address(self, _nameserver):
                pass

            def start(self):
                pass

            def send_sync(self, message):
                self.sent.append(message)

            def shutdown(self):
                pass

        class PushConsumer:
            instance = None

            def __init__(self, _group):
                self.callback = None
                PushConsumer.instance = self

            def set_name_server_address(self, _nameserver):
                pass

            def set_thread_count(self, _workers):
                pass

            def subscribe(self, _topic, callback):
                self.callback = callback

            def start(self):
                pass

            def shutdown(self):
                pass

        client = types.ModuleType("rocketmq.client")
        client.ConsumeStatus = ConsumeStatus
        client.Message = Message
        client.Producer = Producer
        client.PushConsumer = PushConsumer
        package = types.ModuleType("rocketmq")
        package.client = client

        with patch.dict(sys.modules, {"rocketmq": package, "rocketmq.client": client}):
            task_queue = TaskQueue(
                lambda _payload: (_ for _ in ()).throw(RuntimeError("boom")),
                workers=1, rocketmq_nameserver="namesrv:9876", max_attempts=2,
            )
            body = json.dumps({"message_id": "dead", "payload": {"task_id": "dead"}}).encode()
            message = types.SimpleNamespace(body=body, id="broker-id", reconsume_times=0)
            self.assertEqual("retry", PushConsumer.instance.callback(message))
            message.reconsume_times = 1
            self.assertEqual("ack", PushConsumer.instance.callback(message))
            task_queue.close()

        self.assertEqual(TaskQueue.DLQ_TOPIC, Producer.sent[-1].topic)

    def test_canary_assignment_and_error_budget_rollback(self):
        release = ReleaseManager(self.store)
        release.configure("tenant", "skill", {
            "stable_version": 1, "candidate_version": 2,
            "canary_percent": 100, "shadow_percent": 100,
            "min_samples": 2, "max_error_rate": .25,
        })
        self.assertEqual("canary", release.assignment("tenant", "skill", "task")["lane"])
        release.observe("tenant", "skill", True)
        result = release.observe("tenant", "skill", False)
        self.assertEqual("rolled_back", result["status"])
        self.assertTrue(self.store.list_alerts("tenant"))

    def test_repair_verifier_blocks_invalid_python(self):
        result = RepairVerifier().verify_contents({"app.py": "def broken(:\n"})
        self.assertFalse(result["passed"])
        self.assertEqual("compile:app.py", result["checks"][0]["name"])


if __name__ == "__main__":
    unittest.main()
