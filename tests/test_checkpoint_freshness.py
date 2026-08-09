import os
import tempfile
import unittest

from evoagent.harness import ReviewHarness
from evoagent.models import ReviewReport, TaskState, TraceEvent
from evoagent.reviewer import LocalRuleReviewer
from evoagent.runtime import AgentRuntime, RuntimeNode
from evoagent.store import TaskStore, utc_now


DIFF = "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+eval(data)\n"
OTHER_DIFF = "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+print(data)\n"


class CheckpointFreshnessTests(unittest.TestCase):
    def setUp(self):
        handle, self.path = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        self.store = TaskStore(self.path)
        self.store.create("task", "org/repo", 1, {})

    def tearDown(self):
        os.unlink(self.path)

    def _states(self, state):
        return [item for item in self.store.get("task")["trace"] if item["state"] == state]

    def test_changed_diff_invalidates_completed_planning_checkpoint(self):
        class FailingReviewHarness(ReviewHarness):
            def _reviewing(self, state):
                super()._reviewing(state)
                raise RuntimeError("reporting failure")

        with self.assertRaises(RuntimeError):
            FailingReviewHarness(self.store, LocalRuleReviewer(), node_retries=0,
                                 runtime_fingerprint="profile-1").run(
                "task", "org/repo", 1, DIFF
            )

        report = ReviewHarness(self.store, LocalRuleReviewer(), node_retries=0,
                               runtime_fingerprint="profile-1").resume(
            "task", "org/repo", 1, OTHER_DIFF
        )

        self.assertEqual(2, len(self._states("PLANNING")))
        self.assertEqual(2, len(self._states("EXECUTING")))
        self.assertEqual("low", report.risk)

    def test_changed_runtime_profile_reuses_planning_but_reruns_execution(self):
        class FailingReviewHarness(ReviewHarness):
            def _reviewing(self, state):
                super()._reviewing(state)
                raise RuntimeError("reporting failure")

        with self.assertRaises(RuntimeError):
            FailingReviewHarness(self.store, LocalRuleReviewer(), node_retries=0,
                                 runtime_fingerprint="profile-1").run(
                "task", "org/repo", 1, DIFF
            )

        ReviewHarness(self.store, LocalRuleReviewer(), node_retries=0,
                      runtime_fingerprint="profile-2").resume(
            "task", "org/repo", 1, DIFF
        )

        self.assertEqual(1, len(self._states("PLANNING")))
        self.assertEqual(2, len(self._states("EXECUTING")))

    def test_changed_report_revision_only_reruns_reviewing(self):
        class FailingReviewHarness(ReviewHarness):
            def _reviewing(self, state):
                super()._reviewing(state)
                raise RuntimeError("reporting failure")

        with self.assertRaises(RuntimeError):
            FailingReviewHarness(self.store, LocalRuleReviewer(), node_retries=0,
                                 runtime_fingerprint="profile-1").run(
                "task", "org/repo", 1, DIFF
            )

        resumed = ReviewHarness(self.store, LocalRuleReviewer(), node_retries=0,
                                runtime_fingerprint="profile-1")
        resumed.REPORT_REVISION = "2"
        resumed.resume("task", "org/repo", 1, DIFF)

        self.assertEqual(1, len(self._states("PLANNING")))
        self.assertEqual(1, len(self._states("EXECUTING")))
        self.assertEqual(2, len(self._states("REVIEWING")))

    def test_matching_fingerprints_restore_completed_stages(self):
        class FailingReviewHarness(ReviewHarness):
            def _reviewing(self, state):
                super()._reviewing(state)
                raise RuntimeError("reporting failure")

        with self.assertRaises(RuntimeError):
            FailingReviewHarness(self.store, LocalRuleReviewer(), node_retries=0,
                                 runtime_fingerprint="profile-1").run(
                "task", "org/repo", 1, DIFF
            )

        ReviewHarness(self.store, LocalRuleReviewer(), node_retries=0,
                      runtime_fingerprint="profile-1").resume(
            "task", "org/repo", 1, DIFF
        )

        self.assertEqual(1, len(self._states("PLANNING")))
        self.assertEqual(1, len(self._states("EXECUTING")))

    def test_runtime_emits_checkpoint_invalidated_event(self):
        calls = []
        events = []
        runtime = AgentRuntime(max_steps=2, timeout_seconds=5)
        nodes = [RuntimeNode("plan", lambda _state: calls.append("plan") or {"value": 1})]

        runtime.execute({}, nodes, "task", self.store,
                        checkpoint_fingerprints={"plan": "first"})
        runtime.execute({}, nodes, "task", self.store,
                        checkpoint_fingerprints={"plan": "second"},
                        event_sink=events.append)

        self.assertEqual(["plan", "plan"], calls)
        self.assertIn("checkpoint_invalidated", [item.kind for item in events])

    def test_legacy_checkpoint_without_fingerprint_is_not_restored(self):
        self.store.save_checkpoint("task", "planning", {"parsed": {}}, "completed")

        ReviewHarness(self.store, LocalRuleReviewer(), node_retries=0,
                      runtime_fingerprint="profile-1").run(
            "task", "org/repo", 1, DIFF
        )

        checkpoint = self.store.load_checkpoints("task")["planning"]
        self.assertTrue(checkpoint["fingerprint"])
        self.assertEqual(1, len(self._states("PLANNING")))

    def test_superseded_run_token_cannot_write_checkpoint_or_state(self):
        old_token = self.store.issue_run_token("task")
        current_token = self.store.issue_run_token("task")
        old_claim = "old-claim"
        current_claim = "current-claim"
        self.assertTrue(self.store.claim_run("task", current_token, old_claim, 60))
        with self.store._connect() as conn:
            conn.execute("UPDATE tasks SET run_claimed_until='2000-01-01T00:00:00+00:00'")
        self.assertTrue(self.store.claim_run("task", current_token, current_claim, 60))

        self.assertFalse(self.store.save_checkpoint(
            "task", "planning", {"parsed": {}}, fingerprint="fingerprint",
            run_token=old_token, claim_token=old_claim,
        ))
        self.assertFalse(self.store.save_checkpoint(
            "task", "planning", {"parsed": {}}, fingerprint="fingerprint",
            run_token=current_token, claim_token=old_claim,
        ))
        self.assertFalse(self.store.transition(
            "task", TraceEvent(1, TaskState.PLANNING, "old worker", utc_now()),
            current_token, old_claim,
        ))
        self.assertFalse(self.store.succeed(
            "task", ReviewReport("org/repo", 1, "old", "low"),
            TraceEvent(1, TaskState.SUCCESS, "old worker", utc_now()),
            current_token, old_claim,
        ))
        self.assertTrue(self.store.transition(
            "task", TraceEvent(1, TaskState.PLANNING, "current worker", utc_now()),
            current_token, current_claim,
        ))
        self.assertNotIn("planning", self.store.load_checkpoints("task"))
