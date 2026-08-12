import os
import tempfile
import unittest

from evoagent.memory import MemoryManager
from evoagent.evaluation_harness import memory_comparison
from evoagent.store import TaskStore


class MemoryLifecycleTests(unittest.TestCase):
    def setUp(self):
        handle, self.path = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        self.store = TaskStore(self.path)
        self.memory = MemoryManager(self.store, promotion_support_threshold=2)

    def tearDown(self):
        os.unlink(self.path)

    def _finding(self, line=10, path="payments/repository.py"):
        return {"rule_id": "SEC-SQL-CONCAT", "severity": "high", "path": path,
                "line": line, "evidence": "query('select ' + user_id)",
                "confidence": .9, "evidence_refs": ["E01"],
                "related_symbols": ["PaymentRepository.query"]}

    def test_case_a_working_isolated_by_agent_and_shard(self):
        for agent, shard, question in (("security", "S1", "sql?"), ("correctness", "S1", "null?"), ("security", "S2", "auth?")):
            self.memory.remember_working_state("t", "repo", "task", agent, shard, {"open_questions": [question]})
        self.assertEqual(["sql?"], self.memory.load_working_state("t", "repo", "task", "security", "S1")["metadata"]["open_questions"])
        self.assertEqual(["null?"], self.memory.load_working_state("t", "repo", "task", "correctness", "S1")["metadata"]["open_questions"])
        self.assertEqual(["auth?"], self.memory.load_working_state("t", "repo", "task", "security", "S2")["metadata"]["open_questions"])

    def test_case_b_consolidation_releases_working_but_keeps_episode(self):
        self.memory.remember_working_state("t", "repo", "task", "security", "S1", {})
        self.memory.remember_finding("t", "repo", "task", self._finding(), True, pr_number=1, source_sha="A")
        self.memory.consolidate_task("t", "repo", "task", {"shard_count": 1})
        self.assertIsNone(self.memory.load_working_state("t", "repo", "task", "security", "S1"))
        self.assertTrue(self.store.list_agent_memories("t", "repo", ("episodic",), 10))

    def test_case_c_finding_lifecycle_is_new_open_fixed(self):
        finding = self._finding()
        self.memory.remember_finding("t", "repo", "a", finding, True, pr_number=7, source_sha="A")
        self.assertEqual(1, len(self.memory.compute_finding_delta("t", "repo", 7, "A", [finding])["new"]))
        self.memory.remember_finding("t", "repo", "b", finding, True, pr_number=7, source_sha="B")
        delta = self.memory.compute_finding_delta("t", "repo", 7, "B", [finding])
        self.assertEqual(1, len(delta["open"]))
        fixed = self.memory.compute_finding_delta("t", "repo", 7, "C", [])
        self.assertEqual(1, len(fixed["fixed"]))

    def test_case_d_fuzzy_identity_handles_line_drift(self):
        self.memory.remember_finding("t", "repo", "a", self._finding(10), True, pr_number=7, source_sha="A")
        self.memory.remember_finding("t", "repo", "b", self._finding(30), True, pr_number=7, source_sha="B")
        episodes = [item for item in self.store.list_agent_memories("t", "repo", ("episodic",), 10)
                    if (item["metadata"] or {}).get("episode_type") == "finding"]
        self.assertEqual(1, len(episodes))
        self.assertEqual("open", episodes[0]["metadata"]["status"])

    def test_case_e_semantic_scope_does_not_leak_to_auth(self):
        self.memory.remember_feedback("t", "repo", "task", "false_positive", self._finding(),
                                      "This wrapper parameter-binds the payment query.")
        recalled = self.memory.recall("t", "repo", "auth login", scope={"files": ["auth/login.py"], "risk_domains": ["security"]})
        self.assertEqual([], recalled)

    def test_case_f_semantic_changes_to_symbol_mark_knowledge_stale(self):
        self.memory.remember_feedback("t", "repo", "task", "false_positive", self._finding(), "Bound parameters.", "A")
        self.memory.mark_stale_memories("t", "repo", symbols=["PaymentRepository.query"], source_sha="B")
        semantic = self.store.list_agent_memories("t", "repo", ("semantic",), 10)[0]
        self.assertEqual("needs_revalidation", semantic["metadata"]["status"])

    def test_staleness_honors_path_pattern_scope(self):
        self.memory.remember("t", "repo", "semantic", "knowledge", "payment query convention", {
            "path_patterns": ["payments/*.py"], "confidence": .8, "status": "active"}, importance=.8)
        self.memory.mark_stale_memories("t", "repo", paths=["payments/repository.py"], source_sha="B")
        semantic = self.store.list_agent_memories("t", "repo", ("semantic",), 10)[0]
        self.assertEqual("needs_revalidation", semantic["metadata"]["status"])

    def test_case_g_only_verified_repeated_episode_promotes(self):
        self.memory.remember_working_state("t", "repo", "task", "security", "S1", {"open_questions": ["maybe SQL"]})
        self.assertIsNone(self.memory.promote_episode("t", "repo", "not-an-episode"))
        self.memory.remember_finding("t", "repo", "a", self._finding(), True, pr_number=7, source_sha="A")
        self.assertEqual([], self.store.list_agent_memories("t", "repo", ("semantic",), 10))
        self.memory.remember_finding("t", "repo", "b", self._finding(), True, pr_number=7, source_sha="B")
        self.assertEqual("verified_episode", self.store.list_agent_memories("t", "repo", ("semantic",), 10)[0]["kind"])

    def test_case_h_agent_domain_changes_assignment_recall(self):
        self.memory.remember("t", "repo", "semantic", "knowledge", "security payment SQL", {
            "risk_domain": "security", "path_patterns": ["payments/*.py"], "confidence": .8, "status": "active"}, importance=.8)
        self.memory.remember("t", "repo", "semantic", "knowledge", "reliability payment retry", {
            "risk_domain": "reliability", "path_patterns": ["payments/*.py"], "confidence": .8, "status": "active"}, importance=.8)
        security = self.memory.recall_for_assignment("t", "repo", "task", "security", "S1", ["payments/repository.py"], [], ["security"], "security")
        reliability = self.memory.recall_for_assignment("t", "repo", "task", "reliability", "S1", ["payments/repository.py"], [], ["reliability"], "reliability")
        self.assertIn("security payment SQL", security[0]["content"])
        self.assertIn("reliability payment retry", reliability[0]["content"])

    def test_case_i_high_importance_zero_overlap_is_not_recalled(self):
        self.memory.remember("t", "repo", "semantic", "knowledge", "legacy compiler internals", {
            "path_patterns": ["compiler/*.c"], "confidence": 1.0, "status": "active"}, importance=1.0)
        recalled = self.memory.recall_for_assignment("t", "repo", "task", "security", "S1", ["payments/repository.py"], [], ["security"], "payment SQL")
        self.assertEqual([], recalled)

    def test_memory_evaluation_compares_legacy_and_scoped_recall(self):
        useful = self.memory.remember_feedback("t", "repo", "old-task", "false_positive", self._finding(), "payment wrapper binds values")
        self.memory.remember("t", "repo", "semantic", "knowledge", "compiler implementation trivia", {
            "path_patterns": ["compiler/*.c"], "confidence": 1.0, "status": "active"}, importance=1.0)
        report = memory_comparison(self.memory, [{
            "tenant_id": "t", "repository": "repo", "task_id": "new-task", "agent": "security", "shard_id": "S1",
            "files": ["payments/repository.py"], "symbols": ["PaymentRepository.query"],
            "risk_domains": ["security"], "objective": "payment SQL", "useful_memory_ids": [useful["id"]],
            "expected_lifecycle": "open", "observed_lifecycle": "open",
        }])
        self.assertGreater(report["new"]["recall_precision"], report["baseline"]["recall_precision"])
        self.assertIn("average_memory_tokens_injected_per_agent_shard", report["new"])


if __name__ == "__main__":
    unittest.main()
