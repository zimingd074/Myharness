import unittest

from evoagent.agents import MultiAgentCoordinator
from evoagent.context.loop_context import LoopContext
from evoagent.context.pr_map import build_pr_context_map
from evoagent.context.reducers import reduce_tool_result
from evoagent.context.retrieval import InMemorySnapshotProvider, RepositoryRetrieval
from evoagent.context.budget import ContextBudget
from evoagent.context.shard_planner import ShardPlanner
from evoagent.context_manager import ContextManager
from evoagent.diff_parser import parse_unified_diff
from evoagent.models import Finding, Severity
from evoagent.reviewer import Reviewer
from evoagent.runtime import AgentLoop, AgentTool, ToolRegistry
from evoagent.evaluation_benchmark import (
    coverage_ab_reviewer, generate_coverage_ab_cases, generate_large_pr_context_cases,
    generate_broadened_context_cases,
)
from evoagent.evaluation_harness import EndToEndEvaluationHarness


def multi_file_diff(count=14):
    parts = []
    for index in range(count):
        parts.append("--- a/src/module_%02d.py\n+++ b/src/module_%02d.py\n@@ -1 +1,2 @@\n-old\n+def changed_%02d(value):\n+    return value\n" % (index, index, index))
    return "".join(parts)


class CoverageContextTests(unittest.TestCase):
    class _CrossShardAgent(Reviewer):
        name = "cross-shard-agent"
        domains = ("correctness",)

        def review(self, diff, parsed):
            return []

        def agent_step(self, state):
            if state.get("cross_shard"):
                return {"action": "final", "findings": [Finding(
                    "XSHARD-SIGNATURE", Severity.HIGH, "caller keeps stale signature",
                    "The changed caller still supplies a removed parameter.", "src/caller.py", 2,
                    "service(legacy=True)", "Update the caller to the new signature.",
                    "Add an integration test for the caller.", .9,
                )]}
            return {"action": "final", "findings": []}

    class _FailingReviewer(Reviewer):
        name = "failing-agent"
        domains = ("correctness",)
        def review(self, diff, parsed):
            raise RuntimeError("intentional shard failure")

    class _NoopReviewer(Reviewer):
        name = "noop-agent"
        domains = ("reliability",)
        def review(self, diff, parsed):
            return []

    def test_pr_map_and_shards_cover_every_changed_file(self):
        diff = multi_file_diff()
        parsed = parse_unified_diff(diff)
        pr_map = build_pr_context_map(diff, parsed)
        shards = ShardPlanner(diff_budget_tokens=80, file_threshold=2).plan(diff, parsed, pr_map)
        self.assertEqual(set(parsed.files), {path for shard in shards for path in shard.files})
        self.assertTrue(all(item.symbols for item in pr_map.files))
        self.assertGreater(len(shards), 1)

    def test_large_hunk_tracks_omitted_changed_lines_with_honest_marker(self):
        diff = "--- a/app.py\n+++ b/app.py\n@@ -1 +1,400 @@\n" + "".join(
            "+value_%d = %d\n" % (index, index) for index in range(400)
        )
        bundle = ContextManager(512, 64).build(diff, {"objective": "review"})
        self.assertGreater(bundle.omitted_added_lines, 0)
        self.assertIn("lower-priority diff content omitted", bundle.text)

    def test_snapshot_retrieval_is_bounded_and_rejects_escape(self):
        retrieval = RepositoryRetrieval(InMemorySnapshotProvider({"src/a.py": "one\ntwo\nneedle\n"}))
        self.assertEqual(3, retrieval.read_file("src/a.py", 1, 3)["end_line"])
        self.assertEqual(3, retrieval.grep_repo("needle")["hits"][0]["line"])
        with self.assertRaises(ValueError):
            retrieval.read_file("../secret", 1, 2)
        with self.assertRaises(ValueError):
            retrieval.read_file("src/a.py", 1, 500)

    def test_loop_compaction_keeps_pinned_evidence(self):
        context = LoopContext(active_rounds=2)
        for step in range(1, 9):
            context.add_round(step, {"action": "tool"}, {"tool": "lookup", "ok": True, "result": "evidence-%d" % step})
            if step == 1:
                context.pin(["R1"], "finding-1")
            context.compact()
        rendered = context.render()
        self.assertGreater(rendered["compaction_count"], 0)
        self.assertEqual("finding-1", rendered["pinned_evidence"][0]["finding_ids"][0])

    def test_search_reducer_keeps_late_structured_hit_not_raw_prefix(self):
        result = {"hits": [{"path": "a.py", "line": index, "content": "hit-%d" % index} for index in range(60)]}
        rendered = reduce_tool_result("grep_repo", result, 4000)
        self.assertIn('"hits"', rendered)
        self.assertIn("hit-19", rendered)
        self.assertNotIn("hit-20", rendered)

    def test_ten_large_pr_fixtures_are_multifile_and_shardable(self):
        cases = generate_large_pr_context_cases()
        self.assertEqual(10, len(cases))
        for case in cases:
            parsed = parse_unified_diff(case["diff"])
            self.assertGreaterEqual(len(parsed.files), 13)
            self.assertGreater(len(ShardPlanner().plan(case["diff"], parsed, build_pr_context_map(case["diff"], parsed))), 1)

    def test_broadened_fixture_set_has_business_cross_file_and_clean_cases(self):
        cases = generate_broadened_context_cases()
        self.assertEqual(10, len(cases))
        self.assertTrue(any(not item["expected_findings"] for item in cases))
        self.assertTrue(any(item["expected_findings"] and item["expected_findings"][0]["cwe"] == "CWE-628" for item in cases))
        self.assertTrue(all(len(parse_unified_diff(item["diff"]).files) >= 13 for item in cases))
        for item in cases:
            parsed = parse_unified_diff(item["diff"])
            self.assertGreater(len(ShardPlanner().plan(item["diff"], parsed, build_pr_context_map(item["diff"], parsed))), 1)

    def test_coverage_ab_recovers_low_keyword_business_defect(self):
        cases = generate_coverage_ab_cases()
        legacy = EndToEndEvaluationHarness().run(coverage_ab_reviewer("legacy"), cases)
        coverage = EndToEndEvaluationHarness().run(coverage_ab_reviewer("coverage-first"), cases)
        self.assertEqual(0.0, legacy["metrics"]["recall"])
        self.assertEqual(1.0, coverage["metrics"]["recall"])
        self.assertLess(coverage["metrics"]["max_input_tokens"], legacy["metrics"]["max_input_tokens"])

    def test_compose_uses_frozen_active_compressed_and_retrieved_layers(self):
        manager = ContextManager(2000, 300)
        bundle = manager.build("--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+new\n")
        rendered = manager.compose(
            bundle, {"agent": "a", "objective": "review"},
            frozen_context={"pr_map": {"files": [{"path": "a.py"}]}},
            loop_context={"active_rounds": [{"step": 2, "action": {"action": "tool"}, "observation": {"ok": True}}],
                          "compressed_history": {"open_questions": ["is caller stale?"]},
                          "pinned_evidence": [{"id": "R1", "path": "a.py", "line": 1}]},
            retrieved_context=[{"tool": "read_file", "result": "caller context"}],
        ).text
        for label in ("FROZEN_CONTEXT", "ACTIVE_ROUND", "COMPRESSED_HISTORY", "PINNED_EVIDENCE", "RETRIEVED_CONTEXT"):
            self.assertIn(label, rendered)

    def test_cross_shard_pass_enters_normal_verification_flow(self):
        diff = (
            "--- a/src/api.py\n+++ b/src/api.py\n@@ -1 +1,2 @@\n-def service(legacy):\n+def service():\n+    return 1\n"
            "--- a/src/caller.py\n+++ b/src/caller.py\n@@ -1 +1,2 @@\n-old\n+service(legacy=True)\n+pass\n"
        )
        parsed = parse_unified_diff(diff)
        coordinator = MultiAgentCoordinator([self._CrossShardAgent()], agent_retries=0,
            shard_file_threshold=2, shard_changed_line_threshold=9999, max_workers=1)
        coordinator.review(diff, parsed)
        summary = coordinator.last_collaboration_summary()
        self.assertEqual(1, summary["cross_shard_findings"])
        self.assertGreaterEqual(summary["proposed_findings"], 1)

    def test_failed_responsible_shard_is_a_visible_coverage_gap(self):
        parsed = parse_unified_diff(multi_file_diff(4))
        coordinator = MultiAgentCoordinator(
            [self._FailingReviewer(), self._NoopReviewer()], fallback_agent=self._FailingReviewer(),
            agent_retries=0, shard_file_threshold=2, shard_changed_line_threshold=9999, max_workers=1,
        )
        coordinator.review(multi_file_diff(4), parsed)
        self.assertTrue(any("responsible shard review" in gap["reason"]
                            for gap in coordinator.last_collaboration_summary()["coverage_gaps"]))

    def test_agent_loop_keeps_tail_error_from_long_tool_result(self):
        calls = iter([
            {"action": "tool", "tool": "logs", "arguments": {}},
            {"action": "final", "findings": []},
        ])
        registry = ToolRegistry([AgentTool("logs", "logs", {"type": "object", "properties": {}, "additionalProperties": False},
            lambda: "prefix\n" + ("x" * 5000) + "\nERROR: tail signal")])
        result = AgentLoop(2, 10).run(lambda state: next(calls), registry, {})
        self.assertIn("ERROR: tail signal", result.observations[0]["result"])

    def test_read_diff_payload_obeys_byte_limit_and_paginates(self):
        diff = "--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1,80 @@\n" + "".join(
            "+%s\n" % ("x" * 500) for _ in range(80)
        )
        page = RepositoryRetrieval(diff=diff).read_diff("src/a.py", 0, 400)
        self.assertLessEqual(len("\n".join(page["lines"]).encode("utf-8")), 16080)
        self.assertTrue(page["truncated"])
        self.assertIsNotNone(page["next_cursor"])

    def test_eight_round_explicit_evidence_reference_survives_compaction(self):
        calls = []
        for _ in range(7):
            calls.append({"action": "tool", "tool": "read_file", "arguments": {}})
        calls.append({"action": "final", "findings": [Finding(
            "PINNED", Severity.HIGH, "pinned", "evidence remains traceable", "src/a.py", 2,
            "value", "fix", "test", .9, evidence_refs=["R1"],
        )]})
        registry = ToolRegistry([AgentTool(
            "read_file", "bounded", {"type": "object", "properties": {}, "additionalProperties": False},
            lambda: {"path": "src/a.py", "line": 2, "content": "important evidence"},
        )])
        result = AgentLoop(8, 10, active_rounds=2).run(lambda state: calls.pop(0), registry, {})
        pinned = result.loop_context["pinned_evidence"]
        self.assertEqual("R1", pinned[0]["id"])
        self.assertGreaterEqual(result.loop_context["compaction_count"], 1)

    def test_activation_policy_exposes_owner_and_supplemental_matrix(self):
        class Generalist(Reviewer):
            name = "llm-generalist"
            domains = ("correctness",)
            def review(self, diff, parsed): return []
            def agent_step(self, state): return {"action": "final", "findings": []}
        class Security(Reviewer):
            name = "security-specialist"
            domains = ("security",)
            def review(self, diff, parsed): return []
        diff = multi_file_diff(4)
        parsed = parse_unified_diff(diff)
        summaries = {}
        for policy in ("directed", "hybrid", "all"):
            coordinator = MultiAgentCoordinator(
                [Generalist(), Security()], specialist_activation=policy,
                shard_file_threshold=2, shard_changed_line_threshold=9999, max_workers=1,
            )
            coordinator.review(diff, parsed)
            summaries[policy] = coordinator.last_collaboration_summary()
        self.assertTrue(all(item["coverage_owner"] == "llm-generalist"
                            for item in summaries["hybrid"]["shard_assignments"]))
        self.assertGreater(
            summaries["all"]["planned_assignments"], summaries["hybrid"]["planned_assignments"]
        )
        self.assertTrue(summaries["hybrid"]["review_complete"])

    def test_budget_uses_minimum_quotas_and_elastic_shared_pool(self):
        budget = ContextBudget(max_tokens=12000, reserved_tokens=2500)
        small = budget.allocations(1000)
        full = budget.allocations(9500)
        self.assertGreater(small["runtime"], full["runtime"])
        self.assertGreater(small["shared"], 0)
        self.assertTrue(budget.should_compact(7200))


if __name__ == "__main__":
    unittest.main()
