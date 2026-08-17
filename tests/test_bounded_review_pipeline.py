import unittest

from evoagent.adaptive import SharedAgentBudget
from evoagent.agents import MultiAgentCoordinator
from evoagent.context.budget_policy import BudgetPolicy
from evoagent.context.pr_map import build_pr_context_map
from evoagent.context.risk_priority import DiffRiskScanner
from evoagent.context.shard_planner import ShardPlanner
from evoagent.context.retrieval import InMemorySnapshotProvider
from evoagent.diff_parser import parse_unified_diff
from evoagent.models import Finding, Severity
from evoagent.auditor_agent import AuditStopPolicy
from evoagent.reviewer import Reviewer
from evoagent.verifier_agent import VerifierAgentNode


def finding(path, line, rule="CWE-863"):
    return Finding(rule, Severity.HIGH, "risk", "authorization bypass", path, line,
                   "return allowed", "fix", "test", .9)


class _Primary(Reviewer):
    execution_kind = "agent"
    agent_role = "primary"
    name = "primary"

    def __init__(self, findings=None):
        self.findings = list(findings or [])
        self.tool_calls = 0

    def review(self, _diff, _parsed):
        return []

    def agent_step(self, state):
        if state.get("cross_shard") and not state.get("observations"):
            self.tool_calls += 1
            return {"action": "tool", "tool": "find_symbol", "arguments": {"symbol": "service"}}
        return {"action": "final", "findings": list(self.findings),
                "_usage": {"input_tokens": 1, "output_tokens": 1, "usage_source": "provider"}}


class _Auditor(Reviewer):
    execution_kind = "agent"
    agent_role = "auditor"
    name = "auditor"

    def __init__(self):
        self.rounds = 0
        self.assignments = []

    def review(self, _diff, _parsed):
        return []

    def agent_step(self, state):
        reason = (state.get("assignment") or {}).get("reason")
        self.assignments.append(reason)
        if reason != "reverse-audit":
            raise AssertionError("auditor received a non-audit assignment")
        self.rounds += 1
        return {"action": "final", "findings": [],
                "_usage": {"input_tokens": 1, "output_tokens": 1, "usage_source": "provider"}}


class _Verifier(Reviewer):
    execution_kind = "agent"
    agent_role = "verifier"
    name = "verifier"

    def __init__(self):
        self.assignments = []

    def review(self, _diff, _parsed):
        return []

    def agent_step(self, state):
        assignment = state.get("assignment") or {}
        self.assignments.append(assignment.get("reason"))
        self.assert_verifier_assignment(assignment)
        import re
        claims = re.findall(r'"claim_id"\s*:\s*"([^"]+)"', str(state.get("managed_context", "")))
        if not state.get("observations"):
            return {"action": "tool", "tool": "read_file",
                    "arguments": {"path": "src/a.py", "start_line": 1, "end_line": 4}}
        return {"action": "final", "output": [
            {"claim_id": item, "verdict": "support", "evidence_refs": ["R1"]} for item in claims
        ], "_usage": {"input_tokens": 1, "output_tokens": 1, "usage_source": "provider"}}

    @staticmethod
    def assert_verifier_assignment(assignment):
        if assignment.get("reason") != "verify-findings":
            raise AssertionError("verifier received a non-verification assignment")


class _ToolUntilExhausted(_Primary):
    def agent_step(self, _state):
        return {"action": "tool", "tool": "read_diff", "arguments": {"path": "src/a.py"}}


class _Security(_Primary):
    agent_role = "security"
    name = "security"


class _Reliability(_Primary):
    agent_role = "reliability"
    name = "reliability"


class _CrossTracer(Reviewer):
    execution_kind = "agent"
    agent_role = "cross_shard"
    name = "cross-tracer"

    def __init__(self):
        self.calls = 0
        self.contexts = []
        self.feedback = []

    def review(self, _diff, _parsed):
        return []

    def agent_step(self, state):
        self.calls += 1
        self.contexts.append(str(state.get("managed_context", "")))
        self.feedback.extend(state.get("feedback") or [])
        if not state.get("observations"):
            return {"action": "tool", "tool": "find_references",
                    "arguments": {"symbol": "service"}}
        return {"action": "final", "findings": [],
                "_usage": {"input_tokens": 1, "output_tokens": 1, "usage_source": "provider"}}


class _AuditorWithNewFinding(_Auditor):
    def __init__(self):
        super().__init__()
        self.emitted = False

    def agent_step(self, state):
        if not self.emitted:
            self.emitted = True
            return {"action": "final", "findings": [finding("src/a.py", 2, "CWE-628")]}
        return super().agent_step(state)


class BoundedReviewPipelineTests(unittest.TestCase):
    def test_module_first_shards_and_secondary_400_line_split_cover_all_changes(self):
        body = "".join("+    value_%03d = value\n" % index for index in range(460))
        diff = "--- a/src/billing/service.py\n+++ b/src/billing/service.py\n@@ -1 +1,462 @@\n+def first(value):\n" + body + "+    return value\n"
        parsed = parse_unified_diff(diff)
        shards = ShardPlanner(changed_line_threshold=1).plan(diff, parsed, build_pr_context_map(diff, parsed))
        self.assertGreaterEqual(len(shards), 2)
        self.assertEqual({"src/billing/service.py"}, {path for shard in shards for path in shard.files})
        self.assertEqual(len(parsed.added_lines), sum(len(shard.parsed.added_lines) for shard in shards))

    def test_hunk_risk_priority_is_shard_local_and_does_not_change_expert_coverage(self):
        diff = (
            "--- a/src/auth.py\n+++ b/src/auth.py\n@@ -10,2 +10 @@\n"
            "-if check_access(user):\n-return allowed\n+return allowed\n"
            "@@ -30 +29 @@\n+await retry()\n"
        )
        parsed = parse_unified_diff(diff)
        scanner = DiffRiskScanner()
        risk_hunks = scanner.scan(diff)
        self.assertEqual(["removed_guard", "security"], risk_hunks[0].tags[:2])
        repair_diff = scanner.select(diff, [risk_hunks[1].hunk_id])
        self.assertIn("await retry()", repair_diff)
        self.assertNotIn("check_access", repair_diff)
        coordinator = MultiAgentCoordinator(
            [_Primary(), _Security(), _Reliability()], review_mode="adaptive_multi_agent",
            review_pipeline="bounded-v2", agent_budget={"max_agent_runs": 12, "max_llm_calls": 30,
                "max_tool_calls": 30, "max_input_tokens": 60000, "max_output_tokens": 12000},
        )
        coordinator.review_with_context("priority", diff, parsed, source_sha="sha")
        summary = coordinator.collaboration_summary("priority")
        self.assertEqual({"primary", "security", "reliability"},
                         {item["agent"] for item in summary["agents"]})
        receipt = next(item for item in summary["coverage_receipts"] if item["assignment_id"].endswith("-full"))
        self.assertEqual(receipt["required_hunks"], receipt["covered_hunks"])
        self.assertFalse(receipt["uncovered_hunks"])

    def test_size_policy_and_tail_reserve_are_bounded(self):
        policy = BudgetPolicy()
        small, large = policy.for_shard(1), policy.for_shard(10_000)
        self.assertLess(small.tool_budget, large.tool_budget)
        self.assertEqual(policy.TOOL_MAX, large.tool_budget)
        budget = SharedAgentBudget(max_agent_runs=4, tail_reserve={"agent_runs": 2})
        self.assertTrue(budget.reserve("agent_runs", stage="review"))
        self.assertTrue(budget.reserve("agent_runs", stage="review"))
        self.assertFalse(budget.reserve("agent_runs", stage="review"))
        self.assertTrue(budget.reserve("agent_runs", stage="tail"))

    def test_independent_verifier_and_auditor_stage_contracts(self):
        batches = VerifierAgentNode(8).batches([[index] for index in range(9)])
        self.assertEqual([8, 1], [len(item.claims) for item in batches])
        policy = AuditStopPolicy(max_rounds=5, consecutive_dry_rounds=2)
        self.assertFalse(policy.should_stop(1, 1))
        self.assertTrue(policy.should_stop(2, 2))

    def test_cross_shard_tracer_runs_a_repository_tool_loop(self):
        diff = (
            "--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n-old\n+def service():\n"
            "--- a/src/b.py\n+++ b/src/b.py\n@@ -1 +1 @@\n-old\n+service()\n"
        )
        primary = _Primary()
        coordinator = MultiAgentCoordinator(
            [primary], review_mode="adaptive_multi_agent", review_pipeline="bounded-v2",
            shard_file_threshold=2, shard_changed_line_threshold=9999, max_workers=1,
            snapshot_provider=InMemorySnapshotProvider({"src/a.py": "def service():\n    return 1\n", "src/b.py": "service()\n"}),
            agent_budget={"max_agent_runs": 8, "max_llm_calls": 30, "max_tool_calls": 30,
                          "max_input_tokens": 60000, "max_output_tokens": 12000},
        )
        coordinator.review_with_context("trace", diff, parse_unified_diff(diff), source_sha="sha")
        summary = coordinator.collaboration_summary("trace")
        self.assertGreater(primary.tool_calls, 0)
        self.assertGreater(summary["cross_shard_execution"]["repo_tool_calls"], 0)

    def test_dedicated_tracer_runs_with_specialists_but_without_their_findings(self):
        diff = (
            "--- a/src/api.py\n+++ b/src/api.py\n@@ -1 +1 @@\n-old\n+def service():\n"
            "--- a/src/client.py\n+++ b/src/client.py\n@@ -1 +1 @@\n-old\n+service()\n"
        )
        primary = _Primary([Finding(
            "CWE-863", Severity.HIGH, "primary-only-marker", "primary-only-marker",
            "src/api.py", 1, "def service():", "fix", "test", .9,
        )])
        tracer = _CrossTracer()
        coordinator = MultiAgentCoordinator(
            [primary, _Security(), _Reliability(), tracer], review_mode="adaptive_multi_agent",
            review_pipeline="bounded-v2", shard_file_threshold=2,
            shard_changed_line_threshold=9999, max_workers=4,
            snapshot_provider=InMemorySnapshotProvider({
                "src/api.py": "def service():\n    return 1\n",
                "src/client.py": "service()\n",
            }),
            agent_budget={"max_agent_runs": 16, "max_llm_calls": 40, "max_tool_calls": 40,
                          "max_input_tokens": 60000, "max_output_tokens": 12000},
        )
        coordinator.review_with_context("parallel-trace", diff, parse_unified_diff(diff), source_sha="sha")
        summary = coordinator.collaboration_summary("parallel-trace")
        self.assertGreaterEqual(tracer.calls, 2)
        self.assertTrue(any("Structured PR relation seeds" in item for item in tracer.feedback))
        self.assertTrue(all("primary-only-marker" not in item for item in tracer.contexts))
        self.assertGreater(summary["cross_shard_execution"]["repo_tool_calls"], 0)

    def test_verifier_batches_eight_and_auditor_stops_after_two_dry_rounds(self):
        diff = "--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1,9 @@\n-old\n" + "".join(
            "+return allowed_%d\n" % index for index in range(9)
        )
        primary = _Primary([finding("src/a.py", index + 1) for index in range(9)])
        verifier, auditor = _Verifier(), _Auditor()
        coordinator = MultiAgentCoordinator(
            [primary, verifier, auditor], review_mode="adaptive_multi_agent", review_pipeline="bounded-v2", max_workers=4,
            snapshot_provider=InMemorySnapshotProvider({"src/a.py": "\n".join("return allowed_%d" % i for i in range(9))}),
            agent_budget={"max_agent_runs": 30, "max_llm_calls": 80, "max_tool_calls": 80,
                          "max_input_tokens": 60000, "max_output_tokens": 12000},
        )
        coordinator.review_with_context("verify", diff, parse_unified_diff(diff), source_sha="sha")
        summary = coordinator.collaboration_summary("verify")
        self.assertEqual([8, 1], [item["size"] for item in summary["verifier_batches"]])
        self.assertEqual(2, len(summary["auditor_rounds"]))
        self.assertTrue(all(item["dry"] for item in summary["auditor_rounds"]))
        self.assertTrue(summary["coverage_receipts"])
        self.assertTrue(verifier.assignments)
        self.assertTrue(all(reason == "verify-findings" for reason in verifier.assignments))
        self.assertTrue(auditor.assignments)
        self.assertTrue(all(reason == "reverse-audit" for reason in auditor.assignments))

    def test_budget_exhaustion_returns_receipt_instead_of_raising(self):
        diff = "--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n-old\n+return allowed\n"
        coordinator = MultiAgentCoordinator(
            [_ToolUntilExhausted()], review_mode="adaptive_multi_agent", review_pipeline="bounded-v2",
            agent_loop_max_steps=8, agent_budget={"max_agent_runs": 8, "max_llm_calls": 30,
                "max_tool_calls": 30, "max_input_tokens": 60000, "max_output_tokens": 12000},
        )
        coordinator.review_with_context("receipt", diff, parse_unified_diff(diff), source_sha="sha")
        receipt = coordinator.collaboration_summary("receipt")["coverage_receipts"][0]
        self.assertEqual("budget-exhausted", receipt["status"])
        self.assertTrue(receipt["budget_gaps"])
        summary = coordinator.collaboration_summary("receipt")
        self.assertTrue(receipt["uncovered_hunks"])
        self.assertEqual("needs-human-review", summary["verdict_cap"])
        self.assertEqual(receipt["uncovered_hunks"], summary["coverage_repairs"][0]["required_hunk_ids"])

    def test_auditor_new_finding_is_reverified_before_deterministic_decision(self):
        diff = "--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1,2 @@\n-old\n+return allowed\n+service()\n"
        primary, verifier, auditor = _Primary([finding("src/a.py", 1)]), _Verifier(), _AuditorWithNewFinding()
        coordinator = MultiAgentCoordinator(
            [primary, verifier, auditor], review_mode="adaptive_multi_agent", review_pipeline="bounded-v2",
            snapshot_provider=InMemorySnapshotProvider({"src/a.py": "return allowed\nservice()\n"}),
            agent_budget={"max_agent_runs": 30, "max_llm_calls": 80, "max_tool_calls": 80,
                          "max_input_tokens": 60000, "max_output_tokens": 12000},
        )
        coordinator.review_with_context("reverify", diff, parse_unified_diff(diff), source_sha="sha")
        summary = coordinator.collaboration_summary("reverify")
        self.assertEqual([2], [item["size"] for item in summary["verifier_batches"]])
        self.assertEqual(2, len(summary["challenges"]))


if __name__ == "__main__":
    unittest.main()
