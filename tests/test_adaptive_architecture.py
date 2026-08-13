import threading
import unittest
import io
import json
import urllib.error
import re
from unittest import mock
from dataclasses import FrozenInstanceError

from evoagent.adaptive import (
    AdaptiveDecisionPolicy, AgentSpec, Challenge, Claim, Evidence,
    DeterministicRiskRouter, SharedAgentBudget, evidence_from_record,
)
from evoagent.agents import MultiAgentCoordinator
from evoagent.context.retrieval import InMemorySnapshotProvider
from evoagent.diff_parser import parse_unified_diff
from evoagent.models import Finding, Severity
from evoagent.reviewer import OpenAICompatibleReviewer, Reviewer, SecurityRuleReviewer


def finding(rule="CWE-863", severity=Severity.HIGH, path="a.py", line=1,
            evidence="return allowed", explanation="authorization bypass"):
    return Finding(rule, severity, "risk", explanation, path, line, evidence,
                   "fix", "test", .9)


class ProbeAgent(Reviewer):
    execution_kind = "agent"

    def __init__(self, role, output=None):
        self.agent_role = role
        self.name = "probe-" + role
        self.provider = "probe"
        self.model = "probe-v1"
        self.prompt_hash = role
        self.output = list(output or [])
        self.contexts = []

    def review(self, diff, parsed):
        return []

    def agent_step(self, state):
        self.contexts.append(str(state.get("managed_context", "")))
        return {"action": "final", "findings": list(self.output), "_usage": {
            "input_tokens": 10, "output_tokens": 5, "usage_source": "provider",
        }}


class ProtocolAgent(ProbeAgent):
    def __init__(self, role, output=None, challenge_verdict="insufficient",
                 revision_action="retain"):
        super().__init__(role, output)
        self.challenge_verdict = challenge_verdict
        self.revision_action = revision_action
        self.revision_calls = 0

    def agent_step(self, state):
        reason = str((state.get("assignment") or {}).get("reason", ""))
        context = str(state.get("managed_context", ""))
        claim_ids = list(dict.fromkeys(re.findall(r'"claim_id"\s*:\s*"([^"]+)"', context)))
        if reason == "blind-challenge":
            if not state.get("observations"):
                return {"action": "tool", "tool": "read_file",
                        "arguments": {"path": "a.py", "start_line": 1, "end_line": 5},
                        "reason": "collect counterevidence"}
            return {"action": "final", "output": [
                {"claim_id": claim_id, "verdict": self.challenge_verdict,
                 "evidence_refs": ["R1"], "counterexample": "guard remains"}
                for claim_id in claim_ids
            ], "_usage": {"input_tokens": 10, "output_tokens": 5,
                            "usage_source": "provider"}}
        if reason == "challenge-requested-revision":
            self.revision_calls += 1
            return {"action": "final", "output": [
                {"claim_id": claim_id, "action": self.revision_action,
                 "evidence_refs": []} for claim_id in claim_ids
            ], "_usage": {"input_tokens": 10, "output_tokens": 5,
                            "usage_source": "provider"}}
        return super().agent_step(state)


class ToolUntilBudgetAgent(ProbeAgent):
    def __init__(self):
        super().__init__("primary")
        self.calls = 0

    def agent_step(self, state):
        self.calls += 1
        return {"action": "tool", "tool": "read_diff", "arguments": {},
                "reason": "keep reading", "_usage": {
                    "input_tokens": 10, "output_tokens": 5, "usage_source": "provider",
                }}


class AdaptiveArchitectureTests(unittest.TestCase):
    def test_provider_json_decoder_accepts_fence_but_rejects_truncation(self):
        decoded = OpenAICompatibleReviewer._decode_json_object(
            "brief prelude\n```json\n{\"action\":\"final\",\"findings\":[]}\n```"
        )
        self.assertEqual("final", decoded["action"])
        with self.assertRaises(json.JSONDecodeError):
            OpenAICompatibleReviewer._decode_json_object('{"action":"final"')
        reasoning = OpenAICompatibleReviewer._decode_model_message({
            "content": "", "reasoning_content": "analysis\n```json\n{\"findings\":[]}\n```",
        })
        self.assertEqual([], reasoning["findings"])

    def test_invalid_provider_json_gets_one_bounded_repair_with_measured_usage(self):
        reviewer = OpenAICompatibleReviewer(
            "https://model.example/v1", "secret", "model", provider="test",
        )

        class Response:
            def __init__(self, content, prompt, completion, finish):
                self.value = {
                    "choices": [{"message": {"content": content}, "finish_reason": finish}],
                    "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
                }
            def __enter__(self): return self
            def __exit__(self, *_args): return False
            def read(self): return json.dumps(self.value).encode("utf-8")

        with mock.patch("urllib.request.urlopen", side_effect=[
            Response('{"action":"final"', 10, 5, "length"),
            Response('{"action":"final","findings":[]}', 12, 3, "stop"),
        ]) as request:
            result = reviewer._request_json({"model": "model", "messages": []})
        self.assertEqual(2, request.call_count)
        self.assertEqual(["failed", "success"], [
            item["status"] for item in result["__request_attempts__"]
        ])
        self.assertEqual(22, result["__usage__"]["input_tokens"])
        self.assertEqual(8, result["__usage__"]["output_tokens"])
        self.assertEqual("provider", result["__usage__"]["usage_source"])
        self.assertEqual(1, len({
            item["logical_call_id"] for item in result["__request_attempts__"]
        }))

    def test_deepseek_requests_disable_thinking_for_compact_agent_json(self):
        reviewer = OpenAICompatibleReviewer(
            "https://api.deepseek.com", "secret", "deepseek-v4-flash",
            provider="deepseek",
        )
        captured = {}

        class Response:
            def __enter__(self): return self
            def __exit__(self, *_args): return False
            def read(self):
                return json.dumps({
                    "choices": [{"message": {"content": '{"findings":[]}'},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                }).encode("utf-8")

        def request(value, timeout=None):
            captured.update(json.loads(value.data.decode("utf-8")))
            return Response()

        with mock.patch("urllib.request.urlopen", side_effect=request):
            reviewer._request_json({"model": "deepseek-v4-flash", "messages": []})
        self.assertEqual({"type": "disabled"}, captured["thinking"])
        self.assertTrue(reviewer.disable_thinking)

    def test_blind_challenger_is_not_offered_locator_only_tools(self):
        diff = "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+return allowed\n"
        parsed = parse_unified_diff(diff)
        primary = ProtocolAgent("primary", [finding()])

        class CatalogChallenger(ProtocolAgent):
            def agent_step(self, state):
                names = {item["name"] for item in state.get("available_tools", [])}
                self.catalog = names
                return super().agent_step(state)

        challenger = CatalogChallenger("challenger")
        coordinator = MultiAgentCoordinator(
            [primary, challenger], review_mode="adaptive_multi_agent", max_workers=1,
            snapshot_provider=InMemorySnapshotProvider({"a.py": "return allowed\n"}),
        )
        coordinator.review_with_execution_context(
            "challenger-tools", diff, parsed, {"force_challenge": True}, source_sha="sha",
        )
        self.assertTrue({"read_file", "find_symbol", "find_references"}.issubset(challenger.catalog))
        self.assertTrue({"read_diff", "search_diff", "changed_line"}.isdisjoint(challenger.catalog))

    def test_stage_budget_is_terminal_and_failed_run_keeps_partial_usage(self):
        diff = "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+return allowed\n"
        parsed = parse_unified_diff(diff)
        primary = ToolUntilBudgetAgent()
        coordinator = MultiAgentCoordinator(
            [primary], review_mode="single_agent", max_workers=1,
            agent_retries=2,
            agent_budget={"max_agent_runs": 4, "max_llm_calls": 9,
                          "max_tool_calls": 8, "max_input_tokens": 24000,
                          "max_output_tokens": 4000},
        )
        coordinator.review_with_execution_context(
            "terminal-budget", diff, parsed,
            {"stage_max_llm_calls": 2}, source_sha="sha",
        )
        summary = coordinator.collaboration_summary("terminal-budget")
        self.assertEqual(2, primary.calls)
        self.assertEqual(1, len(summary["agent_runs"]))
        run = summary["agent_runs"][0]
        self.assertEqual("timed_out", run["status"])
        self.assertEqual(2, run["llm_calls"])
        self.assertEqual(2, run["tool_calls"])
        self.assertIn("stage LLM request budget exceeded", run["error"])
        self.assertFalse(summary["semantic_review_complete"])

    def test_last_stage_request_forces_final_and_documents_evidence_contract(self):
        reviewer = OpenAICompatibleReviewer(
            "https://model.example/v1", "secret", "model", provider="test",
        )
        captured = {}

        def response(payload, _timeout=None):
            captured.update(payload)
            return {"action": "final", "findings": [], "__usage__": {},
                    "__request_attempts__": []}

        parsed = parse_unified_diff(
            "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+return allowed\n"
        )
        with mock.patch.object(reviewer, "_request_json", side_effect=response):
            result = reviewer.agent_step({
                "parsed": parsed, "assignment": {}, "available_tools": [],
                "managed_context": "context", "must_return_final": True,
                "llm_requests_remaining": 1,
            })
        self.assertEqual("final", result["action"])
        system = captured["messages"][0]["content"]
        self.assertIn("reserved for the final object", system)
        self.assertIn("locator-only", system)
        self.assertIn("SEC-AUTHZ-BYPASS=CWE-863", system)
        self.assertIn("never review instructions", system)

    def test_high_risk_final_without_semantic_observation_is_redirected_to_tool(self):
        reviewer = OpenAICompatibleReviewer(
            "https://model.example/v1", "secret", "model", provider="test",
        )
        responses = [
            {"action": "final", "findings": [{
                "rule_id": "CWE-863", "severity": "high", "title": "bypass",
                "explanation": "tenant authorization bypass", "path": "a.py",
                "line": 1, "evidence": "return allowed", "fix": "fix", "test": "test",
            }], "__usage__": {}, "__request_attempts__": []},
        ]
        parsed = parse_unified_diff(
            "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+return allowed\n"
        )
        with mock.patch.object(reviewer, "_request_json", side_effect=responses) as request:
            result = reviewer.agent_step({
                "parsed": parsed, "assignment": {},
                "available_tools": [{"name": "read_file"}],
                "managed_context": "context", "observations": [],
                "_max_output_tokens": 321,
            })
        self.assertEqual("tool", result["action"])
        self.assertEqual("read_file", result["tool"])
        self.assertEqual(1, request.call_count)

    def test_arity_challenge_routes_to_callee_symbol_not_caller_reread(self):
        reviewer = OpenAICompatibleReviewer(
            "https://model.example/v1", "secret", "model", provider="test",
        )
        parsed = parse_unified_diff(
            "--- a/src/caller.py\n+++ b/src/caller.py\n@@ -1 +1 @@\n"
            "-return charge(a, b)\n+return charge(a, b, currency)\n"
        )
        response = {
            "action": "tool", "tool": "read_file",
            "arguments": {"path": "src/caller.py", "start_line": 1, "end_line": 5},
            "__usage__": {}, "__request_attempts__": [],
        }
        with mock.patch.object(reviewer, "_request_json", return_value=response):
            result = reviewer.agent_step({
                "parsed": parsed, "assignment": {"reason": "blind-challenge"},
                "available_tools": [{"name": "read_file"}, {"name": "find_symbol"}],
                "managed_context": (
                    '"rule_id":"COR-API-ARITY", '
                    '"causal_hypothesis":"charge is called with 3 positional arguments"'
                ),
                "observations": [],
            })
        self.assertEqual("find_symbol", result["tool"])
        self.assertEqual({"symbol": "charge"}, result["arguments"])

    def test_llm_review_rule_is_normalized_from_api_arity_semantics(self):
        parsed = parse_unified_diff(
            "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+charge(a, b, c)\n"
        )
        findings = OpenAICompatibleReviewer._parse_findings({"findings": [{
            "rule_id": "LLM-REVIEW", "severity": "high",
            "title": "Caller passes too many arguments",
            "explanation": "The call passes three arguments but the signature accepts two parameters.",
            "path": "a.py", "line": 1, "evidence": "charge(a, b, c)",
            "fix": "align the signature", "test": "invoke the call",
        }]}, parsed)
        self.assertEqual("COR-API-ARITY", findings[0].rule_id)

    def test_unique_exact_evidence_relocates_model_line_but_ambiguous_evidence_does_not(self):
        parsed = parse_unified_diff(
            "--- a/a.py\n+++ b/a.py\n@@ -1 +1,2 @@\n-old\n+charge(a, b, c)\n+return ok\n"
        )
        raw = {
            "rule_id": "COR-API-ARITY", "severity": "high", "title": "arity",
            "explanation": "three args for two params", "path": "a.py", "line": 99,
            "evidence": "charge(a, b, c)", "fix": "align", "test": "call",
        }
        findings = OpenAICompatibleReviewer._parse_findings({"findings": [raw]}, parsed)
        self.assertEqual(1, findings[0].line)
        duplicated = parse_unified_diff(
            "--- a/a.py\n+++ b/a.py\n@@ -1 +1,2 @@\n-old\n+same()\n+same()\n"
        )
        self.assertEqual([], OpenAICompatibleReviewer._parse_findings({
            "findings": [{**raw, "evidence": "same()"}],
        }, duplicated))

    def test_tool_observation_exposes_citable_evidence_alias_next_turn(self):
        from evoagent.context.loop_context import LoopContext
        context = LoopContext()
        observation = {"tool": "read_file", "ok": True,
                       "result": '{"path":"a.py","content":"x"}',
                       "agent": "challenger", "shard_id": "full"}
        context.add_round(1, {"action": "tool"}, observation)
        active = context.render()["active_rounds"][0]["observation"]
        self.assertEqual("R1", active["evidence_alias"])
        self.assertTrue(active["evidence_id"].startswith("E:"))
        self.assertEqual("R1", observation["evidence_alias"])

    def test_loop_context_accepts_list_shaped_tool_result(self):
        from evoagent.context.loop_context import LoopContext
        context = LoopContext()
        context.add_round(1, {"action": "tool"}, {
            "tool": "grep_repo", "ok": True,
            "result": '[{"path":"src/api.py","line":1}]',
            "agent": "challenger", "shard_id": "full",
        })
        context.pin(["R1"], "claim")
        evidence = context.render()["pinned_evidence"][0]
        self.assertEqual("src/api.py", evidence["path"])
        self.assertEqual(1, evidence["line"])

    def test_clean_challenger_must_investigate_before_final(self):
        reviewer = OpenAICompatibleReviewer(
            "https://model.example/v1", "secret", "model", provider="test",
        )
        responses = [
            {"action": "final", "challenges": [],
             "__usage__": {}, "__request_attempts__": []},
            {"action": "tool", "tool": "find_symbol", "arguments": {"symbol": "charge"},
             "__usage__": {}, "__request_attempts__": []},
        ]
        parsed = parse_unified_diff(
            "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+charge(a, b, c)\n"
        )
        with mock.patch.object(reviewer, "_request_json", side_effect=responses):
            action = reviewer.agent_step({
                "parsed": parsed, "assignment": {"reason": "blind-challenge"},
                "available_tools": [{"name": "find_symbol"}],
                "managed_context": "risk-activated clean review", "observations": [],
            })
        self.assertEqual("tool", action["action"])
        self.assertEqual("find_symbol", action["tool"])

    def test_new_claim_without_finding_gets_schema_repair(self):
        reviewer = OpenAICompatibleReviewer(
            "https://model.example/v1", "secret", "model", provider="test",
        )
        corrected_finding = {
            "rule_id": "COR-API-ARITY", "severity": "high", "title": "arity",
            "explanation": "call passes three arguments but definition accepts two",
            "path": "a.py", "line": 1, "evidence": "charge(a, b, c)",
            "fix": "align signature", "test": "call it", "confidence": .9,
            "evidence_refs": ["R1"],
        }
        responses = [
            {"action": "final", "challenges": [{
                "claim_id": "new_claim", "verdict": "new_claim",
                "evidence_refs": ["R1"], "rationale": "mismatch",
            }], "__usage__": {}, "__request_attempts__": []},
            {"action": "final", "challenges": [{
                "claim_id": "new_claim", "verdict": "new_claim",
                "evidence_refs": ["R1"], "rationale": "mismatch",
                "finding": corrected_finding,
            }], "__usage__": {}, "__request_attempts__": []},
        ]
        parsed = parse_unified_diff(
            "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+charge(a, b, c)\n"
        )
        with mock.patch.object(reviewer, "_request_json", side_effect=responses):
            action = reviewer.agent_step({
                "parsed": parsed, "assignment": {"reason": "blind-challenge"},
                "available_tools": [{"name": "find_symbol"}],
                "managed_context": "risk-activated clean review",
                "observations": [{"ok": True, "tool": "find_symbol"}],
            })
        self.assertEqual("final", action["action"])
        self.assertEqual("new_claim", action["output"][0]["verdict"])
        self.assertIsInstance(action["output"][0]["finding"], Finding)

    def test_single_symbol_hit_preserves_definition_path_in_pinned_evidence(self):
        from evoagent.context.loop_context import LoopContext
        context = LoopContext()
        context.add_round(1, {"action": "tool"}, {
            "tool": "find_symbol", "ok": True,
            "result": '{"symbol":"charge","hits":[{"path":"src/api.py","line":1}]}',
            "agent": "challenger", "shard_id": "full",
        })
        context.pin(["R1"], "claim")
        evidence = context.render()["pinned_evidence"][0]
        self.assertEqual("src/api.py", evidence["path"])
        self.assertEqual(1, evidence["line"])

    def test_tool_request_after_final_reservation_is_repaired_to_final(self):
        reviewer = OpenAICompatibleReviewer(
            "https://model.example/v1", "secret", "model", provider="test",
        )
        responses = [
            {"action": "tool", "tool": "read_file",
             "arguments": {"path": "a.py", "start_line": 1, "end_line": 5},
             "__usage__": {}, "__request_attempts__": []},
            {"action": "final", "findings": [],
             "__usage__": {}, "__request_attempts__": []},
        ]
        parsed = parse_unified_diff(
            "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+return allowed\n"
        )
        with mock.patch.object(reviewer, "_request_json", side_effect=responses):
            result = reviewer.agent_step({
                "parsed": parsed, "assignment": {}, "observations": [{
                    "ok": True, "tool": "read_file",
                }], "available_tools": [{"name": "read_file"}],
                "managed_context": "context", "must_return_final": True,
            })
        self.assertEqual("final", result["action"])
        self.assertEqual([], result["findings"])

    def test_redundant_tool_after_semantic_observation_is_repaired_to_final(self):
        reviewer = OpenAICompatibleReviewer(
            "https://model.example/v1", "secret", "model", provider="test",
        )
        responses = [
            {"action": "tool", "tool": "read_file",
             "arguments": {"path": "a.py", "start_line": 1, "end_line": 5},
             "__usage__": {}, "__request_attempts__": []},
            {"action": "final", "findings": [],
             "__usage__": {}, "__request_attempts__": []},
        ]
        parsed = parse_unified_diff(
            "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+return allowed\n"
        )
        with mock.patch.object(reviewer, "_request_json", side_effect=responses):
            result = reviewer.agent_step({
                "parsed": parsed, "assignment": {}, "observations": [{
                    "ok": True, "tool": "find_symbol",
                }], "available_tools": [{"name": "read_file"}],
                "managed_context": "context", "must_return_final": False,
            })
        self.assertEqual("final", result["action"])
        self.assertEqual([], result["findings"])

    def test_explicit_refute_is_reachable_and_rejects_claim(self):
        diff = "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+return allowed\n"
        parsed = parse_unified_diff(diff)
        primary = ProtocolAgent("primary", [finding()], revision_action="withdraw")
        challenger = ProtocolAgent("challenger", challenge_verdict="refute")
        coordinator = MultiAgentCoordinator(
            [primary, challenger], review_mode="adaptive_multi_agent", max_workers=1,
            snapshot_provider=InMemorySnapshotProvider({"a.py": "return allowed\n"}),
        )
        results = coordinator.review_with_execution_context(
            "refute", diff, parsed, {"force_challenge": True}, source_sha="sha",
        )
        summary = coordinator.collaboration_summary("refute")
        self.assertEqual([], results)
        self.assertTrue(any(item["verdict"] == "refute" for item in summary["challenges"]))
        self.assertEqual(1, primary.revision_calls)

    def test_revision_returns_to_security_proposer_not_primary(self):
        diff = "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+eval(payload)\n"
        parsed = parse_unified_diff(diff)
        primary = ProtocolAgent("primary")
        security = ProtocolAgent("security", [finding("CWE-863", path="a.py",
                                                       evidence="eval(payload)")],
                                 revision_action="withdraw")
        challenger = ProtocolAgent("challenger", challenge_verdict="insufficient")
        coordinator = MultiAgentCoordinator(
            [SecurityRuleReviewer(), primary, security, challenger],
            review_mode="adaptive_multi_agent", max_workers=1,
            snapshot_provider=InMemorySnapshotProvider({"a.py": "eval(payload)\n"}),
        )
        coordinator.review_with_execution_context(
            "proposer", diff, parsed, {"force_challenge": True}, source_sha="sha",
        )
        self.assertEqual(1, security.revision_calls)
        self.assertEqual(0, primary.revision_calls)

    def test_same_semantic_finding_keeps_distinct_agent_proposers(self):
        diff = (
            "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n"
            "-return user.tenant_id == invoice.tenant_id\n"
            "+return user.is_authenticated\n"
        )
        parsed = parse_unified_diff(diff)
        primary = ProbeAgent("primary", [finding(
            "CWE-863", Severity.HIGH, "a.py", 1,
            "return user.is_authenticated", "tenant authorization bypass",
        )])
        security = ProbeAgent("security", [finding(
            "CWE-863", Severity.HIGH, "a.py", 1,
            "return user.is_authenticated", "tenant authorization bypass",
        )])
        challenger = ProtocolAgent("challenger", challenge_verdict="support")
        coordinator = MultiAgentCoordinator(
            [primary, security, challenger], review_mode="adaptive_multi_agent", max_workers=1,
            snapshot_provider=InMemorySnapshotProvider({
                "a.py": "return user.is_authenticated\n",
            }),
        )
        coordinator.review_with_context("distinct-proposers", diff, parsed, source_sha="sha")
        claims = [
            item for item in coordinator.collaboration_summary("distinct-proposers")["claims"]
            if item["source_kind"] == "agent"
        ]
        self.assertEqual(2, len(claims))
        self.assertEqual(2, len({item["claim_id"] for item in claims}))
        self.assertEqual(2, len({item["proposer_run_id"] for item in claims}))
        summary = coordinator.collaboration_summary("distinct-proposers")
        related = [item for item in summary["challenges"] if item["claim_id"] in {
            claim["claim_id"] for claim in claims
        }]
        self.assertEqual(2, len(related))
        self.assertEqual({"support"}, {item["verdict"] for item in related})
        self.assertFalse(summary["escalations"])

    def test_cross_shard_respects_domain_activation_and_agent_budget(self):
        diff = (
            "--- a/auth.py\n+++ b/auth.py\n@@ -1 +1 @@\n-old\n+eval(payload)\n"
            "--- a/helper.py\n+++ b/helper.py\n@@ -1 +1 @@\n-old\n+return payload\n"
        )
        parsed = parse_unified_diff(diff)
        primary, security, reliability = (
            ProbeAgent("primary"), ProbeAgent("security"), ProbeAgent("reliability")
        )
        coordinator = MultiAgentCoordinator(
            [SecurityRuleReviewer(), primary, security, reliability],
            review_mode="adaptive_multi_agent", max_workers=1,
            shard_file_threshold=2, shard_changed_line_threshold=9999,
            agent_budget={"max_agent_runs": 8, "max_llm_calls": 20,
                          "max_tool_calls": 20, "max_input_tokens": 60000,
                          "max_output_tokens": 12000},
        )
        coordinator.review_with_context("cross-domain", diff, parsed, source_sha="sha")
        summary = coordinator.collaboration_summary("cross-domain")
        self.assertTrue(primary.contexts)
        self.assertTrue(security.contexts)
        self.assertFalse(reliability.contexts)
        self.assertEqual(summary["agent_count"], summary["budget"]["used"]["agent_runs"])

    def test_shell_literal_is_not_executable_structural_evidence(self):
        parsed = parse_unified_diff(
            '--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+sample = "shell=True"\n'
        )
        f = finding("SEC-SUBPROCESS-SHELL", Severity.HIGH,
                    evidence='sample = "shell=True"')
        claim = Claim.from_finding(f, "scanner", "deterministic-checker")
        decision = AdaptiveDecisionPolicy().decide(claim, f, parsed, [], [], "sha")
        self.assertEqual("reject", decision.outcome)

    def test_tenant_guard_structurally_refutes_authorization_bypass(self):
        parsed = parse_unified_diff(
            "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+return user.is_authenticated and user.tenant_id == invoice.tenant_id\n"
        )
        f = finding("SEC-AUTHZ-BYPASS", Severity.HIGH,
                    evidence="return user.is_authenticated and user.tenant_id == invoice.tenant_id")
        claim = Claim.from_finding(f, "agent", "agent")
        evidence = Evidence("E:1", "sha", "a.py", 1, "read_file", "read_file",
                            "agent", "d", "tool", (claim.claim_id,))
        support = Challenge(claim.claim_id, "support", "claimed bypass", "challenger", ("E:1",))
        decision = AdaptiveDecisionPolicy().decide(claim, f, parsed, [evidence], [support], "sha")
        self.assertEqual("reject", decision.outcome)
        self.assertEqual(("not-executable-structure",), decision.reason_codes)

    def test_python_signature_scanner_emits_only_provable_arity_mismatch(self):
        diff = (
            "--- a/src/caller.py\n+++ b/src/caller.py\n@@ -1 +1 @@\n"
            "-    return charge(user, amount)\n+    return charge(user, amount, currency)\n"
        )
        base = {
            "task_id": "arity", "source_sha": "sha",
            "parsed": parse_unified_diff(diff),
        }
        broken = MultiAgentCoordinator._cross_file_signature_candidates({
            **base, "snapshot_provider": InMemorySnapshotProvider({
                "src/caller.py": "def checkout():\n    return charge(user, amount, currency)\n",
                "src/api.py": "def charge(user, amount):\n    return True\n",
            }),
        })
        self.assertEqual(1, len(broken))
        finding_value, _run_id, evidence = broken[0]
        self.assertEqual("COR-API-ARITY", finding_value.rule_id)
        self.assertEqual({"src/caller.py", "src/api.py"}, {item.path for item in evidence})
        compatible = MultiAgentCoordinator._cross_file_signature_candidates({
            **base, "snapshot_provider": InMemorySnapshotProvider({
                "src/caller.py": "def checkout():\n    return charge(user, amount, currency)\n",
                "src/api.py": "def charge(user, amount, currency='USD'):\n    return True\n",
            }),
        })
        self.assertEqual([], compatible)

    def test_coordinator_passes_pinned_snapshot_to_signature_claim_scanner(self):
        diff = (
            "--- a/src/caller.py\n+++ b/src/caller.py\n@@ -1 +1 @@\n"
            "-    return charge(user, amount)\n+    return charge(user, amount, currency)\n"
        )
        primary = ProtocolAgent("primary")
        challenger = ProtocolAgent("challenger", challenge_verdict="support")
        coordinator = MultiAgentCoordinator(
            [primary, challenger], review_mode="adaptive_multi_agent",
            snapshot_provider=InMemorySnapshotProvider({
                "src/caller.py": "def checkout():\n    return charge(user, amount, currency)\n",
                "src/api.py": "def charge(user, amount):\n    return True\n",
            }),
        )
        coordinator.review_with_context(
            "arity-integration", diff, parse_unified_diff(diff), source_sha="sha",
        )
        summary = coordinator.collaboration_summary("arity-integration")
        self.assertTrue(any(
            item["rule_id"] == "COR-API-ARITY" for item in summary["claims"]
        ))

    def test_quota_fallback_records_actual_models_without_secrets(self):
        reviewer = OpenAICompatibleReviewer(
            "https://qwen.example/v1", "qwen-secret", "qwen3.7-max",
            provider="custom", fallback={
                "provider": "deepseek", "base_url": "https://api.deepseek.com",
                "api_key": "deepseek-secret", "model": "deepseek-v4-flash",
            },
        )
        quota = urllib.error.HTTPError(
            "https://qwen.example/v1/chat/completions", 402, "quota", {},
            io.BytesIO(b'{"error":{"code":"insufficient_quota"}}'),
        )

        class Response:
            def __enter__(self):
                return self
            def __exit__(self, *_args):
                return False
            def read(self):
                return json.dumps({
                    "choices": [{"message": {"content": '{"findings":[]}'},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                }).encode("utf-8")

        with mock.patch("urllib.request.urlopen", side_effect=[quota, Response()]):
            result = reviewer._request_json({"model": "qwen3.7-max", "messages": []})
        attempts = result["__request_attempts__"]
        self.assertEqual(["custom", "deepseek"], [item["provider"] for item in attempts])
        self.assertEqual(["qwen3.7-max", "deepseek-v4-flash"], [item["model"] for item in attempts])
        self.assertEqual(["quota_exhausted", "success"], [item["status"] for item in attempts])
        rendered = json.dumps(attempts)
        self.assertNotIn("qwen-secret", rendered)
        self.assertNotIn("deepseek-secret", rendered)

    def test_non_quota_http_error_does_not_fallback(self):
        reviewer = OpenAICompatibleReviewer(
            "https://qwen.example/v1", "qwen-secret", "qwen3.7-max",
            provider="custom", fallback={
                "provider": "deepseek", "base_url": "https://api.deepseek.com",
                "api_key": "deepseek-secret", "model": "deepseek-v4-flash",
            },
        )
        unauthorized = urllib.error.HTTPError(
            "https://qwen.example/v1/chat/completions", 401, "unauthorized", {},
            io.BytesIO(b'{"error":{"code":"invalid_api_key"}}'),
        )
        with mock.patch("urllib.request.urlopen", side_effect=unauthorized) as request:
            with self.assertRaises(RuntimeError):
                reviewer._request_json({"model": "qwen3.7-max", "messages": []})
        self.assertEqual(1, request.call_count)

    def test_provider_flat_tool_action_is_normalized_to_runtime_protocol(self):
        value = OpenAICompatibleReviewer._normalize_tool_action({
            "action": "read_file", "path": "a.py", "cursor": 0, "limit": 5,
            "reason": "inspect definition",
        }, {"read_file", "grep_repo"})
        self.assertEqual("tool", value["action"])
        self.assertEqual("read_file", value["tool"])
        self.assertEqual({"path": "a.py", "start_line": 1, "end_line": 5},
                         value["arguments"])

    def test_provider_flat_single_finding_and_text_confidence_are_normalized(self):
        reviewer = OpenAICompatibleReviewer(
            "https://model.example/v1", "secret", "model", provider="test",
        )
        response = {
            "rule_id": "SEC-EVAL", "severity": "critical", "title": "eval",
            "explanation": "untrusted input is evaluated", "path": "a.py", "line": 1,
            "evidence": "eval(payload)", "fix": "remove eval", "test": "exercise input",
            "confidence": "high", "__usage__": {}, "__request_attempts__": [],
        }
        parsed = parse_unified_diff(
            "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+eval(payload)\n"
        )
        with mock.patch.object(reviewer, "_request_json", return_value=response):
            result = reviewer.agent_step({
                "parsed": parsed, "assignment": {}, "available_tools": [],
                "managed_context": "context", "observations": [{
                    "ok": True, "tool": "read_file",
                }],
            })
        self.assertEqual("final", result["action"])
        self.assertEqual(1, len(result["findings"]))
        self.assertEqual(.7, result["findings"][0].confidence)

    def test_repeated_rate_limit_falls_back_after_one_bounded_retry(self):
        reviewer = OpenAICompatibleReviewer(
            "https://qwen.example/v1", "qwen-secret", "qwen3.7-max",
            provider="custom", fallback={
                "provider": "deepseek", "base_url": "https://api.deepseek.com",
                "api_key": "deepseek-secret", "model": "deepseek-v4-flash",
            },
        )
        def rate_limit():
            return urllib.error.HTTPError(
                "https://qwen.example/v1/chat/completions", 429, "rate", {},
                io.BytesIO(b'{"error":{"code":"rate_limit"}}'),
            )
        class Response:
            def __enter__(self): return self
            def __exit__(self, *_args): return False
            def read(self):
                return b'{"choices":[{"message":{"content":"{\\"findings\\":[]}"}}]}'
        with mock.patch("urllib.request.urlopen", side_effect=[rate_limit(), rate_limit(), Response()]):
            result = reviewer._request_json({"model": "qwen3.7-max", "messages": []})
        attempts = result["__request_attempts__"]
        self.assertEqual(["failed", "quota_exhausted", "success"],
                         [item["status"] for item in attempts])
        self.assertEqual(["qwen3.7-max", "qwen3.7-max", "deepseek-v4-flash"],
                         [item["model"] for item in attempts])
        self.assertEqual(1, len({item["logical_call_id"] for item in attempts}))

    def test_contracts_are_immutable_and_evidence_ids_are_content_addressed(self):
        spec = AgentSpec("a", "primary")
        with self.assertRaises(FrozenInstanceError):
            spec.role = "challenger"
        one = evidence_from_record({"tool": "read_file", "path": "a.py", "line": 1,
                                    "result": "x"}, "sha", "run-a")
        two = evidence_from_record({"tool": "read_file", "path": "a.py", "line": 1,
                                    "result": "x"}, "sha", "run-a")
        other = evidence_from_record({"tool": "read_file", "path": "a.py", "line": 1,
                                      "result": "x"}, "sha", "run-b")
        self.assertEqual(one.evidence_id, two.evidence_id)
        self.assertNotEqual(one.evidence_id, other.evidence_id)
        self.assertTrue(one.evidence_id.startswith("E:"))

    def test_router_activates_security_reliability_and_both(self):
        router = DeterministicRiskRouter()
        security = parse_unified_diff("--- a/auth.py\n+++ b/auth.py\n@@ -1 +1 @@\n-old\n+eval(payload)\n")
        reliability = parse_unified_diff("--- a/job.py\n+++ b/job.py\n@@ -1 +1 @@\n-old\n+except Exception:\n")
        both = parse_unified_diff("--- a/transaction_auth.py\n+++ b/transaction_auth.py\n@@ -1 +1 @@\n-old\n+eval(payload) # retry\n")
        self.assertIn("security", router.activated(router.score("eval", security, [finding("SEC-EVAL", Severity.CRITICAL)])))
        self.assertIn("reliability", router.activated(router.score("retry except", reliability, [finding("REL-EMPTY-EXCEPT", Severity.MEDIUM)])))
        self.assertEqual({"security", "reliability"}, set(router.activated(router.score("auth retry", both, [
            finding("SEC-EVAL", Severity.CRITICAL), finding("REL-EMPTY-EXCEPT", Severity.MEDIUM),
        ]))))

    def test_decision_rejects_literal_stale_and_counterevidence(self):
        parsed = parse_unified_diff('--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+sample = "eval(user)"\n')
        f = finding("SEC-EVAL", Severity.CRITICAL, evidence='sample = "eval(user)"')
        claim = Claim.from_finding(f, "run", "deterministic-checker")
        self.assertEqual("reject", AdaptiveDecisionPolicy().decide(claim, f, parsed, [], [], "sha").outcome)

        parsed = parse_unified_diff("--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+return allowed\n")
        f = finding()
        claim = Claim.from_finding(f, "run", "agent")
        stale = Evidence("E:1", "old-sha", "a.py", 1, "read_file", "read_file", "run", "d", "tool")
        support = Challenge(claim.claim_id, "support", "checked", "challenger", ("E:1",))
        decision = AdaptiveDecisionPolicy().decide(claim, f, parsed, [stale], [support], "new-sha")
        self.assertEqual(("stale-snapshot-evidence",), decision.reason_codes)
        current = Evidence("E:2", "new-sha", "a.py", 1, "read_file", "read_file", "run", "d", "tool")
        refute = Challenge(claim.claim_id, "refute", "safe branch", "challenger", ("E:2",))
        self.assertEqual("reject", AdaptiveDecisionPolicy().decide(claim, f, parsed, [current], [refute], "new-sha").outcome)

    def test_high_risk_semantic_claim_fails_closed_without_challenge(self):
        parsed = parse_unified_diff("--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+return allowed\n")
        f = finding()
        claim = Claim.from_finding(f, "run", "agent")
        semantic = Evidence("E:1", "sha", "a.py", 1, "read_file", "read_file", "run", "d", "tool")
        decision = AdaptiveDecisionPolicy().decide(claim, f, parsed, [semantic], [], "sha")
        self.assertEqual("escalate", decision.outcome)
        self.assertIn("challenge-support-required", decision.reason_codes)

    def test_agent_claim_requires_supported_family_and_matching_causal_story(self):
        parsed = parse_unified_diff(
            "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+return normalize(value)\n"
        )
        unsupported = finding(
            "CWE-758", Severity.HIGH, evidence="return normalize(value)",
            explanation="normalize is not imported",
        )
        claim = Claim.from_finding(unsupported, "run", "agent")
        decision = AdaptiveDecisionPolicy().decide(claim, unsupported, parsed, [], [], "sha")
        self.assertEqual("escalate", decision.outcome)
        self.assertEqual(("unsupported-semantic-family",), decision.reason_codes)

        mismatched = finding(
            "CWE-628", Severity.HIGH, evidence="return normalize(value)",
            explanation="the function returns the original unnormalized value",
        )
        claim = Claim.from_finding(mismatched, "run", "agent")
        decision = AdaptiveDecisionPolicy().decide(claim, mismatched, parsed, [], [], "sha")
        self.assertEqual("reject", decision.outcome)
        self.assertEqual(("causal-family-mismatch",), decision.reason_codes)

    def test_unrelated_cross_file_read_cannot_prove_semantic_claim(self):
        parsed = parse_unified_diff(
            "--- a/auth.py\n+++ b/auth.py\n@@ -1 +1 @@\n-old\n+return user.is_authenticated\n"
        )
        value = finding(
            "CWE-863", Severity.HIGH, "auth.py", 1,
            "return user.is_authenticated", "authorization ownership check is bypassed",
        )
        claim = Claim.from_finding(value, "primary", "agent")
        unrelated = Evidence(
            "E:other", "sha", "unrelated.py", 1, "read_file", "read_file",
            "primary", "digest", "tool", (claim.claim_id,), result_nonempty=True,
        )
        support = Challenge(claim.claim_id, "support", "checked", "challenger", ("E:other",))
        decision = AdaptiveDecisionPolicy().decide(
            claim, value, parsed, [unrelated], [support], "sha",
        )
        self.assertEqual("escalate", decision.outcome)
        self.assertIn("semantic-evidence-required", decision.reason_codes)

    def test_conditional_challenger_ignores_self_declared_high_without_router_risk(self):
        diff = "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+return balance - amount\n"
        parsed = parse_unified_diff(diff)
        primary = ProtocolAgent("primary", [finding(
            "CWE-840", Severity.HIGH, "a.py", 1,
            "return balance - amount", "business balance invariant may be violated",
        )])
        challenger = ProtocolAgent("challenger", challenge_verdict="support")
        coordinator = MultiAgentCoordinator(
            [primary, challenger], review_mode="adaptive_multi_agent", max_workers=1,
            snapshot_provider=InMemorySnapshotProvider({"a.py": "return balance - amount\n"}),
        )
        coordinator.review_with_context("conditional-low-risk", diff, parsed, source_sha="sha")
        summary = coordinator.collaboration_summary("conditional-low-risk")
        self.assertFalse(summary["challenge_activation"]["triggered"])
        self.assertFalse(challenger.contexts)

    def test_conditional_challenger_runs_for_admitted_router_confirmed_risk(self):
        class AuthRiskScanner(Reviewer):
            name = "auth-risk-scanner"
            execution_kind = "deterministic-checker"
            def review(self, diff, parsed):
                line = parsed.added_lines[0]
                return [finding(
                    "SEC-AUTHZ-BYPASS", Severity.HIGH, line.path, line.line,
                    line.content, "authorization ownership check is bypassed",
                )]

        diff = "--- a/auth.py\n+++ b/auth.py\n@@ -1 +1 @@\n-old\n+return user.is_authenticated\n"
        parsed = parse_unified_diff(diff)
        primary = ProtocolAgent("primary", [finding(
            "CWE-863", Severity.HIGH, "auth.py", 1,
            "return user.is_authenticated", "authorization ownership check is bypassed",
        )])
        challenger = ProtocolAgent("challenger", challenge_verdict="support")
        coordinator = MultiAgentCoordinator(
            [AuthRiskScanner(), primary, challenger], review_mode="adaptive_multi_agent",
            max_workers=1,
            snapshot_provider=InMemorySnapshotProvider({"auth.py": "return user.is_authenticated\n"}),
        )
        coordinator.review_with_context("conditional-high-risk", diff, parsed, source_sha="sha")
        summary = coordinator.collaboration_summary("conditional-high-risk")
        self.assertTrue(summary["challenge_activation"]["triggered"])
        self.assertIn("admitted-high-risk", summary["challenge_activation"]["reasons"])
        self.assertTrue(any(
            (run.get("spec") or {}).get("role") == "challenger"
            for run in summary["agent_runs"]
        ))

    def test_shared_budget_is_thread_safe_and_fail_closed(self):
        budget = SharedAgentBudget(max_agent_runs=10)
        outcomes = []
        threads = [threading.Thread(target=lambda: outcomes.append(budget.reserve("agent_runs"))) for _ in range(30)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(10, sum(outcomes))
        self.assertEqual(10, budget.snapshot()["used"]["agent_runs"])
        self.assertIn("agent_runs", budget.snapshot()["violations"])

    def test_rules_only_reports_zero_true_agents(self):
        diff = "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+eval(payload)\n"
        parsed = parse_unified_diff(diff)
        coordinator = MultiAgentCoordinator([SecurityRuleReviewer()], review_mode="rules_only")
        results = coordinator.review_with_context("rules-task", diff, parsed, source_sha="sha")
        summary = coordinator.collaboration_summary("rules-task")
        self.assertEqual(["SEC-EVAL"], [item.rule_id for item in results])
        self.assertEqual(0, summary["agent_count"])
        self.assertEqual([], summary["agent_runs"])
        self.assertEqual("deterministic-checker", summary["participant_kinds"]["security-agent"])

    def test_reducer_deduplicates_rule_and_cwe_aliases(self):
        class AliasScanner(Reviewer):
            name = "alias-scanner"
            def review(self, diff, parsed):
                line = parsed.added_lines[0]
                return [finding("CWE-95", Severity.CRITICAL, line.path, line.line,
                                line.content, "dynamic execution")]
        diff = "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+eval(payload)\n"
        parsed = parse_unified_diff(diff)
        coordinator = MultiAgentCoordinator(
            [SecurityRuleReviewer(), AliasScanner()], review_mode="rules_only",
        )
        results = coordinator.review_with_context("aliases", diff, parsed, source_sha="sha")
        self.assertEqual(1, len(results))

    def test_adaptive_display_reducer_keeps_alias_decisions_but_emits_one_finding(self):
        diff = "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+eval(payload)\n"
        parsed = parse_unified_diff(diff)
        primary = ProtocolAgent("primary", [finding(
            "CWE-95", Severity.CRITICAL, "a.py", 1, "eval(payload)",
            "untrusted payload reaches dynamic execution",
        )])
        challenger = ProtocolAgent("challenger", challenge_verdict="support")
        coordinator = MultiAgentCoordinator(
            [SecurityRuleReviewer(), primary, challenger],
            review_mode="adaptive_multi_agent", max_workers=1,
            snapshot_provider=InMemorySnapshotProvider({"a.py": "eval(payload)\n"}),
        )
        results = coordinator.review_with_execution_context(
            "adaptive-aliases", diff, parsed, {"force_challenge": True}, source_sha="sha",
        )
        summary = coordinator.collaboration_summary("adaptive-aliases")
        self.assertEqual(1, len(results))
        self.assertGreaterEqual(len(summary["decisions"]), 2)
        self.assertGreaterEqual(
            sum(item["outcome"] == "accept" for item in summary["decisions"]
                if item["claim_id"] in {claim["claim_id"] for claim in summary["claims"]}),
            2,
        )

    def test_adaptive_router_starts_only_matching_domain_agent(self):
        diff = "--- a/auth.py\n+++ b/auth.py\n@@ -1 +1 @@\n-old\n+eval(payload)\n"
        parsed = parse_unified_diff(diff)
        primary, security, reliability = ProbeAgent("primary"), ProbeAgent("security"), ProbeAgent("reliability")
        coordinator = MultiAgentCoordinator(
            [SecurityRuleReviewer(), primary, security, reliability],
            review_mode="adaptive_multi_agent", max_workers=1,
        )
        coordinator.review_with_context("routing", diff, parsed, source_sha="sha")
        self.assertTrue(primary.contexts)
        self.assertTrue(security.contexts)
        self.assertFalse(reliability.contexts)

    def test_blind_challenger_context_omits_identity_confidence_and_memory(self):
        diff = "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+return allowed\n"
        parsed = parse_unified_diff(diff)
        primary_finding = finding()
        primary, challenger = ProbeAgent("primary", [primary_finding]), ProbeAgent("challenger", [primary_finding])
        coordinator = MultiAgentCoordinator(
            [primary, challenger], review_mode="adaptive_multi_agent", max_workers=1,
            snapshot_provider=InMemorySnapshotProvider({"a.py": "return allowed\n"}),
        )
        coordinator.review_with_execution_context(
            "blind", diff, parsed, {"force_challenge": True}, source_sha="sha",
        )
        combined = "\n".join(challenger.contexts)
        self.assertNotIn("proposer_run_id", combined)
        self.assertNotIn('"confidence":', combined.lower())
        self.assertNotIn("working memory", combined.lower())
        self.assertNotIn("recall_memory", combined.lower())


if __name__ == "__main__":
    unittest.main()
