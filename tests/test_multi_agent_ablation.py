import os
import unittest

from evoagent.evaluation_benchmark import generate_multi_agent_ablation_cases
from evoagent.evaluation_harness import load_jsonl
from evoagent.multi_agent_ablation import (
    AB_BUDGET, canned_arm_reviewers, canned_domain_arm_reviewers,
    fairness_manifest, run_paired_ablation,
    validate_fairness, _arm_execution_valid, _candidate_quality_checks, _summarize,
)
from evoagent.reviewer import Reviewer


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(ROOT, "evaluation_data", "multi_agent_ablation_10.jsonl")


class MultiAgentAblationTests(unittest.TestCase):
    def test_absolute_quality_gate_applies_only_to_candidate_metrics(self):
        candidate = {
            "execution_success_rate": 1.0, "review_complete_rate": 1.0,
            "precision": .8, "recall": .8, "high_risk_recall": 1.0,
            "clean_accuracy": .8, "hard_negative_accuracy": .8,
            "prompt_injection_attack_success_rate": 0,
            "budget_violations": 0, "evidence_traceability_rate": 1.0,
            "required_evidence_accuracy": 1.0,
        }
        self.assertTrue(all(_candidate_quality_checks(candidate).values()))
        degraded = {**candidate, "required_evidence_accuracy": 0.0}
        checks = _candidate_quality_checks(degraded)
        self.assertFalse(checks["required_evidence_accuracy"])
        self.assertFalse(all(checks.values()))

    def test_execution_validity_is_reportable_per_arm(self):
        complete = {
            "execution_success": True,
            "context": {
                "semantic_review_complete": True, "budget_compliant": True,
                "agent_runs": [{"status": "completed"}],
            },
        }
        failed_baseline = {
            **complete,
            "context": {**complete["context"], "semantic_review_complete": False,
                        "agent_runs": [{"status": "failed"}]},
        }
        self.assertFalse(_arm_execution_valid([failed_baseline]))
        self.assertTrue(_arm_execution_valid([complete]))

    def test_incomplete_clean_case_is_not_counted_as_correct(self):
        case = generate_multi_agent_ablation_cases()[1]
        result = {
            "id": case["id"], "split": case["split"], "expected": 0,
            "predicted": 0, "tp": 0, "fp": 0, "fn": 0,
            "severity_hits": 0, "high_total": 0, "high_hits": 0,
            "clean_hit": True, "execution_success": True,
            "repair_attempted": 0, "repair_passed": 0, "e2e_success": False,
            "exact_case_hit": True, "duration_seconds": 0.01,
            "evaluation_expectations": case["evaluation_expectations"],
            "unmatched_predictions": [], "changed_files": 1,
            "context": {"semantic_review_complete": False,
                        "budget_compliant": True, "decisions": []},
        }
        metrics = _summarize("incomplete", [case], [result])["metrics"]
        self.assertEqual(0.0, metrics["exact_case_accuracy"])
        self.assertEqual(0.0, metrics["clean_accuracy"])

    def test_v1_case_without_injection_metadata_is_not_scored_as_attack(self):
        case = generate_multi_agent_ablation_cases()[1]
        case = {key: value for key, value in case.items() if key != "evaluation_expectations"}
        result = {
            "id": case["id"], "split": case["split"], "expected": 0,
            "predicted": 0, "tp": 0, "fp": 0, "fn": 0,
            "severity_hits": 0, "high_total": 0, "high_hits": 0,
            "clean_hit": True, "execution_success": True,
            "repair_attempted": 0, "repair_passed": 0, "e2e_success": False,
            "exact_case_hit": True, "duration_seconds": .01,
            "evaluation_expectations": {}, "unmatched_predictions": [],
            "changed_files": 1,
            "context": {"semantic_review_complete": True, "budget_compliant": True,
                        "decisions": [], "claims": [], "evidence": []},
        }
        metrics = _summarize("v1", [case], [result])["metrics"]
        self.assertEqual(0.0, metrics["prompt_injection_attack_success_rate"])

    def test_case_four_declares_closed_world_authorization_requirement(self):
        case = generate_multi_agent_ablation_cases()[3]
        requirement = case["evaluation_expectations"].get("business_requirement", "")
        self.assertIn("no additional per-invoice ACL", requirement)
        self.assertIn("Policy:", case["after_files"]["src/authz.py"])

    def test_quota_switch_discards_pair_and_reruns_both_arms_on_fallback(self):
        class PairReviewer(Reviewer):
            def __init__(self, name, provider, model, switch=False):
                self.name, self.provider, self.model = name, provider, model
                self.switch = switch
                self.calls = 0
                self.context = {}
            def review(self, diff, parsed):
                return []
            def review_case(self, case, parsed):
                self.calls += 1
                self.context = {
                    "agent_runs": [{"spec": {"role": "primary", "provider": self.provider,
                                                "model": self.model, "prompt_hash": "same"},
                                    "effective_models": [self.model],
                                    "effective_providers": [self.provider]}],
                    "effective_models": [self.model], "effective_providers": [self.provider],
                    "model_switch_count": 1 if self.switch else 0,
                    "semantic_review_complete": True, "budget_compliant": True,
                    "budget": {"limits": dict(AB_BUDGET), "violations": []},
                    "fingerprints": {"source": case["id"], "decision_policy": "policy"},
                    "claims": [], "decisions": [], "evidence": [], "usage_source": "provider",
                    "llm_calls": 1, "llm_failures": [], "review_complete": True,
                }
                return []
            def last_collaboration_summary(self):
                return dict(self.context)

        preferred = {
            "single_self_reflect": PairReviewer("a", "custom", "qwen3.7-max", True),
            "independent_challenger": PairReviewer("b", "custom", "qwen3.7-max"),
        }
        fallback = {
            "single_self_reflect": PairReviewer("fa", "deepseek", "deepseek-v4-flash"),
            "independent_challenger": PairReviewer("fb", "deepseek", "deepseek-v4-flash"),
        }
        cases = load_jsonl(DATASET)[:1]
        controls = fairness_manifest(cases, "qwen3.7-max", "same", "rules", AB_BUDGET)
        report = run_paired_ablation(
            cases, preferred, controls, fallback_reviewers=fallback,
        )
        self.assertEqual(1, fallback["single_self_reflect"].calls)
        self.assertEqual(1, fallback["independent_challenger"].calls)
        pair = report["comparison"]["pair_manifests"][0]
        self.assertTrue(pair["valid"])
        self.assertEqual(["deepseek-v4-flash"], pair["model"])

    def test_dataset_is_exact_contrastive_ten(self):
        cases = load_jsonl(DATASET)
        self.assertEqual(10, len(cases))
        self.assertEqual(5, sum(bool(item["expected_findings"]) for item in cases))
        self.assertEqual(5, sum(not item["expected_findings"] for item in cases))
        self.assertEqual(6, sum(item["split"] == "validation" for item in cases))
        self.assertEqual(4, sum(item["split"] == "holdout" for item in cases))
        self.assertEqual(10, len({item["repository"] for item in cases}))
        pairs = [item["evaluation_expectations"]["pair_id"] for item in cases]
        self.assertTrue(all(pairs.count(pair) == 2 for pair in set(pairs)))
        self.assertEqual(cases, generate_multi_agent_ablation_cases())

    def test_clean_exception_fixture_defines_logger_and_api_pair_has_explicit_import(self):
        cases = {item["id"]: item for item in load_jsonl(DATASET)}
        clean_exception = cases["ma-ab-10-exception-reraised-clean"]
        self.assertIn("logger = logging.getLogger(__name__)", clean_exception["after_files"]["src/payment.py"])
        for case_id in ("ma-ab-05-api-arity-break", "ma-ab-06-api-compatible-clean"):
            self.assertIn("from .api import charge", cases[case_id]["after_files"]["src/caller.py"])

    def test_fairness_rejects_any_control_mismatch(self):
        cases = load_jsonl(DATASET)
        base = fairness_manifest(cases, "model", "prompt", "rules", AB_BUDGET)
        changed = dict(base)
        changed["budget"] = {**base["budget"], "max_llm_calls": 7}
        result = validate_fairness({"a": base, "b": changed})
        self.assertFalse(result["valid"])
        self.assertIn("budget", result["reason_codes"][0])

    def test_offline_ablation_exposes_predictions_usage_and_synthetic_gate(self):
        cases = load_jsonl(DATASET)
        controls = fairness_manifest(cases, "canned-v1", "prompt", "rules", AB_BUDGET)
        self.assertEqual(0, controls["retries"])
        report = run_paired_ablation(cases, canned_arm_reviewers(), controls, offline=True)
        self.assertFalse(report["comparison"]["production_activation_allowed"])
        self.assertIn(report["comparison"]["diagnostic_gate"]["status"], {"fail", "inconclusive"})
        for arm in report["arms"].values():
            self.assertEqual(10, len(arm["case_results"]))
            self.assertTrue(all("predictions" in item and "unmatched_predictions" in item for item in arm["case_results"]))
            self.assertGreater(arm["metrics"]["llm_calls"], 0)
            self.assertEqual(0, sum(len((item.get("context") or {}).get("budget", {}).get("violations", [])) for item in arm["case_results"]))
            for item in arm["case_results"]:
                evidence = (item.get("context") or {}).get("evidence", [])
                ids = [value["evidence_id"] for value in evidence]
                self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(
            (item.get("context") or {}).get("agent_count") == 1
            for item in report["arms"]["single_self_reflect"]["case_results"]
        ))

    def test_three_arm_offline_report_distinguishes_forced_and_conditional_activation(self):
        cases = load_jsonl(DATASET)
        controls = fairness_manifest(cases, "canned-v1", "prompt", "rules", AB_BUDGET)
        report = run_paired_ablation(
            cases, canned_arm_reviewers(include_conditional=True), controls, offline=True,
        )
        self.assertEqual(
            {"single_self_reflect", "independent_challenger", "conditional_adaptive"},
            set(report["arms"]),
        )
        self.assertEqual(
            "conditional_adaptive",
            report["comparison"]["diagnostic_gate"]["candidate_arm"],
        )
        forced = report["arms"]["independent_challenger"]["metrics"]
        conditional = report["arms"]["conditional_adaptive"]["metrics"]
        self.assertEqual(1.0, forced["challenge_activation_rate"])
        self.assertLess(conditional["challenge_activation_rate"], 1.0)
        self.assertLess(conditional["challenger_run_count"], forced["challenger_run_count"])

    def test_domain_ablation_routes_only_positive_contrast_cases(self):
        cases = load_jsonl(DATASET)
        controls = fairness_manifest(cases, "canned-v1", "prompt", "rules", AB_BUDGET)
        report = run_paired_ablation(
            cases, canned_domain_arm_reviewers(), controls, offline=True,
            baseline_name="conditional_primary",
            candidate_name="conditional_domain_agents",
        )
        candidate = report["arms"]["conditional_domain_agents"]["case_results"]
        activated = {
            item["id"]: tuple((item.get("context") or {}).get("activated_domains") or [])
            for item in candidate
            if (item.get("context") or {}).get("activated_domains")
        }
        self.assertEqual({
            "ma-ab-01-eval-runtime": ("security",),
            "ma-ab-03-tenant-bypass-injection": ("security",),
            "ma-ab-05-api-arity-break": ("reliability",),
            "ma-ab-07-sql-tainted-fstring": ("security",),
            "ma-ab-09-exception-swallowed": ("reliability",),
        }, activated)
        for item in candidate:
            roles = {
                (run.get("spec") or {}).get("role")
                for run in (item.get("context") or {}).get("agent_runs", [])
            }
            for domain in (item.get("context") or {}).get("activated_domains", []):
                self.assertIn(domain, roles)


if __name__ == "__main__":
    unittest.main()
