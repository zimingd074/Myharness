import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class BadCaseDatasetTests(unittest.TestCase):
    def test_bad_case_dataset_preserves_truth_and_diagnostic_partition(self):
        source = {item["id"]: item for item in load(ROOT / "evaluation_data" / "pr_diff_100.jsonl")}
        cases = load(ROOT / "evaluation_data" / "multi_agent_bad_cases_53.jsonl")
        self.assertEqual(53, len(cases))
        self.assertEqual(53, len({item["id"] for item in cases}))
        counts = {}
        raw_errors = 0
        policy_only = 0
        for case in cases:
            diagnostic = case["bad_case_diagnostics"]
            self.assertEqual(source[case["id"]]["expected_findings"], case["expected_findings"])
            self.assertEqual("synthetic-controlled", case["source"]["kind"])
            self.assertFalse(diagnostic["policy_hit"])
            self.assertTrue(diagnostic["failure_classes"])
            attribution = diagnostic["primary_attribution"]
            counts[attribution] = counts.get(attribution, 0) + 1
            raw_errors += int(not diagnostic["raw_exact_case_hit"])
            policy_only += int(diagnostic["raw_exact_case_hit"])
        self.assertEqual({
            "dataset-fixture-or-label": 36,
            "implementation": 11,
            "mixed": 6,
        }, counts)
        self.assertEqual(17, raw_errors)
        self.assertEqual(36, policy_only)

    def test_analysis_summary_matches_derived_dataset(self):
        summary = json.loads(
            (ROOT / "reports" / "multi_agent_bad_cases_53_analysis.json").read_text(encoding="utf-8")
        )
        self.assertEqual(53, summary["case_count"])
        self.assertEqual(17, summary["raw_finding_error_cases"])
        self.assertEqual(36, summary["policy_only_failure_cases"])
        self.assertEqual(["pr-0024", "pr-0034"], summary["candidate_regressions"])
        self.assertEqual(["pr-0082"], summary["candidate_improvements"])
        self.assertEqual(3, summary["failure_class_counts"]["semantic-review-incomplete"])
        self.assertEqual(3, summary["failure_class_counts"]["budget-violation"])


if __name__ == "__main__":
    unittest.main()
