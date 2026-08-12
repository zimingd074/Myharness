from datetime import datetime
import os
import tempfile
import unittest

from evoagent.ab_report import build_ab_summary, markdown_sections, report_paths, timestamped_run_directory


class ABReportTests(unittest.TestCase):
    def test_end_to_end_summary_lists_introduced_rules(self):
        summary = build_ab_summary(
            "e2e",
            baseline_rules=[{"rule_id": "SEC-EVAL", "severity": "critical"}],
            introduced_rules=[{"rule_id": "SEC-YAML-LOAD", "severity": "high"}],
        )
        rendered = "\n".join(markdown_sections(summary))
        self.assertIn("single-agent-baseline", rendered)
        self.assertIn("SEC-YAML-LOAD", rendered)
        self.assertIn("代码逻辑优化", rendered)

    def test_prompt_summary_lists_learned_rule_ids(self):
        summary = build_ab_summary("prompt-evolution", learned_rule_ids=["SEC-WEAK-HASH"])
        self.assertIn("SEC-WEAK-HASH", "\n".join(markdown_sections(summary)))

    def test_report_names_have_timestamp_and_content_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = timestamped_run_directory(
                directory, "e2e-diff-ab-comparison", datetime(2026, 8, 11, 9, 30, 5)
            )
            paths = report_paths(run_dir, "e2e-ab-comparison")
            self.assertEqual("20260811-093005_e2e-diff-ab-comparison", os.path.basename(run_dir))
            self.assertTrue(paths["json"].endswith("e2e-ab-comparison.json"))
            self.assertEqual(os.path.dirname(paths["markdown"]), run_dir)


if __name__ == "__main__":
    unittest.main()
