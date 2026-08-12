"""Run the deterministic 10-case large-PR context architecture A/B evaluation."""
import json
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evoagent.evaluation_benchmark import coverage_ab_reviewer, generate_coverage_ab_cases
from evoagent.evaluation_harness import EndToEndEvaluationHarness


FIELDS = (
    "precision", "recall", "f1", "high_risk_recall",
    "changed_file_coverage", "changed_hunk_coverage", "coverage_gaps",
    "input_tokens", "average_input_tokens", "max_input_tokens",
    "tool_calls", "repo_tool_calls", "average_shards",
)


def main() -> None:
    cases = generate_coverage_ab_cases()
    reports = {
        architecture: EndToEndEvaluationHarness().run(
            coverage_ab_reviewer(architecture), cases, architecture
        )
        for architecture in ("legacy", "coverage-first")
    }
    result = {
        "dataset": {
            "cases": len(cases), "kind": "synthetic-context-management",
            "purpose": "large-PR context visibility and coverage regression",
        },
        "baseline": {key: reports["legacy"]["metrics"][key] for key in FIELDS},
        "coverage_first": {key: reports["coverage-first"]["metrics"][key] for key in FIELDS},
        "duration_seconds": {
            key: reports[key]["duration_seconds"] for key in reports
        },
    }
    baseline = result["baseline"]
    candidate = result["coverage_first"]
    result["deltas"] = {
        "recall_points": round(candidate["recall"] - baseline["recall"], 4),
        "file_coverage_points": round(candidate["changed_file_coverage"] - baseline["changed_file_coverage"], 4),
        "hunk_coverage_points": round(candidate["changed_hunk_coverage"] - baseline["changed_hunk_coverage"], 4),
        "total_input_tokens_percent": round(
            (candidate["input_tokens"] / baseline["input_tokens"] - 1) * 100, 1
        ),
        "max_input_tokens_percent": round(
            (candidate["max_input_tokens"] / baseline["max_input_tokens"] - 1) * 100, 1
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
