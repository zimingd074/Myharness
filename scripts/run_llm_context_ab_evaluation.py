"""Run the same ten large PR fixtures through real configured LLM review paths."""
import argparse
import json
import os
import sys
import tempfile


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evoagent.evaluation_benchmark import generate_large_pr_context_cases
from evoagent.evaluation_harness import EndToEndEvaluationHarness
from evoagent.full_chain_evaluation import QueuedServiceReviewer, build_service


FIELDS = (
    "precision", "recall", "f1", "high_risk_recall", "changed_file_coverage",
    "changed_hunk_coverage", "coverage_gaps", "llm_failures", "input_tokens", "average_input_tokens",
    "max_input_tokens", "tool_calls", "repo_tool_calls", "average_shards",
    "cross_shard_findings", "retrieved_context_tokens", "context_compaction_count",
    "compressed_rounds", "pinned_evidence_count",
)


def run_architecture(architecture: str, cases: list, timeout: int) -> dict:
    fd, path = tempfile.mkstemp(prefix="evoagent-llm-%s-" % architecture, suffix=".db")
    os.close(fd)
    service = build_service(
        path, with_llm=True, llm_timeout_seconds=timeout, async_workers=1,
        context_architecture=architecture,
    )
    reviewer = QueuedServiceReviewer(service, "llm-%s" % architecture, timeout)
    try:
        return EndToEndEvaluationHarness().run(reviewer, cases, architecture)
    finally:
        reviewer.close()
        try:
            os.unlink(path)
        except OSError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    if args.timeout_seconds < 1:
        parser.error("--timeout-seconds must be positive")
    cases = generate_large_pr_context_cases()
    reports = {name: run_architecture(name, cases, args.timeout_seconds)
               for name in ("legacy", "coverage-first")}
    baseline, candidate = reports["legacy"], reports["coverage-first"]
    result = {
        "dataset": {"cases": len(cases), "kind": "ten-multifile-large-pr-fixtures"},
        "runtime": {"llm": True, "serial": True, "timeout_seconds": args.timeout_seconds},
        "legacy": {field: baseline["metrics"].get(field) for field in FIELDS},
        "coverage_first": {field: candidate["metrics"].get(field) for field in FIELDS},
        "latency_seconds": {name: report["duration_seconds"] for name, report in reports.items()},
    }
    result["deltas"] = {
        "recall_points": round(result["coverage_first"]["recall"] - result["legacy"]["recall"], 4),
        "single_prompt_peak_tokens_percent": round((
            result["coverage_first"]["max_input_tokens"] / max(1, result["legacy"]["max_input_tokens"]) - 1
        ) * 100, 1),
        "end_to_end_input_tokens_percent": round((
            result["coverage_first"]["input_tokens"] / max(1, result["legacy"]["input_tokens"]) - 1
        ) * 100, 1),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")


if __name__ == "__main__":
    main()
