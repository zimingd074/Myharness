"""Run directed, hybrid and all activation policies on the same ten PR fixtures.

This command deliberately uses a configured OpenAI-compatible provider.  It
does not substitute deterministic probes for LLM measurements.  The fixture
snapshot is supplied to the normal queued service path, so retrieval remains
read-only and bounded exactly as in production.
"""
import argparse
import json
import os
import sys
import tempfile


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evoagent.evaluation_benchmark import generate_broadened_context_cases, generate_large_pr_context_cases
from evoagent.evaluation_harness import EndToEndEvaluationHarness
from evoagent.full_chain_evaluation import QueuedServiceReviewer, build_service


METRICS = (
    "precision", "recall", "f1", "high_risk_recall", "changed_file_coverage",
    "changed_hunk_coverage", "coverage_gaps", "average_shards", "cross_shard_findings",
    "input_tokens", "average_input_tokens", "max_input_tokens", "llm_calls", "tool_calls",
    "repo_tool_calls", "retrieved_context_tokens", "context_compaction_count",
    "compressed_rounds", "pinned_evidence_count", "llm_failures",
)


def run_policy(policy: str, cases: list, timeout_seconds: int, checkpoint_path: str = "", retry_errors: bool = False) -> dict:
    handle, database_path = tempfile.mkstemp(prefix="evoagent-activation-%s-" % policy, suffix=".db")
    os.close(handle)
    service = build_service(
        database_path, with_llm=True, llm_timeout_seconds=timeout_seconds,
        async_workers=1, context_architecture="coverage-first",
        specialist_activation=policy,
    )
    reviewer = QueuedServiceReviewer(service, "llm-activation-%s" % policy, timeout_seconds)
    try:
        harness = EndToEndEvaluationHarness()
        # Persist completed case results. Long real-provider experiments can be
        # resumed after a runner/process timeout without pretending missing
        # cases were evaluated.
        cached = {}
        if checkpoint_path and os.path.exists(checkpoint_path):
            with open(checkpoint_path, encoding="utf-8") as handle:
                cached = {item["id"]: item for item in json.load(handle).get("case_results", [])
                          if not retry_errors or not item.get("error")}
        results = []
        for case in cases:
            if case["id"] in cached:
                results.append(cached[case["id"]])
            else:
                results.append(harness._run_case(reviewer, case))
                if checkpoint_path:
                    with open(checkpoint_path, "w", encoding="utf-8") as handle:
                        json.dump({"case_results": results}, handle, ensure_ascii=False, indent=2)
        totals = harness._empty_totals()
        for item in results:
            harness._accumulate(totals, item)
        return {
            "schema_version": 1, "name": policy, "reviewer": reviewer.name,
            "execution_path": reviewer.execution_path, "parallelism": 1,
            "dataset": {"cases": len(cases)}, "metrics": harness._metrics(totals),
            "duration_seconds": 0.0, "case_results": results,
        }
    finally:
        reviewer.close()
        try:
            os.unlink(database_path)
        except OSError:
            pass


def _summary(report: dict) -> dict:
    metrics = report["metrics"]
    per_case = report["case_results"]
    active = [len((case.get("context") or {}).get("shard_assignments", [])) for case in per_case]
    assignments = [
        sum(1 + len(item.get("supplemental_reviewers", []))
            for item in (case.get("context") or {}).get("shard_assignments", []))
        for case in per_case
    ]
    return {
        **{name: metrics.get(name) for name in METRICS},
        "duration_seconds": report["duration_seconds"],
        "average_shards_with_schedule": round(sum(active) / max(1, len(active)), 2),
        "average_activated_reviewers": round(sum(assignments) / max(1, len(assignments)), 2),
        "coverage_statuses": [str((case.get("context") or {}).get("coverage_status", "unknown")) for case in per_case],
    }


def _markdown(result: dict) -> str:
    rows = ["| Policy | Precision | Recall | F1 | File coverage | Hunk coverage | Input tokens | LLM failures | Duration s |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, value in result["policies"].items():
        rows.append("| {name} | {precision} | {recall} | {f1} | {changed_file_coverage} | {changed_hunk_coverage} | {input_tokens} | {llm_failures} | {duration_seconds} |".format(name=name, **value))
    rows.append("")
    rows.append("Single-prompt token size, end-to-end token total, and call counts are intentionally separate: shard coverage can improve recall while increasing end-to-end cost.")
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--fixture-set", choices=("large", "broadened"), default="large")
    parser.add_argument("--policy", choices=("directed", "hybrid", "all"), default="")
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--output", default="")
    parser.add_argument("--markdown-output", default="")
    args = parser.parse_args()
    if args.timeout_seconds < 1:
        parser.error("--timeout-seconds must be positive")
    cases = (generate_broadened_context_cases() if args.fixture_set == "broadened"
             else generate_large_pr_context_cases())
    selected_policies = (args.policy,) if args.policy else ("directed", "hybrid", "all")
    reports = {policy: run_policy(
        policy, cases, args.timeout_seconds,
        os.path.join(args.checkpoint_dir, "%s-%s.json" % (args.fixture_set, policy)) if args.checkpoint_dir else "",
        args.retry_errors,
    )
               for policy in selected_policies}
    result = {
        "dataset": {"cases": len(cases), "kind": "ten-multifile-%s-pr-fixtures" % args.fixture_set},
        "runtime": {"llm": True, "serial": True, "timeout_seconds": args.timeout_seconds},
        "policies": {name: _summary(report) for name, report in reports.items()},
    }
    if {"hybrid", "all"}.issubset(reports):
        hybrid, all_policy = result["policies"]["hybrid"], result["policies"]["all"]
        result["hybrid_acceptance"] = {
            "file_coverage_100": hybrid["changed_file_coverage"] == 1.0,
            "hunk_coverage_100": hybrid["changed_hunk_coverage"] == 1.0,
            "no_coverage_gaps": hybrid["coverage_gaps"] == 0 and all(
                item == "complete" for item in hybrid["coverage_statuses"]
            ),
            "critical_high_recall_within_one_fixture": hybrid["high_risk_recall"] >= all_policy["high_risk_recall"] - .1,
            "precision_within_five_points": hybrid["precision"] >= all_policy["precision"] - .05,
            "tokens_or_calls_reduced_25_percent": (
                hybrid["input_tokens"] <= all_policy["input_tokens"] * .75
                or hybrid["tool_calls"] <= all_policy["tool_calls"] * .75
            ),
        }
        result["hybrid_acceptance"]["passed"] = all(result["hybrid_acceptance"].values())
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    if args.markdown_output:
        with open(args.markdown_output, "w", encoding="utf-8") as handle:
            handle.write(_markdown(result))


if __name__ == "__main__":
    main()
