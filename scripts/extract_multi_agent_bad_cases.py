"""Extract the policy-level failures from the controlled 100-case LLM A/B.

The derived dataset preserves the original expected_findings verbatim.  The
additional diagnostics explain why a case failed the current strict policy
metric; they are not replacement ground truth.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


DATASET_PRIMARY = {
    # The risk fixture adds a target line but also changes `return normalized`
    # to `return value`; that second regression is not labelled.
    "pr-0002", "pr-0004", "pr-0031", "pr-0043", "pr-0053", "pr-0063", "pr-0073",
    # These clean fixtures add unused/throwing conversion logic.
    "pr-0006", "pr-0009", "pr-0010",
    # These clean fixtures are not executable as stored because required names
    # (cursor/os/hashlib) are absent from the complete after_files snapshot.
    "pr-0016", "pr-0018", "pr-0020", "pr-0027", "pr-0028", "pr-0029", "pr-0030",
    "pr-0036", "pr-0037", "pr-0038", "pr-0039", "pr-0040", "pr-0065", "pr-0076",
    "pr-0077", "pr-0078", "pr-0085", "pr-0086", "pr-0087", "pr-0088", "pr-0089",
    "pr-0090",
    # The expected label covers one seeded risk but omits another observable
    # runtime/control-flow defect in the same synthetic file.
    "pr-0042", "pr-0062", "pr-0072", "pr-0082",
}

IMPLEMENTATION_PRIMARY = {
    # Intended clean hard negatives: placeholder tokens and fixture checksums
    # must be rejected by structure/context, not reported as credentials/hash risk.
    "pr-0005", "pr-0015", "pr-0025", "pr-0035", "pr-0045",
    # Same causal defect emitted twice (or at the wrong adjacent locator).
    "pr-0024", "pr-0034",
    # Shared LLM/token budget is exhausted before challenge/revision completes.
    "pr-0051", "pr-0061",
    # CWE-628 admission is too broad and accepts unrelated NameError/TypeError claims.
    "pr-0066", "pr-0083",
}

MIXED_PRIMARY = {
    # The implementation misses the seeded rule, but the fixture does not
    # establish the real-world precondition (money, async, concurrency or
    # untrusted input); REL-NAIVE-DATETIME also uses an unsuitable CWE-367 label.
    "pr-0044", "pr-0054", "pr-0064", "pr-0074", "pr-0084", "pr-0094",
}

DIAGNOSIS_GROUPS = {
    "synthetic-omitted-normalization-regression": {
        "pr-0002", "pr-0004", "pr-0031", "pr-0043", "pr-0053", "pr-0063", "pr-0073",
    },
    "synthetic-clean-runtime-defect": {
        "pr-0006", "pr-0009", "pr-0010", "pr-0016", "pr-0018", "pr-0020",
        "pr-0027", "pr-0028", "pr-0029", "pr-0030", "pr-0036", "pr-0037",
        "pr-0038", "pr-0039", "pr-0040", "pr-0065", "pr-0076", "pr-0077",
        "pr-0078", "pr-0085", "pr-0086", "pr-0087", "pr-0088", "pr-0089",
        "pr-0090",
    },
    "synthetic-secondary-defect-unlabelled": {"pr-0042", "pr-0062", "pr-0072", "pr-0082"},
    "context-insensitive-rule-false-positive": {"pr-0005", "pr-0015", "pr-0025", "pr-0035", "pr-0045"},
    "semantic-duplicate-or-locator-error": {"pr-0024", "pr-0034"},
    "shared-budget-exhaustion": {"pr-0051", "pr-0061"},
    "overbroad-semantic-claim-admission": {"pr-0066", "pr-0083"},
    "weak-or-ambiguous-seeded-label": MIXED_PRIMARY,
}

DIAGNOSIS_TEXT = {
    "synthetic-omitted-normalization-regression": (
        "The generator also changes the function contract from returning normalized to returning value, "
        "but expected_findings labels only the seeded rule."
    ),
    "synthetic-clean-runtime-defect": (
        "The case is labelled clean although its complete synthetic snapshot contains unused/throwing logic "
        "or references a name that is not defined or imported."
    ),
    "synthetic-secondary-defect-unlabelled": (
        "The seeded finding is labelled, but an additional observable runtime/control-flow defect is omitted."
    ),
    "context-insensitive-rule-false-positive": (
        "The deterministic/semantic pipeline treats an explicit placeholder or fixture checksum as a real secret "
        "or security hash use instead of using context as counterevidence."
    ),
    "semantic-duplicate-or-locator-error": (
        "A domain Agent re-emits the same causal defect as another CWE or attaches it to an adjacent wrong line, "
        "so display deduplication and exact matching produce an extra finding."
    ),
    "shared-budget-exhaustion": (
        "Conditional domain review consumes the shared request/token budget before challenge or revision completes."
    ),
    "overbroad-semantic-claim-admission": (
        "The decision policy accepts an unrelated NameError/TypeError hypothesis as CWE-628 based on a generic call shape."
    ),
    "weak-or-ambiguous-seeded-label": (
        "The implementation misses the seeded pattern, but the synthetic diff also lacks the business/trust/concurrency "
        "precondition needed to establish the labelled defect unambiguously."
    ),
}


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _is_escalated(result: dict) -> bool:
    return any(
        item.get("outcome") == "escalate"
        for item in (result.get("context") or {}).get("decisions", [])
    )


def _semantic_valid(result: dict) -> bool:
    context = result.get("context") or {}
    return bool(
        result.get("execution_success")
        and context.get("semantic_review_complete")
        and context.get("budget_compliant")
    )


def _attribution(case_id: str) -> str:
    if case_id in DATASET_PRIMARY:
        return "dataset-fixture-or-label"
    if case_id in IMPLEMENTATION_PRIMARY:
        return "implementation"
    if case_id in MIXED_PRIMARY:
        return "mixed"
    raise ValueError(f"unclassified bad case: {case_id}")


def _diagnosis(case_id: str) -> str:
    matches = [name for name, ids in DIAGNOSIS_GROUPS.items() if case_id in ids]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one diagnosis for {case_id}, got {matches}")
    return matches[0]


def extract(dataset_path: Path, report_path: Path) -> tuple[list[dict], dict]:
    source_cases = {item["id"]: item for item in _load_jsonl(dataset_path)}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    baseline_name = "conditional_primary"
    candidate_name = "conditional_domain_agents"
    baseline = {
        item["id"]: item for item in report["arms"][baseline_name]["case_results"]
    }
    candidate = {
        item["id"]: item for item in report["arms"][candidate_name]["case_results"]
    }
    derived = []
    records = []
    for case_id, result in candidate.items():
        escalated = _is_escalated(result)
        semantic_valid = _semantic_valid(result)
        policy_hit = bool(result["exact_case_hit"] and not escalated and semantic_valid)
        if policy_hit:
            continue
        context = result.get("context") or {}
        failure_classes = []
        if not result["exact_case_hit"]:
            failure_classes.append("final-finding-mismatch")
        if escalated:
            failure_classes.append("claim-escalation")
        if not context.get("semantic_review_complete"):
            failure_classes.append("semantic-review-incomplete")
        if not context.get("budget_compliant"):
            failure_classes.append("budget-violation")
        code = _diagnosis(case_id)
        diagnostic = {
            "source_report": report_path.name,
            "arm": candidate_name,
            "policy_hit": False,
            "raw_exact_case_hit": bool(result["exact_case_hit"]),
            "baseline_raw_exact_case_hit": bool(baseline[case_id]["exact_case_hit"]),
            "tp": int(result["tp"]), "fp": int(result["fp"]), "fn": int(result["fn"]),
            "failure_classes": failure_classes,
            "primary_attribution": _attribution(case_id),
            "diagnosis_code": code,
            "diagnosis": DIAGNOSIS_TEXT[code],
            "effective_models": list(context.get("effective_models") or []),
        }
        copied = dict(source_cases[case_id])
        copied["bad_case_diagnostics"] = diagnostic
        derived.append(copied)
        records.append({"case_id": case_id, **diagnostic})

    expected_ids = DATASET_PRIMARY | IMPLEMENTATION_PRIMARY | MIXED_PRIMARY
    actual_ids = {item["id"] for item in derived}
    if actual_ids != expected_ids:
        raise ValueError(
            "curated attribution does not match report failures: "
            f"missing={sorted(expected_ids - actual_ids)} extra={sorted(actual_ids - expected_ids)}"
        )
    attribution = Counter(item["primary_attribution"] for item in records)
    diagnoses = Counter(item["diagnosis_code"] for item in records)
    failure_classes = Counter(value for item in records for value in item["failure_classes"])
    summary = {
        "schema_version": 1,
        "source_dataset": dataset_path.name,
        "source_report": report_path.name,
        "arm": candidate_name,
        "definition": (
            "A bad case is a candidate case that is not raw-exact, contains an escalated Decision, "
            "or did not complete semantic review within the shared budget."
        ),
        "case_count": len(records),
        "raw_finding_error_cases": sum(not item["raw_exact_case_hit"] for item in records),
        "policy_only_failure_cases": sum(item["raw_exact_case_hit"] for item in records),
        "baseline_raw_error_cases": sum(not item["baseline_raw_exact_case_hit"] for item in records),
        "candidate_regressions": sorted(
            item["case_id"] for item in records
            if item["baseline_raw_exact_case_hit"] and not item["raw_exact_case_hit"]
        ),
        "candidate_improvements": sorted(
            case_id for case_id, item in baseline.items()
            if not item["exact_case_hit"] and candidate[case_id]["exact_case_hit"]
        ),
        "attribution_counts": dict(sorted(attribution.items())),
        "diagnosis_counts": dict(sorted(diagnoses.items())),
        "failure_class_counts": dict(sorted(failure_classes.items())),
        "records": records,
        "ground_truth_policy": (
            "Original expected_findings are preserved. Dataset-attributed and mixed cases require human relabelling "
            "before they can be used as a production quality gate."
        ),
    }
    return derived, summary


def render_markdown(summary: dict) -> str:
    lines = [
        "# Multi-Agent 100-case bad-case audit", "",
        f"Source: `{summary['source_report']}` / `{summary['source_dataset']}`", "",
        "## Counts", "",
        f"- Strict policy failures: **{summary['case_count']}**",
        f"- Final Finding mismatches: **{summary['raw_finding_error_cases']}**",
        f"- Raw-exact but policy-incomplete/escalated: **{summary['policy_only_failure_cases']}**",
        f"- Candidate-only regressions: `{', '.join(summary['candidate_regressions'])}`",
        f"- Candidate improvements: `{', '.join(summary['candidate_improvements'])}`", "",
        "## Primary attribution", "",
        "| Attribution | Cases |", "|---|---:|",
    ]
    lines.extend(
        f"| {name} | {count} |" for name, count in summary["attribution_counts"].items()
    )
    lines.extend(["", "## Diagnosis groups", "", "| Diagnosis | Cases |", "|---|---:|"])
    lines.extend(
        f"| {name} | {count} |" for name, count in summary["diagnosis_counts"].items()
    )
    lines.extend([
        "", "## Interpretation", "",
        "The 53-case strict-policy failure count must not be read as 53 wrong user-facing reports. "
        "Only 17 cases have FP/FN output; the other 36 are penalized because an additional Claim was escalated "
        "or semantic execution was incomplete.", "",
        "Dataset-attributed failures are concentrated in generator artifacts: omitted return-contract changes, "
        "undefined names in snapshots labelled clean, and secondary defects absent from expected_findings. "
        "Implementation-attributed failures remain real regressions: contextual scanner false positives, duplicate "
        "semantic findings, overbroad CWE-628 admission, and shared-budget exhaustion.", "",
        "Do not change expected_findings automatically. Audit dataset/mixed cases with a human, while keeping "
        "implementation cases as regression tests against the current labels.", "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="evaluation_data/pr_diff_100.jsonl")
    parser.add_argument("--report", default="reports/domain_agents_100_real.json")
    parser.add_argument("--output", default="evaluation_data/multi_agent_bad_cases_53.jsonl")
    parser.add_argument("--analysis-json", default="reports/multi_agent_bad_cases_53_analysis.json")
    parser.add_argument("--analysis-markdown", default="reports/multi_agent_bad_cases_53_analysis.md")
    args = parser.parse_args()
    cases, summary = extract(Path(args.dataset), Path(args.report))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in cases),
        encoding="utf-8",
    )
    analysis_json = Path(args.analysis_json)
    analysis_json.parent.mkdir(parents=True, exist_ok=True)
    analysis_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    analysis_markdown = Path(args.analysis_markdown)
    analysis_markdown.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps({
        "bad_cases": len(cases), "dataset": str(output),
        "analysis": str(analysis_json), "markdown": str(analysis_markdown),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
