"""Shared, auditable A/B comparison report metadata and rendering helpers."""
from datetime import datetime
import os
import re
from typing import Any, Dict, Iterable, List, Optional


_MANIFESTS = {
    "e2e": {
        "baseline": {
            "name": "single-agent-baseline",
            "description": "A deterministic LocalRuleReviewer that applies local rules to added Diff lines.",
        },
        "candidate": {
            "name": "multi-agent-candidate",
            "description": "A MultiAgentCoordinator that combines the local reviewer with a context security and reliability specialist.",
        },
        "technologies": [
            "Queued ReviewService execution with task, trace, and final-report persistence.",
            "Multi-agent finding aggregation with deterministic de-duplication.",
            "Context-specific security and reliability rule review on changed lines.",
            "One-to-one matching by path, CWE, and changed-line location.",
        ],
        "features": [
            "Adds a context-security-reliability specialist to the baseline reviewer.",
            "Runs each case through enqueue, queue consumption, ReviewHarness, and persisted ReviewReport.",
            "Evaluates Validation and repository-isolated Holdout splits separately.",
        ],
        "optimizations": [
            "Expands coverage for context-sensitive risks without changing metric definitions.",
            "Merges findings by rule, path, and line to avoid duplicate agent output.",
        ],
    },
    "prompt-evolution": {
        "baseline": {
            "name": "Prompt v1",
            "description": "The active default prompt before feedback-driven evolution.",
        },
        "candidate": {
            "name": "Prompt v2",
            "description": "The feedback-derived prompt version activated only after replay gates pass.",
        },
        "technologies": [
            "Feedback-driven prompt version generation with version and SHA-256 provenance.",
            "Validation replay followed by repository-isolated Holdout non-regression checks.",
            "Persisted evolution runs, activation decisions, and reproducibility fingerprints.",
        ],
        "features": [
            "Uses confirmed Validation false-negative feedback as the evolution signal.",
            "Records learned rule IDs and resolves feedback only after activation.",
            "Blocks production activation without independently labelled public PR data.",
        ],
        "optimizations": [
            "Focuses the candidate prompt on confirmed misses instead of changing model weights.",
            "Protects quality with minimum Validation improvement and Holdout non-regression gates.",
        ],
    },
}


def rule_catalog(rules: Iterable[tuple]) -> List[Dict[str, str]]:
    values = []
    for rule in rules:
        values.append({"rule_id": str(rule[0]), "severity": rule[1].value})
    return sorted(values, key=lambda item: item["rule_id"])


def build_ab_summary(
    experiment: str,
    baseline_rules: Optional[Iterable[Dict[str, str]]] = None,
    introduced_rules: Optional[Iterable[Dict[str, str]]] = None,
    learned_rule_ids: Optional[Iterable[str]] = None,
    removed_rule_ids: Optional[Iterable[str]] = None,
    llm: Optional[Dict[str, object]] = None,
) -> Dict[str, Any]:
    if experiment not in _MANIFESTS:
        raise ValueError("unsupported A/B report experiment: %s" % experiment)
    manifest = _MANIFESTS[experiment]
    candidate = dict(manifest["candidate"])
    runtime_llm = {}
    if llm:
        candidate["description"] += " It also includes the configured LLM review specialist."
        runtime_llm = {
            "enabled": True,
            "provider": str(llm.get("provider", "")),
            "model": str(llm.get("model", "")),
        }
    return {
        "experiment": experiment,
        "baseline": dict(manifest["baseline"]),
        "candidate": candidate,
        "technologies": list(manifest["technologies"]),
        "features": list(manifest["features"]),
        "logic_optimizations": list(manifest["optimizations"]),
        "rule_changes": {
            "baseline_rules": sorted(list(baseline_rules or []), key=lambda item: item["rule_id"]),
            "introduced_rules": sorted(list(introduced_rules or []), key=lambda item: item["rule_id"]),
            "learned_rule_ids": sorted(set(learned_rule_ids or [])),
            "removed_rule_ids": sorted(set(removed_rule_ids or [])),
        },
        "runtime_llm": runtime_llm,
    }


def markdown_sections(summary: Dict[str, Any]) -> List[str]:
    rules = summary["rule_changes"]
    lines = [
        "## A/B 方案说明", "", "### 基线", "",
        "- **%s**：%s" % (summary["baseline"]["name"], summary["baseline"]["description"]),
        "", "### 修改后方案", "",
        "- **%s**：%s" % (summary["candidate"]["name"], summary["candidate"]["description"]),
        "", "### 引入的技术", "",
    ]
    lines.extend("- %s" % item for item in summary["technologies"])
    if summary.get("runtime_llm", {}).get("enabled"):
        lines.append(
            "- Runtime LLM: `%s` / `%s`" % (
                summary["runtime_llm"]["provider"], summary["runtime_llm"]["model"],
            )
        )
    lines.extend(["", "### 新增功能", ""])
    lines.extend("- %s" % item for item in summary["features"])
    lines.extend(["", "### 代码逻辑优化", ""])
    lines.extend("- %s" % item for item in summary["logic_optimizations"])
    lines.extend(["", "### 规则变化", ""])
    if rules["baseline_rules"]:
        lines.append(
            "- 基线规则（%d）：`%s`" % (
                len(rules["baseline_rules"]),
                "`, `".join(item["rule_id"] for item in rules["baseline_rules"]),
            )
        )
    if rules["introduced_rules"]:
        lines.append(
            "- 候选新增规则（%d）：%s" % (
                len(rules["introduced_rules"]),
                "; ".join(
                    "`%s` (%s)" % (item["rule_id"], item["severity"])
                    for item in rules["introduced_rules"]
                ),
            )
        )
    if rules["learned_rule_ids"]:
        lines.append("- 本次自进化学习到的检查焦点：`%s`" % "`, `".join(rules["learned_rule_ids"]))
    if rules["removed_rule_ids"]:
        lines.append("- 本次自进化移除的规则：`%s`" % "`, `".join(rules["removed_rule_ids"]))
    if not any(rules.values()):
        lines.append("- 本次运行没有规则集合变化。")
    return lines


def timestamped_run_directory(
    output_dir: str, modification_summary: str, now: Optional[datetime] = None,
) -> str:
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    brief = re.sub(r"[^a-z0-9-]+", "-", modification_summary.lower()).strip("-")
    if not brief:
        raise ValueError("modification_summary must contain letters or digits")
    run_dir = os.path.join(output_dir, "%s_%s" % (stamp, brief))
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


def report_paths(run_dir: str, content_summary: str) -> Dict[str, str]:
    brief = re.sub(r"[^a-z0-9-]+", "-", content_summary.lower()).strip("-")
    if not brief:
        raise ValueError("content_summary must contain letters or digits")
    base = os.path.join(run_dir, brief)
    return {"json": base + ".json", "markdown": base + ".md"}
