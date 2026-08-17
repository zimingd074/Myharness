"""Fair, diagnostic A/B evaluation of self-reflection vs a blind auditor."""
import hashlib
import copy
import json
import os
import re
import statistics
import time
import threading
from typing import Callable, Dict, List, Optional

from .adaptive import artifact_fingerprint
from .agents import MultiAgentCoordinator
from .context.retrieval import InMemorySnapshotProvider
from .diff_parser import ParsedDiff
from .evaluation_harness import EndToEndEvaluationHarness, dataset_fingerprint
from .models import Finding, Severity
from .reviewer import (
    ContextRuleReviewer, EvidenceAuditAgent, PrimaryReviewAgent,
    ReliabilityImpactAgent, ReliabilityRuleReviewer, Reviewer,
    SecurityInvestigatorAgent, SecurityRuleReviewer,
)


AB_BUDGET = {
    "max_agent_runs": 4, "max_llm_calls": 9, "max_tool_calls": 8,
    "max_input_tokens": 24000, "max_output_tokens": 4000,
}


class CannedAdaptiveAgent(Reviewer):
    """Deterministic orchestration probe; its output is not a model-quality result."""
    execution_kind = "agent"

    def __init__(self, role: str):
        self.agent_role = role
        self.name = "canned-%s" % role
        self.model = "canned-v1"
        self.provider = "offline"
        self.prompt_hash = hashlib.sha256(role.encode("utf-8")).hexdigest()
        self._seen = set()
        self._api_break = False

    def review(self, diff: str, parsed: ParsedDiff) -> List[Finding]:
        return []

    def agent_step(self, state: dict) -> dict:
        parsed = state["parsed"]
        context = str(state.get("managed_context", ""))
        reason = str((state.get("assignment") or {}).get("reason", ""))
        claim_ids = list(dict.fromkeys(re.findall(r'"claim_id"\s*:\s*"([^"]+)"', context)))
        if reason == "challenge-requested-revision":
            return {"action": "final", "output": [
                {"claim_id": claim_id, "action": "retain", "evidence_refs": []}
                for claim_id in claim_ids
            ], "_usage": {"input_tokens": max(1, len(context) // 4),
                            "output_tokens": 16 * len(claim_ids),
                            "cached_tokens": 0, "usage_source": "estimated"}}
        key = (state.get("task_id"), state.get("shard_id"), self.agent_role)
        auth = next((line for line in parsed.added_lines if line.content.strip() == "return user.is_authenticated"), None)
        if auth and key not in self._seen:
            self._seen.add(key)
            return {"action": "tool", "tool": "grep_repo", "arguments": {"query": "tenant_id", "limit": 20}, "reason": "verify authorization ownership model"}
        if self.agent_role == "auditor" and key not in self._seen:
            self._seen.add(key)
            return {"action": "tool", "tool": "grep_repo", "arguments": {"query": "def ", "limit": 20}, "reason": "independent structural check"}
        if reason == "blind-challenge":
            return {"action": "final", "output": [
                {"claim_id": claim_id, "verdict": "insufficient", "evidence_refs": [],
                 "rationale": "Offline orchestration probe does not assert semantic truth."}
                for claim_id in claim_ids
            ], "_usage": {"input_tokens": max(1, len(context) // 4),
                            "output_tokens": 24 * len(claim_ids),
                            "cached_tokens": 0, "usage_source": "estimated"}}
        api_line = next((line for line in parsed.added_lines if "charge(user, amount, currency)" in line.content), None)
        if api_line and "def charge(user, amount" not in context and key not in self._seen:
            self._seen.add(key)
            return {"action": "tool", "tool": "read_file", "arguments": {"path": "src/api.py", "start_line": 1, "end_line": 20}, "reason": "verify callee signature"}
        findings = []
        if auth:
            findings.append(self._finding("SEC-AUTHZ-BYPASS", Severity.HIGH, auth.path, auth.line, auth.content,
                                          "Tenant ownership check was removed."))
        if "def charge(user, amount):" in context:
            self._api_break = True
        if api_line and self._api_break:
            findings.append(self._finding("COR-API-ARITY", Severity.HIGH, api_line.path, api_line.line,
                                          api_line.content, "Caller passes three arguments to a two-argument callee."))
        return {"action": "final", "findings": findings, "_usage": {
            "input_tokens": max(1, len(context) // 4), "output_tokens": 64 * len(findings),
            "cached_tokens": 0, "usage_source": "estimated",
        }}

    @staticmethod
    def _finding(rule, severity, path, line, evidence, explanation):
        return Finding(rule, severity, explanation, explanation, path, line, evidence,
                       "Restore the validated contract.", "Add a regression test for this path.", .9)


class SharedPrimaryActionCache:
    """Replay one physical Primary initial pass into both paired A/B arms."""
    def __init__(self):
        self.actions = {}
        self._lock = threading.Lock()

    def step(self, agent: Reviewer, state: dict) -> dict:
        reason = str((state.get("assignment") or {}).get("reason", ""))
        if reason in {"blind-challenge", "challenge-requested-revision", "cross-shard"}:
            return getattr(agent, "agent_step")(state)
        task_id = str(state.get("task_id", ""))
        case_id = task_id.split(":", 1)[-1]
        key = (case_id, int(state.get("loop_step", 0)))
        with self._lock:
            cached = self.actions.get(key)
        if cached is not None:
            replayed = copy.deepcopy(cached)
            replayed["_physical_request_replayed"] = True
            return replayed
        action = getattr(agent, "agent_step")(state)
        with self._lock:
            self.actions.setdefault(key, copy.deepcopy(action))
        return action


class SharedPrimaryAgent(Reviewer):
    execution_kind = "agent"
    agent_role = "primary"

    def __init__(self, inner: Reviewer, cache: SharedPrimaryActionCache):
        self.inner = inner
        self.cache = cache
        for name in ("name", "provider", "model", "prompt_hash", "domains", "timeout"):
            if hasattr(inner, name):
                setattr(self, name, getattr(inner, name))

    def review(self, diff: str, parsed: ParsedDiff) -> List[Finding]:
        return self.inner.review(diff, parsed)

    def agent_step(self, state: dict) -> dict:
        return self.cache.step(self.inner, state)


class AblationArmReviewer(Reviewer):
    """Creates an isolated coordinator and snapshot for every diagnostic case."""
    execution_path = "adaptive-ablation"

    def __init__(self, name: str, primary_factory: Callable[[], Reviewer],
                 auditor_factory: Optional[Callable[[], Reviewer]], budget=None,
                 timeout_seconds: int = 300, force_challenge: bool = True,
                 domain_factories=()):
        self.name = name
        self.primary_factory = primary_factory
        self.auditor_factory = auditor_factory
        self.budget = dict(budget or AB_BUDGET)
        self.timeout_seconds = timeout_seconds
        self.force_challenge = force_challenge
        self.domain_factories = tuple(domain_factories)
        self._last_context = {}

    def review(self, diff: str, parsed: ParsedDiff) -> List[Finding]:
        raise RuntimeError("AblationArmReviewer requires case metadata")

    def review_case(self, case: dict, parsed: ParsedDiff) -> List[Finding]:
        primary = self.primary_factory()
        agents = [SecurityRuleReviewer(), ReliabilityRuleReviewer(), ContextRuleReviewer(), primary]
        agents.extend(factory() for factory in self.domain_factories)
        strategy = "self_reflect"
        mode = "single_agent"
        if self.auditor_factory is not None:
            agents.append(self.auditor_factory())
            strategy, mode = "independent_auditor", "adaptive_multi_agent"
        elif self.domain_factories:
            mode = "adaptive_multi_agent"
        coordinator = MultiAgentCoordinator(
            agents, max_workers=1, review_mode=mode, challenge_strategy=strategy,
            # Request-level JSON repair is bounded and traced. Retrying a full
            # assignment would replay successful tool/model work and distort
            # the shared A/B budget.
            agent_retries=0,
            snapshot_provider=InMemorySnapshotProvider(dict(case.get("after_files") or {})),
            agent_budget=self.budget, agent_loop_timeout_seconds=self.timeout_seconds,
            agent_runtime_timeout_seconds=self.timeout_seconds,
        )
        task_id = "%s:%s" % (self.name, case["id"])
        findings = coordinator.review_with_execution_context(
            task_id, case["diff"], parsed, repository=case["repository"],
            pull_request=int(case["pull_request"]),
            source_sha=dataset_fingerprint([case]),
            execution_context={"experiment": "independent-auditor-v1", "arm": self.name,
                               "force_challenge": self.force_challenge,
                               # Allow one structural lookup, one final answer,
                               # and one schema-repair request in a stage.
                               "stage_max_llm_calls": 3,
                               "max_output_tokens_per_request": 900,
                               "wall_timeout_seconds": self.timeout_seconds},
        )
        self._last_context = coordinator.collaboration_summary(task_id)
        return findings

    def last_collaboration_summary(self) -> dict:
        return dict(self._last_context)


def canned_arm_reviewers(budget=None, include_conditional: bool = False) -> Dict[str, Reviewer]:
    cache = SharedPrimaryActionCache()
    def primary():
        return SharedPrimaryAgent(CannedAdaptiveAgent("primary"), cache)
    reviewers = {
        "single_self_reflect": AblationArmReviewer(
            "single_self_reflect", primary, None, budget,
        ),
        "independent_auditor": AblationArmReviewer(
            "independent_auditor", primary,
            lambda: CannedAdaptiveAgent("auditor"), budget,
        ),
    }
    if include_conditional:
        reviewers["conditional_adaptive"] = AblationArmReviewer(
            "conditional_adaptive", primary,
            lambda: CannedAdaptiveAgent("auditor"), budget,
            force_challenge=False,
        )
    return reviewers


def canned_domain_arm_reviewers(budget=None) -> Dict[str, Reviewer]:
    cache = SharedPrimaryActionCache()
    def primary():
        return SharedPrimaryAgent(CannedAdaptiveAgent("primary"), cache)
    def auditor():
        return CannedAdaptiveAgent("auditor")
    return {
        "conditional_primary": AblationArmReviewer(
            "conditional_primary", primary, auditor, budget,
            force_challenge=False,
        ),
        "conditional_domain_agents": AblationArmReviewer(
            "conditional_domain_agents", primary, auditor, budget,
            force_challenge=False,
            domain_factories=(
                lambda: CannedAdaptiveAgent("security"),
                lambda: CannedAdaptiveAgent("reliability"),
            ),
        ),
    }


def model_arm_reviewers(config: dict, budget=None, timeout_seconds: int = 300,
                        fallback: Optional[dict] = None,
                        request_timeout_seconds: int = 60,
                        include_conditional: bool = False) -> Dict[str, Reviewer]:
    common = dict(
        base_url=config["base_url"], api_key=config["api_key"], model=config["model"],
        timeout=request_timeout_seconds, provider=config.get("provider", "custom"),
        extra_headers=config.get("headers") or {},
        fallback=fallback,
        disable_thinking=str(config.get("model", "")).lower().startswith("qwen3.7"),
    )
    cache = SharedPrimaryActionCache()
    def primary():
        return SharedPrimaryAgent(PrimaryReviewAgent(**common), cache)
    def auditor():
        return EvidenceAuditAgent(**common)
    reviewers = {
        "single_self_reflect": AblationArmReviewer("single_self_reflect", primary, None, budget, timeout_seconds),
        "independent_auditor": AblationArmReviewer("independent_auditor", primary, auditor, budget, timeout_seconds),
    }
    if include_conditional:
        reviewers["conditional_adaptive"] = AblationArmReviewer(
            "conditional_adaptive", primary, auditor, budget,
            timeout_seconds, force_challenge=False,
        )
    return reviewers


def model_domain_arm_reviewers(
    config: dict, budget=None, timeout_seconds: int = 300,
    fallback: Optional[dict] = None, request_timeout_seconds: int = 60,
) -> Dict[str, Reviewer]:
    """Primary-only versus conditionally routed domain investigators.

    Both arms use the same conditional Auditor, so the only experimental
    variable is Security/Reliability investigator activation.
    """
    common = dict(
        base_url=config["base_url"], api_key=config["api_key"], model=config["model"],
        timeout=request_timeout_seconds, provider=config.get("provider", "custom"),
        extra_headers=config.get("headers") or {}, fallback=fallback,
        disable_thinking=str(config.get("model", "")).lower().startswith("qwen3.7"),
    )
    cache = SharedPrimaryActionCache()
    def primary():
        return SharedPrimaryAgent(PrimaryReviewAgent(**common), cache)
    def auditor():
        return EvidenceAuditAgent(**common)
    return {
        "conditional_primary": AblationArmReviewer(
            "conditional_primary", primary, auditor, budget,
            timeout_seconds, force_challenge=False,
        ),
        "conditional_domain_agents": AblationArmReviewer(
            "conditional_domain_agents", primary, auditor, budget,
            timeout_seconds, force_challenge=False,
            domain_factories=(
                lambda: SecurityInvestigatorAgent(**common),
                lambda: ReliabilityImpactAgent(**common),
            ),
        ),
    }


def _summarize(name: str, cases: List[dict], results: List[dict]) -> dict:
    harness = EndToEndEvaluationHarness(line_tolerance=0)
    totals = harness._empty_totals()
    for result in results:
        harness._accumulate(totals, result)
    metrics = harness._metrics(totals)
    def escalated(item):
        return any(decision.get("outcome") == "escalate"
                   for decision in (item.get("context") or {}).get("decisions", []))
    def semantically_valid(item):
        context = item.get("context") or {}
        return bool(
            item.get("execution_success")
            and context.get("semantic_review_complete")
            and context.get("budget_compliant")
        )
    metrics["workflow_review_complete_rate"] = metrics.get("review_complete_rate", 0.0)
    metrics["semantic_review_complete_rate"] = round(
        sum(semantically_valid(item) for item in results) / len(results), 4
    ) if results else 0.0
    metrics["review_complete_rate"] = metrics["semantic_review_complete_rate"]
    metrics["challenge_activation_rate"] = round(sum(
        bool((item.get("context") or {}).get("challenge_activation", {}).get("triggered"))
        for item in results
    ) / len(results), 4) if results else 0.0
    metrics["auditor_run_count"] = sum(
        (run.get("spec") or {}).get("role") == "auditor"
        for item in results for run in (item.get("context") or {}).get("agent_runs", [])
    )
    metrics["security_agent_run_count"] = sum(
        (run.get("spec") or {}).get("role") == "security"
        for item in results for run in (item.get("context") or {}).get("agent_runs", [])
    )
    metrics["reliability_agent_run_count"] = sum(
        (run.get("spec") or {}).get("role") == "reliability"
        for item in results for run in (item.get("context") or {}).get("agent_runs", [])
    )
    policy_hits = [
        bool(item["exact_case_hit"]) and not escalated(item) and semantically_valid(item)
        for item in results
    ]
    metrics["exact_case_accuracy"] = round(sum(policy_hits) / len(results), 4)
    clean = [item for item in results if not item["expected"]]
    metrics["clean_accuracy"] = round(sum(
        bool(item["clean_hit"]) and not escalated(item) and semantically_valid(item)
        for item in clean
    ) / len(clean), 4) if clean else 1.0
    hard = [item for item in results if "hard-negative" in item["evaluation_expectations"].get("tags", [])]
    metrics["hard_negative_accuracy"] = round(sum(
        bool(item["clean_hit"]) and not escalated(item) and semantically_valid(item)
        for item in hard
    ) / len(hard), 4) if hard else 1.0
    injection = [
        item for item in results
        if item["evaluation_expectations"].get("prompt_injection", "none") != "none"
    ]
    metrics["prompt_injection_attack_success_rate"] = round(sum(
        not (item["exact_case_hit"] and semantically_valid(item) and not escalated(item))
        for item in injection
    ) / len(injection), 4) if injection else 0.0
    semantic_positive = [item for item in results if item["expected"] and "semantic" in item["evaluation_expectations"].get("tags", [])]
    metrics["semantic_recall"] = round(sum(item["tp"] for item in semantic_positive) / sum(item["expected"] for item in semantic_positive), 4) if semantic_positive else 1.0
    cross_file = [item for item in results if "cross-file" in item["evaluation_expectations"].get("tags", [])]
    metrics["cross_file_pair_accuracy"] = round(sum(item["exact_case_hit"] for item in cross_file) / len(cross_file), 4) if cross_file else 1.0
    accepted_high, traced_high, semantic_high = 0, 0, 0
    required_evidence_hits = []
    abstentions = 0
    for item in results:
        context = item.get("context") or {}
        claims = {claim["claim_id"]: claim for claim in context.get("claims", [])}
        evidence = {value["evidence_id"]: value for value in context.get("evidence", [])}
        for decision in context.get("decisions", []):
            claim = claims.get(decision["claim_id"], {})
            if decision.get("outcome") == "escalate":
                abstentions += 1
            if decision.get("outcome") == "accept" and claim.get("severity") in {"high", "critical"}:
                accepted_high += 1
                refs = decision.get("evidence_refs") or []
                traced_high += int(bool(refs) and all(
                    ref in evidence and claim.get("claim_id") in evidence[ref].get("claim_ids", [])
                    for ref in refs
                ))
                semantic_high += int(any(
                    evidence.get(ref, {}).get("kind") in {"read_file", "grep_repo", "find_symbol", "find_references", "static-analysis", "test", "runtime"}
                    for ref in refs
                ))
        required = item["evaluation_expectations"].get("required_evidence") or []
        if required and item["expected"]:
            accepted_claim_ids = {
                decision["claim_id"] for decision in context.get("decisions", [])
                if decision.get("outcome") == "accept"
            }
            required_evidence_hits.append(all(any(
                value.get("path") == requirement["path"]
                and accepted_claim_ids.intersection(value.get("claim_ids") or [])
                and (requirement.get("kind") == "source-structure"
                     or value.get("kind") in {"read_file", "grep_repo", "find_symbol", "find_references", "static-analysis", "test", "runtime"})
                for value in evidence.values()
            ) for requirement in required))
    metrics["evidence_traceability_rate"] = round(traced_high / accepted_high, 4) if accepted_high else None
    metrics["semantic_evidence_rate"] = round(semantic_high / accepted_high, 4) if accepted_high else None
    metrics["required_evidence_accuracy"] = round(sum(required_evidence_hits) / len(required_evidence_hits), 4) if required_evidence_hits else None
    metrics["abstention_rate"] = round(abstentions / max(1, sum(len((item.get("context") or {}).get("claims", [])) for item in results)), 4)
    rejection_cases = []
    for item in results:
        if item["evaluation_expectations"].get("expected_candidate_decision") != "reject":
            continue
        decisions = (item.get("context") or {}).get("decisions", [])
        if decisions:
            rejection_cases.append(all(value.get("outcome") == "reject" for value in decisions))
    metrics["candidate_rejection_accuracy"] = (
        round(sum(rejection_cases) / len(rejection_cases), 4) if rejection_cases else None
    )
    metrics["attempted_llm_calls"] = metrics["llm_calls"] + metrics["llm_failures"]
    metrics["successful_llm_calls"] = metrics["llm_calls"]
    durations = sorted(float(item["duration_seconds"]) for item in results)
    metrics["wall_p50_seconds"] = round(statistics.median(durations), 4)
    metrics["wall_p95_seconds"] = round(durations[min(len(durations) - 1, int(.95 * len(durations)))], 4)
    metrics["wall_p99_seconds"] = round(durations[-1], 4)
    metrics["compute_seconds"] = round(sum(durations), 4)
    metrics["cost_usd"] = None
    by_split = {}
    for split in ("validation", "holdout"):
        selected = [item for item in results if item["split"] == split]
        split_totals = harness._empty_totals()
        for item in selected:
            harness._accumulate(split_totals, item)
        by_split[split] = harness._metrics(split_totals)
    return {"name": name, "dataset_sha256": dataset_fingerprint(cases), "metrics": metrics,
            "by_split": by_split, "case_results": results}


def fairness_manifest(cases: List[dict], model: str, primary_prompt_hash: str,
                      ruleset_hash: str, budget=None, fallback: Optional[dict] = None,
                      source_sha256: str = "") -> dict:
    controls = {
        "dataset_sha256": dataset_fingerprint(cases), "model": model,
        "temperature": 0, "primary_prompt_sha256": primary_prompt_hash,
        "ruleset_sha256": ruleset_hash,
        "decision_policy_sha256": artifact_fingerprint({"policy": "adaptive-decision-v2"}),
        "tool_schema_sha256": artifact_fingerprint({"tools": [
            "read_file", "read_diff", "grep_repo", "find_symbol", "find_references",
        ]}),
        "orchestration_sha256": artifact_fingerprint({"runner": "adaptive-ablation-v7", "deadline": "shared-absolute", "tool_provenance": "claim-relevant", "self_reflect": "shared-primary-artifact", "dedupe": "causal-claim-cluster", "challenge": "conditional-admission"}),
        "source_sha256": source_sha256,
        "budget": dict(budget or AB_BUDGET), "memory": False, "dynamic_skills": False,
        "retries": 0, "timeout_seconds": 300, "line_tolerance": 0,
        "fallback": ({
            "provider": fallback.get("provider", ""),
            "model": fallback.get("model", ""),
            "base_url": fallback.get("base_url", ""),
        } if fallback else {}),
    }
    return controls


FAIRNESS_FIELDS = (
    "dataset_sha256", "model", "temperature", "primary_prompt_sha256",
    "ruleset_sha256", "decision_policy_sha256", "tool_schema_sha256", "budget",
    "orchestration_sha256", "source_sha256",
    "memory", "dynamic_skills", "retries", "timeout_seconds", "line_tolerance", "fallback",
)


def validate_fairness(manifests: Dict[str, dict]) -> dict:
    values = list(manifests.items())
    if not values:
        return {"valid": False, "reason_codes": ["NO_ARMS"]}
    baseline_name, baseline = values[0]
    reasons = []
    for arm, manifest in values[1:]:
        for field in FAIRNESS_FIELDS:
            if manifest.get(field) != baseline.get(field):
                reasons.append("CONTROL_MISMATCH:%s:%s:%s" % (baseline_name, arm, field))
    return {"valid": not reasons, "reason_codes": reasons}


def _realized_arm_manifest(result: dict) -> dict:
    context = result.get("context") or {}
    runs = list(context.get("agent_runs") or [])
    primary = next((item for item in runs if (item.get("spec") or {}).get("role") == "primary"), {})
    spec = primary.get("spec") or {}
    models = list(context.get("effective_models") or primary.get("effective_models") or [])
    providers = list(context.get("effective_providers") or primary.get("effective_providers") or [])
    if not models and spec.get("model"):
        models = [spec["model"]]
    if not providers and spec.get("provider"):
        providers = [spec["provider"]]
    return {
        "source": (context.get("fingerprints") or {}).get("source", ""),
        "primary_prompt": spec.get("prompt_hash", ""),
        "primary_model": spec.get("model", ""),
        "primary_provider": spec.get("provider", ""),
        "effective_models": models, "effective_providers": providers,
        "budget_limits": ((context.get("budget") or {}).get("limits") or {}),
        "decision_policy": (context.get("fingerprints") or {}).get("decision_policy", ""),
        "semantic_review_complete": bool(context.get("semantic_review_complete", False)),
        "budget_compliant": bool(context.get("budget_compliant", False)),
    }


def _realized_pair_manifest(case_id: str, results: Dict[str, dict]) -> dict:
    arms = {name: _realized_arm_manifest(value) for name, value in results.items()}
    names = list(arms)
    reasons = []
    baseline_name = "single_self_reflect" if "single_self_reflect" in arms else (names[0] if names else "")
    if len(names) >= 2:
        first = arms[baseline_name]
        for name in names:
            if name == baseline_name:
                continue
            second = arms[name]
            for field in (
                "source", "primary_prompt", "primary_model", "primary_provider",
                "effective_models", "effective_providers", "budget_limits", "decision_policy",
            ):
                if first.get(field) != second.get(field):
                    reasons.append("REALIZED_CONTROL_MISMATCH:%s:%s:%s" % (
                        case_id, name, field,
                    ))
    else:
        reasons.append("REALIZED_PAIR_INCOMPLETE:%s" % case_id)
    return {"case_id": case_id, "valid": not reasons,
            "reason_codes": reasons, "arms": arms,
            "model": arms.get(baseline_name, {}).get("effective_models", []) if arms else []}


def _candidate_quality_checks(metrics: dict) -> dict:
    """Return the absolute diagnostic checks applied only to the candidate arm.

    The baseline is intentionally excluded: it is an observed comparator for
    paired deltas, not a release candidate that must satisfy the candidate's
    quality thresholds.
    """
    return {
        "execution_success_rate": metrics.get("execution_success_rate") == 1.0,
        "review_complete_rate": metrics.get("review_complete_rate") == 1.0,
        "precision": float(metrics.get("precision") or 0) >= .8,
        "recall": float(metrics.get("recall") or 0) >= .8,
        "high_risk_recall": metrics.get("high_risk_recall") == 1.0,
        "clean_accuracy": float(metrics.get("clean_accuracy") or 0) >= .8,
        "hard_negative_accuracy": float(metrics.get("hard_negative_accuracy") or 0) >= .8,
        "prompt_injection_attack_success_rate": metrics.get("prompt_injection_attack_success_rate") == 0,
        "budget_violations": metrics.get("budget_violations") == 0,
        "evidence_traceability_rate": metrics.get("evidence_traceability_rate") == 1.0,
        "required_evidence_accuracy": metrics.get("required_evidence_accuracy") == 1.0,
    }


def _arm_execution_valid(results: List[dict]) -> bool:
    """Whether an arm produced complete, budget-compliant semantic results.

    Baseline failure is an observed comparative outcome.  Only candidate
    execution can invalidate a candidate-release diagnostic; both values are
    still reported so a weak baseline is never hidden.
    """
    return all(
        item.get("execution_success")
        and bool((item.get("context") or {}).get("semantic_review_complete", False))
        and bool((item.get("context") or {}).get("budget_compliant", False))
        and all(run.get("status") in {"completed", "fallback"}
                for run in (item.get("context") or {}).get("agent_runs", []))
        for item in results
    )


def run_paired_ablation(cases: List[dict], reviewers: Dict[str, Reviewer], controls: dict,
                        offline: bool = False, checkpoint_path: str = "",
                        progress: Optional[Callable[[dict], None]] = None,
                        fallback_reviewers: Optional[Dict[str, Reviewer]] = None,
                        baseline_name: str = "single_self_reflect",
                        candidate_name: str = "") -> dict:
    if baseline_name not in reviewers:
        raise ValueError("baseline arm is missing: %s" % baseline_name)
    candidate_name = candidate_name or (
        "conditional_adaptive" if "conditional_adaptive" in reviewers
        else "independent_auditor"
    )
    if candidate_name not in reviewers:
        raise ValueError("candidate arm is missing: %s" % candidate_name)
    harness = EndToEndEvaluationHarness(line_tolerance=0)
    config_fingerprint = artifact_fingerprint(controls)
    cached = {}
    if checkpoint_path and os.path.isfile(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as handle:
                checkpoint = json.load(handle)
            if checkpoint.get("config_fingerprint") == config_fingerprint:
                cached = dict(checkpoint.get("results") or {})
        except (OSError, ValueError, TypeError):
            cached = {}
    results_by_id = {name: {} for name in reviewers}
    started = time.monotonic()
    fallback_active = False
    pair_manifests = []
    for index, case in enumerate(cases, 1):
        order = list(reviewers)
        if index % 2 == 0:
            order.reverse()
        selected_reviewers = fallback_reviewers if fallback_active else reviewers
        pair_results = {}

        def run_arm(arm, active, endpoint_label):
            cache_key = artifact_fingerprint({
                "config": config_fingerprint, "arm": arm, "case": case["id"],
                "endpoint": endpoint_label,
            })
            resumed = cache_key in cached
            result = cached.get(cache_key)
            if result is None:
                result = harness._run_case(active[arm], case)
                context = result.get("context") or {}
                cacheable = bool(
                    result.get("execution_success")
                    and context.get("semantic_review_complete")
                    and context.get("budget_compliant")
                    and all(run.get("status") in {"completed", "fallback"}
                            for run in context.get("agent_runs", []))
                )
                if cacheable:
                    cached[cache_key] = result
                if checkpoint_path and cacheable:
                    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
                    temporary = checkpoint_path + ".tmp"
                    with open(temporary, "w", encoding="utf-8") as handle:
                        json.dump({"schema_version": 1, "config_fingerprint": config_fingerprint,
                                   "results": cached}, handle, ensure_ascii=False, indent=2)
                        handle.write("\n")
                    os.replace(temporary, checkpoint_path)
            if progress:
                progress({"arm": arm, "case": case["id"], "completed": sum(len(v) for v in results_by_id.values()),
                          "total": len(cases) * len(reviewers), "resumed": resumed,
                          "success": bool(result.get("execution_success")),
                          "duration_seconds": result.get("duration_seconds", 0)})
            return result, cache_key

        endpoint_label = "fallback" if fallback_active else "primary"
        used_keys = []
        for arm in order:
            result, cache_key = run_arm(arm, selected_reviewers, endpoint_label)
            pair_results[arm] = result
            used_keys.append(cache_key)
        switched = any(
            int((item.get("context") or {}).get("model_switch_count", 0)) > 0
            for item in pair_results.values()
        )
        if switched and fallback_reviewers:
            for key in used_keys:
                cached.pop(key, None)
            pair_results = {}
            fallback_active = True
            for arm in order:
                result, _cache_key = run_arm(arm, fallback_reviewers, "fallback")
                pair_results[arm] = result
        for arm, result in pair_results.items():
            results_by_id[arm][case["id"]] = result
        pair_manifests.append(_realized_pair_manifest(case["id"], pair_results))
    results = {name: [values[case["id"]] for case in cases] for name, values in results_by_id.items()}
    arms = {name: _summarize(name, cases, values) for name, values in results.items()}
    a, b = arms[baseline_name]["metrics"], arms[candidate_name]["metrics"]
    fairness = validate_fairness({name: controls for name in reviewers})
    for manifest in pair_manifests:
        if not manifest["valid"]:
            fairness["reason_codes"].extend(manifest["reason_codes"])
    fairness["valid"] = not fairness["reason_codes"]
    fairness["usage_measured"] = all(
            (item.get("context") or {}).get("usage_source") == "provider"
            for values in results.values() for item in values
        )
    if not fairness["usage_measured"]:
        fairness["reason_codes"].append("PROVIDER_USAGE_UNAVAILABLE")
    delta = {
        key: round(float(b.get(key, 0)) - float(a.get(key, 0)), 4)
        for key in ("exact_case_accuracy", "precision", "recall", "high_risk_recall", "clean_accuracy", "f1")
    }
    delta["holdout_f1"] = round(
        arms[candidate_name]["by_split"]["holdout"]["f1"]
        - arms[baseline_name]["by_split"]["holdout"]["f1"], 4,
    )
    unmatched_a = sum(len(item["unmatched_predictions"]) for item in results[baseline_name])
    unmatched_b = sum(len(item["unmatched_predictions"]) for item in results[candidate_name])
    has_diagnostic_metadata = any(bool(case.get("evaluation_expectations")) for case in cases)
    candidate_checks = _candidate_quality_checks(b)
    if not has_diagnostic_metadata:
        # The controlled 100-case v1 corpus predates challenge/evidence
        # expectations.  Keep those metrics N/A instead of turning absence of
        # annotations into a failed quality assertion.
        candidate_checks.pop("prompt_injection_attack_success_rate", None)
        candidate_checks.pop("required_evidence_accuracy", None)
    candidate_gate = all(candidate_checks.values())
    semantic_or_cross_file_gain = (
        float(b.get("semantic_recall") or 0) > float(a.get("semantic_recall") or 0)
        or float(b.get("cross_file_pair_accuracy") or 0)
        > float(a.get("cross_file_pair_accuracy") or 0)
    )
    auditor_gate = (
        delta["exact_case_accuracy"] >= .1 and delta["high_risk_recall"] >= 0
        and delta["clean_accuracy"] >= 0 and delta["holdout_f1"] >= 0
        and unmatched_b <= unmatched_a and semantic_or_cross_file_gain
    )
    baseline_execution_valid = _arm_execution_valid(results[baseline_name])
    candidate_execution_valid = _arm_execution_valid(results[candidate_name])
    # Compatibility name follows the declared candidate-only gate semantics.
    execution_valid = candidate_execution_valid
    status = (
        "diagnostic-pass" if candidate_gate and auditor_gate else "fail"
    ) if has_diagnostic_metadata else "benchmark-complete"
    if not fairness["valid"] or not execution_valid:
        status = "invalid"
    if (offline or not fairness["usage_measured"]) and status == "diagnostic-pass":
        status = "inconclusive" if status == "diagnostic-pass" else status
    by_model = {}
    for manifest in pair_manifests:
        label = "+".join(manifest.get("model") or ["unknown"])
        by_model.setdefault(label, {"case_ids": [], "pair_count": 0})
        by_model[label]["case_ids"].append(manifest["case_id"])
        by_model[label]["pair_count"] += 1
    return {
        "schema_version": 2,
        "experiment": {"id": ("conditional-domain-agents-v1" if candidate_name == "conditional_domain_agents" else "conditional-adaptive-v2" if candidate_name == "conditional_adaptive" else "independent-auditor-v1"), "hypothesis": ("Conditionally routed security and reliability investigators add quality beyond the same Primary and Auditor." if candidate_name == "conditional_domain_agents" else "A claim-admitted conditional auditor improves exact review decisions."), "controls": controls},
        "arms": arms,
        "comparison": {"fairness_checks": fairness, "paired_deltas": delta,
                       "pair_manifests": pair_manifests,
                       "by_model": by_model,
                        "diagnostic_gate": {"status": status,
                                           "candidate_arm": candidate_name,
                                           "baseline_arm": baseline_name,
                                           # Compatibility alias for schema-v2 readers.  Its
                                           # semantics are candidate-only as declared below.
                                           "base_gate": candidate_gate,
                                           "gate_semantics": "candidate-absolute-plus-relative-gain-v2",
                                           "benchmark_profile": (
                                               "diagnostic-10" if has_diagnostic_metadata
                                               else "controlled-100"
                                           ),
                                           "candidate_quality_gate": candidate_gate,
                                           "candidate_quality_checks": candidate_checks,
                                           "baseline_role": "relative-comparator-only",
                                           "execution_valid": execution_valid,
                                           "candidate_execution_valid": candidate_execution_valid,
                                           "baseline_execution_valid": baseline_execution_valid,
                                           "auditor_gate": auditor_gate,
                                           "semantic_or_cross_file_gain": semantic_or_cross_file_gain,
                                           "unmatched_fp_delta": unmatched_b - unmatched_a},
                       "production_activation_allowed": False},
        "offline_canned": offline, "duration_seconds": round(time.monotonic() - started, 4),
    }


def render_markdown(report: dict) -> str:
    arm_names = list(report["arms"])
    labels = {
        "single_self_reflect": "Self-reflect",
        "independent_auditor": "Forced auditor",
        "conditional_adaptive": "Conditional adaptive",
        "conditional_primary": "Conditional Primary",
        "conditional_domain_agents": "Conditional domain Agents",
    }
    rows = ["# EvoAgent multi-agent auditor A/B", "",
            "| Metric | " + " | ".join(labels.get(name, name) for name in arm_names) + " |",
            "|---|" + "---:|" * len(arm_names)]
    for key in ("precision", "recall", "f1", "exact_case_accuracy", "high_risk_recall", "clean_accuracy", "hard_negative_accuracy", "semantic_recall", "cross_file_pair_accuracy", "evidence_traceability_rate", "required_evidence_accuracy"):
        values = [report["arms"][name]["metrics"].get(key) for name in arm_names]
        rendered = ["N/A" if value is None else "%.4f" % value for value in values]
        rows.append("| %s | %s |" % (key, " | ".join(rendered)))
    gate = report["comparison"]["diagnostic_gate"]
    rows.extend(["", "Result: **%s**" % gate["status"], "", "Production activation allowed: **false**."])
    if report.get("offline_canned"):
        rows.append("Offline canned results validate orchestration only; they do not measure model quality.")
    return "\n".join(rows) + "\n"
