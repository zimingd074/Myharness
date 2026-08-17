"""Multi-agent review with planning, dialogue, verification and arbitration.

The coordinator implements a bounded collaboration protocol:
plan -> specialist review -> peer challenge -> evidence revision -> independent
verification -> arbitration.  Every hand-off is persisted as an agent message
when a task store is available.  Failed specialists are retried and then
replanned to a substitute reviewer.
"""
import ast
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Optional, TypedDict

from .context_manager import ContextBundle, ContextManager
from .context.pr_map import PRContextMap, build_pr_context_map
from .context.retrieval import RepositoryRetrieval, RepositorySnapshotProvider
from .context.evidence import EvidenceLedger
from .context.shard_planner import ReviewShard, ShardPlanner
from .context.budget_policy import BudgetPolicy
from .context.coverage import CoverageReceipt
from .context.risk_priority import DiffRiskScanner
from .verifier_agent import VerifierAgentNode
from .auditor_agent import AuditStopPolicy, AuditorAgentNode
from .diff_parser import ParsedDiff
from .memory import MemoryManager
from .models import Finding, Severity
from .reviewer import LocalRuleReviewer, ModelRequestFailure, Reviewer
from .runtime import AgentLoop, AgentRuntime, AgentTool, RuntimeBudgetExceeded, RuntimeNode, ToolRegistry
from .context.reducers import reduce_tool_result
from .adaptive import (
    AdaptiveDecisionPolicy, AgentRun, AgentSpec, Challenge, Claim, ModelRequestAttempt,
    DeterministicRiskRouter, SharedAgentBudget, artifact_fingerprint,
    canonical_rule_id, evidence_from_record,
)


@dataclass
class AgentMessage:
    sender: str
    recipient: str
    kind: str
    content: Dict[str, Any]
    correlation_id: str = ""
    event_key: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class CollaborationBus:
    """Task-scoped mailbox plus durable transcript."""

    def __init__(self, task_id: str = "", store=None):
        self.task_id = task_id
        self.store = store
        self.messages: List[AgentMessage] = []
        self._lock = threading.Lock()

    def send(
        self, sender: str, recipient: str, kind: str,
        content: Dict[str, Any], correlation_id: str = "",
    ) -> AgentMessage:
        event_key = artifact_fingerprint({
            "protocol": "adaptive-review-v2", "task": self.task_id,
            "sender": sender, "recipient": recipient, "kind": kind,
            "correlation_id": correlation_id, "content": content,
        })
        message = AgentMessage(sender, recipient, kind, content, correlation_id, event_key)
        with self._lock:
            self.messages.append(message)
            if self.store is not None and self.task_id:
                self.store.record_agent_message(self.task_id, message.to_dict())
        return message

    def inbox(self, recipient: str, correlation_id: str = "") -> List[dict]:
        with self._lock:
            values = [
                message.to_dict() for message in self.messages
                if message.recipient in {recipient, "specialists", "all"}
                and (not correlation_id or message.correlation_id == correlation_id)
            ]
        return values

    def count(self, kind: str = "") -> int:
        with self._lock:
            return sum(1 for item in self.messages if not kind or item.kind == kind)


@dataclass
class ReviewAssignment:
    agent: str
    objective: str
    files: List[str]
    risk_domains: List[str]
    assignment_id: str = ""
    round: int = 1
    reason: str = "initial-plan"
    shard_id: str = ""
    shard_files: List[str] = field(default_factory=list)
    coverage_scope: str = "full-pr"
    budget: Dict[str, int] = field(default_factory=dict)
    required_hunk_ids: List[str] = field(default_factory=list)
    priority_hunks: List[Dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReviewPlan:
    languages: List[str]
    changed_files: List[str]
    risk_level: str
    assignments: List[ReviewAssignment]
    audit_rounds: int = BudgetPolicy.REVERSE_AUDIT_ROUNDS
    dry_round_limit: int = BudgetPolicy.REVERSE_AUDIT_DRY_ROUNDS

    def to_dict(self) -> dict:
        return {
            "languages": self.languages,
            "changed_files": self.changed_files,
            "risk_level": self.risk_level,
            "assignments": [item.to_dict() for item in self.assignments],
            "audit_rounds": self.audit_rounds,
            "dry_round_limit": self.dry_round_limit,
        }


class CrossShardTracerAgent:
    """A graph node contract, backed by one short existing AgentLoop run."""

    name = "cross-shard-tracer"

    @staticmethod
    def objective() -> str:
        return ("Trace cross-file caller/callee, API, schema, and producer/consumer "
                "contracts for changed symbols. For a removed guard, locate a possible "
                "replacement only; leave security-policy judgment to the Security agent. "
                "Use repository tools; do not re-read specialist conversations.")


@dataclass
class Critique:
    finding_key: str
    accepted: bool
    objections: List[str]
    confidence_adjustment: float
    questions: List[str]
    requires_revision: bool
    round: int = 1


@dataclass
class Reflection:
    finding_key: str
    revision_needed: bool
    guidance: List[str]
    round: int


@dataclass
class Reproduction:
    finding_key: str
    reproducible: bool
    method: str
    evidence: str


@dataclass
class VerificationDecision:
    finding_key: str
    approved: bool
    reasons: List[str]
    confidence: float


class CollaborationState(TypedDict, total=False):
    diff: str
    parsed: ParsedDiff
    task_id: str
    repository: str
    tenant_id: str
    bus: CollaborationBus
    plan: ReviewPlan
    specialist_findings: List[Finding]
    finding_sources: Dict[str, List[str]]
    assignments_by_agent: Dict[str, ReviewAssignment]
    critiques: Dict[str, Critique]
    reproductions: Dict[str, Reproduction]
    fix_ready: Dict[str, bool]
    decisions: Dict[str, VerificationDecision]
    verified: List[Finding]
    agent_outcomes: List[dict]
    rounds_completed: int
    pr_map: PRContextMap
    shards: List[ReviewShard]
    coverage_gaps: List[dict]
    snapshot_provider: RepositorySnapshotProvider
    evidence_ledger: EvidenceLedger
    risk_scores: Dict[str, int]
    claims: List[Claim]
    challenges: List[Challenge]
    adaptive_decisions: List[Any]
    escalations: List[dict]
    agent_runs: List[AgentRun]
    shared_budget: SharedAgentBudget
    coverage_receipts: List[dict]
    budget_gaps: List[dict]


def finding_key(finding: Finding) -> str:
    raw = "%s:%s:%s" % (finding.path, finding.line, finding.rule_id)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class FilteredAgent(Reviewer):
    """Compatibility adapter for third-party reviewers; built-ins do not use it."""

    def __init__(self, name: str, reviewer: Reviewer, prefixes: tuple):
        self.name = name
        self.reviewer = reviewer
        self.prefixes = prefixes
        self.domains = tuple(
            "security" if item.startswith("SEC") else "reliability"
            for item in prefixes
        )

    def review(self, diff: str, parsed: ParsedDiff) -> List[Finding]:
        return [
            item for item in self.reviewer.review(diff, parsed)
            if item.rule_id.startswith(self.prefixes)
        ]


class PlannerAgent:
    name = "planner-agent"

    DOMAIN_OBJECTIVES = {
        "security": "Trace attacker-controlled data and find exploitable security defects.",
        "reliability": "Find failure handling, observability and runtime reliability regressions.",
        "correctness": "Find behavior and data-flow defects introduced by the change.",
        "regression": "Identify compatibility and test gaps caused by the change.",
    }

    def plan(self, parsed: ParsedDiff, specialists: List[Reviewer]) -> ReviewPlan:
        extensions = {path.rsplit(".", 1)[-1].lower() for path in parsed.files if "." in path}
        languages = sorted({
            "python" if ext == "py" else
            "javascript" if ext in {"js", "jsx", "ts", "tsx"} else
            "configuration" if ext in {"yml", "yaml", "json", "toml"} else ext
            for ext in extensions
        })
        sensitive = any(
            token in path.lower()
            for path in parsed.files
            for token in ("auth", "security", "payment", "permission", "token", "migration")
        )
        default_domains = ["security", "reliability", "correctness", "regression"]
        assignments = []
        for index, agent in enumerate(specialists, 1):
            declared = list(getattr(agent, "domains", ()) or default_domains)
            objectives = [
                self.DOMAIN_OBJECTIVES[item]
                for item in declared if item in self.DOMAIN_OBJECTIVES
            ]
            assignments.append(ReviewAssignment(
                agent=agent.name,
                objective=" ".join(objectives) or "Find actionable defects and cite changed-line evidence.",
                files=list(parsed.files),
                risk_domains=declared,
                assignment_id="A%02d" % index,
            ))
        return ReviewPlan(
            languages=languages or ["unknown"],
            changed_files=list(parsed.files),
            risk_level="high" if sensitive or len(parsed.files) > 10 else "normal",
            assignments=assignments,
        )

    def replan(
        self, failed: ReviewAssignment, substitutes: List[Reviewer], error: str,
    ) -> Optional[ReviewAssignment]:
        if not substitutes:
            return None
        target = max(
            substitutes,
            key=lambda item: len(
                set(getattr(item, "domains", ()) or failed.risk_domains)
                .intersection(failed.risk_domains)
            ),
        )
        return ReviewAssignment(
            agent=target.name,
            objective=(
                failed.objective
                + " Take over a failed assignment and independently reconstruct its evidence."
            ),
            files=list(failed.files), risk_domains=list(failed.risk_domains),
            assignment_id=failed.assignment_id, round=failed.round + 1,
            reason="replacement-after-failure: %s" % error[:160],
            shard_id=failed.shard_id, shard_files=list(failed.shard_files),
            coverage_scope=failed.coverage_scope,
        )


class CriticAgent:
    name = "critic-agent"

    def challenge(
        self, finding: Finding, parsed: ParsedDiff,
        peer_sources: Optional[List[str]] = None, round_number: int = 1,
    ) -> Critique:
        objections = []
        questions = []
        valid_locations = {(line.path, line.line) for line in parsed.added_lines}
        if (finding.path, finding.line) not in valid_locations:
            objections.append("location is not an added line")
        source_line = next(
            (line.content for line in parsed.added_lines
             if line.path == finding.path and line.line == finding.line), ""
        )
        if not finding.evidence or finding.evidence.strip() not in source_line.strip():
            objections.append("quoted evidence does not match the changed line")
        if len(finding.explanation.strip()) < 12:
            objections.append("explanation is not specific enough")
        if len(finding.fix.strip()) < 8:
            objections.append("remediation is not actionable")
        if len(finding.test.strip()) < 8:
            objections.append("test strategy is not actionable")
        if len(peer_sources or []) < 2 and finding.severity in {Severity.CRITICAL, Severity.HIGH}:
            questions.append("High-impact claim needs independent verifier evidence.")
        adjustment = -.35 if objections else (.08 if len(peer_sources or []) > 1 else .03)
        return Critique(
            finding_key(finding), not objections, objections, adjustment,
            questions, bool(objections), round_number,
        )


class EvidenceAgent:
    name = "evidence-agent"

    def reproduce(self, finding: Finding, parsed: ParsedDiff) -> Reproduction:
        line = next(
            (item.content for item in parsed.added_lines
             if item.path == finding.path and item.line == finding.line), ""
        )
        normalized = line.replace(" ", "")
        signatures = {
            "SEC-EVAL": ("eval(" in line or "exec(" in line),
            "SEC-SUBPROCESS-SHELL": "shell=True" in normalized,
            "SEC-HARDCODED-SECRET": any(
                token in line.lower() for token in ("password", "secret", "token", "api_key")
            ),
            "SEC-SQL-CONCAT": any(token in line for token in ("execute(", "query(")),
            "REL-DEBUG-PRINT": "print(" in line or "console.log(" in line,
            "REL-EMPTY-EXCEPT": "except" in line,
        }
        exact_evidence = bool(line and finding.evidence and finding.evidence.strip() in line.strip())
        reproducible = signatures.get(finding.rule_id, exact_evidence)
        return Reproduction(
            finding_key(finding), reproducible,
            "independent changed-line evidence check",
            line.strip()[:240] if reproducible else "No independently matching changed-line evidence.",
        )


class ReflectionAgent:
    name = "reflection-agent"

    def reflect(self, critique: Critique) -> Reflection:
        guidance = list(critique.objections)
        guidance.extend(critique.questions)
        if critique.requires_revision:
            guidance.append(
                "Re-read the changed line, discard unsupported assumptions, and return a corrected claim."
            )
        return Reflection(
            critique.finding_key, critique.requires_revision, guidance, critique.round
        )


class TestAgent(EvidenceAgent):
    """Backward-compatible name for integrations importing TestAgent."""

    name = "test-agent"


class FixAgent:
    name = "fix-agent"

    def assess(self, finding: Finding) -> bool:
        dangerous = ("disable validation", "ignore error", "catch all")
        text = finding.fix.lower()
        return bool(finding.fix and not any(item in text for item in dangerous))


class VerifierAgent:
    name = "verifier-agent"

    def verify(
        self, finding: Finding, critique: Critique,
        reproduction: Reproduction, fix_ready: bool, evidence_valid: bool = True,
    ) -> VerificationDecision:
        reasons = []
        confidence = max(0.0, min(1.0, finding.confidence + critique.confidence_adjustment))
        if not critique.accepted:
            reasons.extend(critique.objections)
        if not reproduction.reproducible:
            reasons.append("independent evidence could not reproduce the claim")
        if not evidence_valid:
            reasons.append("finding evidence is not traceable to a pinned locator")
        if not fix_ready:
            reasons.append("proposed remediation failed the safety/actionability gate")
        if confidence < .55:
            reasons.append("confidence is below the verification threshold")
        approved = not reasons
        return VerificationDecision(finding_key(finding), approved, reasons, confidence)


class ArbiterAgent:
    name = "arbiter-agent"

    def decide(
        self, findings: List[Finding], decisions: Dict[str, VerificationDecision],
    ) -> List[Finding]:
        merged: Dict[tuple, Finding] = {}
        for finding in findings:
            decision = decisions[finding_key(finding)]
            if not decision.approved:
                continue
            finding.confidence = decision.confidence
            identity = (finding.path, finding.line, finding.rule_id)
            current = merged.get(identity)
            if current is None or finding.confidence > current.confidence:
                merged[identity] = finding
        order = {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2, Severity.LOW: 3}
        return sorted(merged.values(), key=lambda item: (order[item.severity], item.path, item.line))


class SynthesizerAgent(ArbiterAgent):
    """Compatibility alias; arbitration is now an explicit final role."""

    name = "synthesizer-agent"


# Semantic names for deterministic workflow stages. Legacy names remain
# import-compatible because they appear in existing transcripts and reports.
AssignmentPlanner = PlannerAgent
FindingCritiquePolicy = CriticAgent
RevisionGuidancePolicy = ReflectionAgent
ChangedLineEvidenceValidator = EvidenceAgent
RemediationSafetyPolicy = FixAgent
FindingAcceptanceGate = VerifierAgent
FindingDecisionReducer = ArbiterAgent


class MultiAgentCoordinator(Reviewer):
    """Bounded collaborative review with dialogue and failure recovery."""

    name = "multi-agent-collaboration"

    def __init__(
        self, agents: List[Reviewer], max_workers: int = 4, store=None,
        agent_retries: int = 1, collaboration_rounds: int = 2,
        fallback_agent: Optional[Reviewer] = None,
        context_manager: Optional[ContextManager] = None,
        memory_manager: Optional[MemoryManager] = None,
        agent_loop_max_steps: int = 6, agent_loop_timeout_seconds: int = 120,
        agent_runtime_max_steps: int = 12, agent_runtime_timeout_seconds: int = 300,
        context_active_rounds: int = 3, context_soft_compact_ratio: float = .60,
        context_hard_compact_ratio: float = .80,
        context_architecture: str = "coverage-first",
        specialist_activation: str = "hybrid",
        shard_file_threshold: int = 12, shard_changed_line_threshold: int = 1200,
        snapshot_provider: Optional[RepositorySnapshotProvider] = None,
        snapshot_factory=None,
        review_mode: str = "legacy", challenge_strategy: str = "independent_auditor",
        agent_budget: Optional[Dict[str, int]] = None,
        review_pipeline: str = "classic",
    ):
        self.agents = agents
        self.max_workers = max_workers
        self.store = store
        self.agent_retries = max(0, agent_retries)
        self.collaboration_rounds = max(1, collaboration_rounds)
        self.fallback_agent = fallback_agent or LocalRuleReviewer()
        self.context_manager = context_manager or ContextManager()
        self.memory_manager = memory_manager
        self.agent_loop = AgentLoop(
            agent_loop_max_steps, agent_loop_timeout_seconds,
            active_rounds=context_active_rounds,
            soft_compact_ratio=context_soft_compact_ratio,
            hard_compact_ratio=context_hard_compact_ratio,
            context_budget=self.context_manager.budget,
        )
        if context_architecture not in {"legacy", "coverage-first"}:
            raise ValueError("context_architecture must be legacy or coverage-first")
        self.context_architecture = context_architecture
        if specialist_activation not in {"directed", "hybrid", "all"}:
            raise ValueError("specialist_activation must be directed, hybrid or all")
        self.specialist_activation = specialist_activation
        self.shard_planner = ShardPlanner(
            self.context_manager.max_tokens - self.context_manager.reserved_tokens,
            shard_file_threshold, shard_changed_line_threshold,
            self.context_manager.token_counter,
        )
        self.snapshot_provider = snapshot_provider
        self.snapshot_factory = snapshot_factory
        if review_mode not in {"legacy", "rules_only", "single_agent", "adaptive_multi_agent"}:
            raise ValueError("invalid review_mode: %s" % review_mode)
        if challenge_strategy not in {"self_reflect", "independent_auditor"}:
            raise ValueError("invalid challenge_strategy: %s" % challenge_strategy)
        self.review_mode = review_mode
        self.challenge_strategy = challenge_strategy
        self.agent_budget = dict(agent_budget or {})
        if review_pipeline not in {"classic", "bounded-v2"}:
            raise ValueError("review_pipeline must be classic or bounded-v2")
        self.review_pipeline = review_pipeline
        self.budget_policy = BudgetPolicy()
        self.cross_shard_tracer = CrossShardTracerAgent()
        self.verifier_stage = VerifierAgentNode(self.budget_policy.VERIFIER_BATCH_SIZE)
        self.auditor_stage = AuditorAgentNode(AuditStopPolicy(
            self.budget_policy.REVERSE_AUDIT_ROUNDS,
            self.budget_policy.REVERSE_AUDIT_DRY_ROUNDS,
        ))
        self.risk_router = DeterministicRiskRouter()
        self.decision_policy = AdaptiveDecisionPolicy()
        # The bounded graph has ten deterministic orchestration nodes. Its
        # runtime limit governs graph progress, not model/tool spend, so never
        # let an old eight-step deployment abort before final adjudication.
        runtime_steps = max(10, agent_runtime_max_steps) if self.review_pipeline == "bounded-v2" else agent_runtime_max_steps
        self.runtime = AgentRuntime(
            max_steps=runtime_steps, timeout_seconds=agent_runtime_timeout_seconds,
        )
        self.planner = PlannerAgent()
        self.critic = CriticAgent()
        self.reflection_agent = ReflectionAgent()
        self.evidence_agent = EvidenceAgent()
        self.test_agent = self.evidence_agent
        self.fix_agent = FixAgent()
        self.verifier = VerifierAgent()
        self.arbiter = ArbiterAgent()
        self.synthesizer = self.arbiter
        self._summaries: Dict[str, dict] = {}
        self._last_summary: Dict[str, Any] = {}
        self._summary_lock = threading.Lock()

    def review(self, diff: str, parsed: ParsedDiff) -> List[Finding]:
        return self.review_with_context("", diff, parsed)

    def review_with_context(
        self, task_id: str, diff: str, parsed: ParsedDiff,
        repository: str = "", tenant_id: str = "default", pull_request: int = None,
        source_sha: str = "", execution_context: Optional[dict] = None,
    ) -> List[Finding]:
        state: CollaborationState = {
            "task_id": task_id, "diff": diff, "parsed": parsed,
            "repository": repository, "tenant_id": tenant_id,
            "bus": CollaborationBus(task_id, self.store),
            "pr_map": build_pr_context_map(diff, parsed), "coverage_gaps": [],
            "evidence_ledger": EvidenceLedger(), "llm_failures": [],
            # Direct diff submissions have no Git ref.  A deterministic diff
            # digest is an explicit version surrogate, never a claim about a
            # repository commit; integrations can pass the actual head SHA.
            "pull_request": pull_request,
            "source_sha": source_sha or ("diff:" + hashlib.sha256(diff.encode("utf-8")).hexdigest()),
            "claims": [], "challenges": [], "adaptive_decisions": [],
            "escalations": [], "agent_runs": [],
            "shared_budget": SharedAgentBudget(
                **self.agent_budget,
                tail_reserve=(self.budget_policy.tail_reserve({
                    "agent_runs": self.agent_budget.get("max_agent_runs", 4),
                    "llm_calls": self.agent_budget.get("max_llm_calls", 12),
                    "tool_calls": self.agent_budget.get("max_tool_calls", 24),
                }) if self.review_pipeline == "bounded-v2" else {}),
            ),
            "coverage_receipts": [], "budget_gaps": [],
            "execution_context": dict(execution_context or {}),
            # Keep the coordinator-level pinned snapshot in the task-local
            # state as well.  Claim scanners and checkpoint fingerprints read
            # task state, while agent tools retain the coordinator fallback.
            "snapshot_provider": self.snapshot_provider,
        }
        timeout = float(state["execution_context"].get("wall_timeout_seconds", 0) or 0)
        if timeout > 0:
            state["deadline_monotonic"] = time.monotonic() + timeout
        if self.snapshot_factory:
            state["snapshot_provider"] = self.snapshot_factory(
                repository, pull_request, state["pr_map"], task_id
            )
            if not source_sha:
                head = str(getattr(state["snapshot_provider"], "ref", ""))
                if head:
                    state["source_sha"] = head
        nodes = [RuntimeNode("planner", self._plan_node, checkpoint=False)]
        if self.review_pipeline == "bounded-v2" and self.review_mode == "adaptive_multi_agent":
            nodes.extend([
                RuntimeNode("investigation", self._investigation_node, checkpoint=False),
                RuntimeNode("coverage_repair", self._coverage_repair_node, checkpoint=False),
            ])
        else:
            nodes.extend([
                RuntimeNode("specialists", self._specialist_node, checkpoint=False),
                RuntimeNode("coverage_repair", self._coverage_repair_node, checkpoint=False),
                RuntimeNode("cross_shard", self._cross_shard_node, checkpoint=False),
            ])
        if self.review_mode == "legacy":
            nodes.extend([
                RuntimeNode("deliberation", self._deliberation_node, checkpoint=False),
                RuntimeNode("evidence", self._evidence_node, checkpoint=False),
                RuntimeNode("verifier", self._verify_node, checkpoint=False),
                RuntimeNode("arbiter", self._arbitrate_node, checkpoint=False),
            ])
        else:
            if self.review_pipeline == "bounded-v2":
                nodes.extend([
                    RuntimeNode("claims", self._claim_node, checkpoint=False),
                    RuntimeNode("verifier", self._challenge_node, checkpoint=False),
                    RuntimeNode("auditor", self._auditor_node, checkpoint=False),
                    RuntimeNode("reclaim", self._claim_node, checkpoint=False),
                    RuntimeNode("reverify", self._challenge_node, checkpoint=False),
                    RuntimeNode("decision", self._adaptive_decision_node, checkpoint=False),
                ])
            else:
                nodes.extend([
                    RuntimeNode("claims", self._claim_node, checkpoint=False),
                    RuntimeNode("challenge", self._challenge_node, checkpoint=False),
                    RuntimeNode("decision", self._adaptive_decision_node, checkpoint=False),
                ])
        result = self.runtime.execute(
            state,
            nodes,
            task_id=task_id,
        )
        summary = self._make_summary(result)
        self._last_summary = summary
        if task_id:
            with self._summary_lock:
                self._summaries[task_id] = summary
        return result["verified"]

    def review_with_execution_context(
        self, task_id: str, diff: str, parsed: ParsedDiff, execution_context: dict,
        repository: str = "", tenant_id: str = "default", pull_request: int = None,
        source_sha: str = "",
    ) -> List[Finding]:
        # The additive entry point lets the outer durable runtime supply claim
        # tokens and fingerprints without breaking third-party Reviewers.
        return self.review_with_context(
            task_id, diff, parsed, repository, tenant_id, pull_request, source_sha,
            execution_context,
        )

    def collaboration_summary(self, task_id: str) -> dict:
        with self._summary_lock:
            return dict(self._summaries.get(task_id, {}))

    def last_collaboration_summary(self) -> dict:
        return dict(self._last_summary)

    @staticmethod
    def _bus(state: CollaborationState) -> CollaborationBus:
        return state["bus"]

    def _emit(
        self, state: CollaborationState, sender: str, recipient: str,
        kind: str, content: Dict[str, Any], correlation_id: str = "",
    ) -> None:
        self._bus(state).send(sender, recipient, kind, content, correlation_id)

    def _plan_node(self, state: CollaborationState) -> Dict[str, Any]:
        if self.memory_manager and state.get("repository"):
            self.memory_manager.mark_stale_memories(
                state.get("tenant_id", "default"), state["repository"],
                state["parsed"].files,
                [symbol for item in state["pr_map"].files for symbol in item.symbols],
                state.get("source_sha", ""),
            )
        if self.review_mode != "legacy":
            return self._adaptive_plan_node(state)
        plan = self.planner.plan(state["parsed"], self.agents)
        shards = self._review_shards(state)
        assignments = []
        if len(shards) == 1:
            # Preserve the legacy/small-PR fanout and behaviour.
            for assignment in plan.assignments:
                assignments.append(ReviewAssignment(
                    agent=assignment.agent, objective=assignment.objective,
                    files=list(shards[0].files), risk_domains=list(assignment.risk_domains),
                    assignment_id=assignment.assignment_id, round=assignment.round,
                    reason=assignment.reason, shard_id=shards[0].shard_id,
                    shard_files=list(shards[0].files), coverage_scope="full-pr",
                ))
        else:
            # The selected strategy always names an accountable owner. Hybrid
            # supplies a generalist for every shard and adds domain reviewers.
            file_map = {item.path: item for item in state["pr_map"].files}
            by_name = {item.name: item for item in self.agents}
            loop_assignments = [item for item in plan.assignments
                                if callable(getattr(by_name.get(item.agent), "agent_step", None))]
            for index, shard in enumerate(shards):
                # A configured loop-capable reviewer is the generalist even
                # when its declared domain is narrow: it is the explicit
                # coverage owner for unknown-domain shards.  Risk specialists
                # remain supplemental below.
                generalist = loop_assignments[0] if loop_assignments else None
                fallback = next((item for item in plan.assignments if item.agent == self.fallback_agent.name), None)
                # The fallback normally is not part of Planner assignments.
                # Give each shard a deterministic owner when no LLM is enabled.
                if fallback is None and generalist is None:
                    fallback = ReviewAssignment(
                        agent=self.fallback_agent.name,
                        objective="Perform a general correctness review.", files=[],
                        risk_domains=["security", "reliability", "correctness"],
                        assignment_id="fallback-generalist", reason="fallback",
                    )
                directed_owner = plan.assignments[index % len(plan.assignments)]
                primary = (directed_owner if self.specialist_activation == "directed"
                           else generalist or fallback or directed_owner)
                selected = (list(plan.assignments) if self.specialist_activation == "all"
                            else [primary])
                tags = {tag for path in shard.files for tag in file_map.get(path, PRContextMap([], 0, 0)).risk_tags}
                wanted = {"security" if tag in {
                    "auth", "permission", "token", "secret", "password", "payment", "sql",
                    "eval", "exec", "shell", "subprocess", "pickle", "yaml", "deserialize",
                } else "reliability" if tag in {"except", "print"} else tag for tag in tags}
                if self.specialist_activation in {"directed", "hybrid"}:
                    extra = next((item for item in plan.assignments if item.agent != primary.agent
                                  and wanted.intersection(item.risk_domains)), None)
                    if extra:
                        selected.append(extra)
                # Hybrid/all include the generalist and therefore cover unknown
                # domains. Directed keeps only its round-robin owner plus a
                # risk-specific supplemental reviewer.
                for assignment in selected:
                    assignments.append(ReviewAssignment(
                        agent=assignment.agent, objective=assignment.objective,
                        files=list(shard.files), risk_domains=list(assignment.risk_domains),
                        assignment_id="%s-%s" % (assignment.assignment_id, shard.shard_id),
                        round=assignment.round, reason=("coverage-owner" if assignment is primary
                                                         else "risk-domain-second-opinion"),
                        shard_id=shard.shard_id, shard_files=list(shard.files), coverage_scope="shard",
                    ))
        plan.assignments = assignments
        shard_by_id = {item.shard_id: item for item in shards}
        for assignment in assignments:
            self._emit(
                state, self.planner.name, assignment.agent, "assignment",
                assignment.to_dict(), assignment.assignment_id,
            )
        return {
            "plan": plan,
            "assignments_by_agent": {item.agent: item for item in plan.assignments},
            "shards": shards, "shards_by_id": shard_by_id,
        }

    def _adaptive_plan_node(self, state: CollaborationState) -> Dict[str, Any]:
        scanners = [item for item in self.agents
                    if getattr(item, "execution_kind", "deterministic-checker") != "agent"]
        active_agents = [item for item in self.agents
                         if getattr(item, "execution_kind", "") == "agent"
                         # Verifiers only falsify supplied claims; auditors
                         # only look for omissions after verification. Neither
                         # belongs in the initial specialist fan-out.
                         and getattr(item, "agent_role", "primary") not in {"auditor", "verifier"}]
        preview = []
        for scanner in scanners:
            try:
                preview.extend(
                    finding for finding in scanner.review(state["diff"], state["parsed"])
                    if self.decision_policy.structurally_valid(
                        finding, state["parsed"], "deterministic-checker"
                    )
                )
            except Exception:
                pass
        preview.extend(
            finding for finding, _run_id, _evidence
            in self._cross_file_signature_candidates(state)
        )
        scores = self.risk_router.score(state["diff"], state["parsed"], preview)
        activated = set(self.risk_router.activated(scores))
        if self.review_mode == "rules_only":
            active_agents = []
        elif self.review_mode == "single_agent":
            active_agents = [item for item in active_agents
                             if getattr(item, "agent_role", "primary") == "primary"][:1]
        elif self.review_pipeline == "classic":
            active_agents = [item for item in active_agents if (
                getattr(item, "agent_role", "primary") == "primary"
                or getattr(item, "agent_role", "") in activated
            )]
        else:
            # Coverage is role-based, not risk-score-based. Risk priorities
            # only order exploration inside a shard.
            required_roles = {"primary", "security", "reliability"}
            active_agents = [item for item in active_agents
                             if getattr(item, "agent_role", "") in required_roles]
            activated.update(getattr(item, "agent_role", "") for item in active_agents)
        shards = self._review_shards(state)
        selected = scanners + active_agents
        assignments = []
        for shard in shards:
            for index, agent in enumerate(selected):
                role = getattr(agent, "agent_role", "scanner")
                assignments.append(ReviewAssignment(
                    agent=agent.name,
                    objective=(
                        "Independently investigate %s risks and produce evidence-backed claims." % role
                        if getattr(agent, "execution_kind", "") == "agent"
                        else "Run the deterministic %s rule scanner." % agent.name
                    ),
                    files=list(shard.files), risk_domains=list(getattr(agent, "domains", ())),
                    assignment_id="%s-%s" % (
                        hashlib.sha256(agent.name.encode("utf-8")).hexdigest()[:10], shard.shard_id,
                    ),
                    reason="coverage-owner" if index == 0 else "risk-domain-second-opinion",
                    shard_id=shard.shard_id, shard_files=list(shard.files),
                    coverage_scope="full-pr" if len(shards) == 1 else "shard",
                    budget=self.budget_policy.for_shard(shard.changed_lines).to_dict(),
                    required_hunk_ids=[item.hunk_id for item in shard.risk_hunks],
                    priority_hunks=[item.to_dict() for item in sorted(
                        shard.risk_hunks, key=lambda item: (-item.score, item.path, item.start_line)
                    )],
                ))
        planned_metadata = self.planner.plan(state["parsed"], selected)
        plan = ReviewPlan(
            languages=planned_metadata.languages,
            changed_files=list(state["parsed"].files),
            risk_level=planned_metadata.risk_level, assignments=assignments,
            audit_rounds=self.budget_policy.REVERSE_AUDIT_ROUNDS,
            dry_round_limit=self.budget_policy.REVERSE_AUDIT_DRY_ROUNDS,
        )
        for assignment in assignments:
            self._emit(state, self.planner.name, assignment.agent, "assignment",
                       assignment.to_dict(), assignment.assignment_id)
        missing_roles = ({"primary", "security", "reliability"} - {
            getattr(item, "agent_role", "") for item in active_agents
        }) if (self.review_mode == "adaptive_multi_agent"
               and self.review_pipeline == "bounded-v2") else set()
        if missing_roles:
            state.setdefault("coverage_gaps", []).append({
                "reason": "required specialists unavailable", "files": list(state["parsed"].files),
                "roles": sorted(missing_roles),
            })
        return {
            "plan": plan, "assignments_by_agent": {item.agent: item for item in assignments},
            "shards": shards, "shards_by_id": {item.shard_id: item for item in shards},
            "risk_scores": scores, "activated_domains": sorted(activated),
        }

    def _review_shards(self, state: CollaborationState) -> List[ReviewShard]:
        if self.context_architecture == "legacy":
            return [self.shard_planner._make_shard("full", list(state["parsed"].files), state["diff"])]
        return self.shard_planner.plan(state["diff"], state["parsed"], state["pr_map"])

    def _recall_memories(
        self, state: CollaborationState, assignment: ReviewAssignment,
    ) -> List[dict]:
        if not self.memory_manager or not state.get("repository"):
            return []
        file_map = {item.path: item for item in state["pr_map"].files}
        symbols = [symbol for path in assignment.files for symbol in file_map.get(path, PRContextMap([], 0, 0)).symbols]
        memories = self.memory_manager.recall_for_assignment(
            state.get("tenant_id", "default"), state.get("repository", ""),
            state.get("task_id", ""), assignment.agent, assignment.shard_id or "full",
            assignment.files, symbols, assignment.risk_domains, assignment.objective,
            state.get("source_sha", ""),
        )
        if memories:
            self._emit(
                state, "memory-manager", assignment.agent, "memory_recalled",
                {
                    "count": len(memories),
                    "memory_ids": [item["id"] for item in memories],
                    "scopes": sorted({item["scope"] for item in memories}),
                }, assignment.assignment_id,
            )
        return memories

    def _agent_tools(
        self, state: CollaborationState, assignment: ReviewAssignment,
        cross_shard: bool = False, local_diff_override: str = "",
    ) -> ToolRegistry:
        shard = (state.get("shards_by_id") or {}).get(assignment.shard_id)
        local_diff = local_diff_override or (state["diff"] if cross_shard else (shard.diff if shard else state["diff"]))
        local_parsed = state["parsed"] if cross_shard else (shard.parsed if shard else state["parsed"])
        retrieval = RepositoryRetrieval(state.get("snapshot_provider") or self.snapshot_provider, local_diff)
        def search_diff(query: str, limit: int = 20):
            value = str(query).strip().lower()
            if not value:
                raise ValueError("search_diff query is required")
            hits = []
            for index, line in enumerate(local_diff.splitlines(), 1):
                if value in line.lower():
                    hits.append({"diff_line": index, "content": line[:500]})
                if len(hits) >= max(1, min(int(limit), 50)):
                    break
            return hits

        def changed_line(path: str, line: int):
            match = next((
                item for item in local_parsed.added_lines
                if item.path == str(path) and item.line == int(line)
            ), None)
            if match is None:
                return {"found": False, "path": path, "line": line}
            return {
                "found": True, "path": match.path, "line": match.line,
                "content": match.content,
            }

        def list_changed_files():
            return list(local_parsed.files)

        def recall_memory(query: str, limit: int = 5):
            if not self.memory_manager or not state.get("repository"):
                return []
            return self.memory_manager.recall_for_assignment(
                state.get("tenant_id", "default"), state["repository"], state.get("task_id", ""),
                assignment.agent, assignment.shard_id or "full", assignment.files,
                [symbol for item in state["pr_map"].files if item.path in assignment.files for symbol in item.symbols],
                assignment.risk_domains, "%s %s" % (assignment.objective, str(query)),
                state.get("source_sha", ""), max(1, min(int(limit), 10)),
            )

        def read_file(path: str, start_line: int, end_line: int):
            return retrieval.read_file(path, start_line, end_line)

        def read_diff(path: str = "", cursor: int = 0, limit: int = 120):
            if cross_shard and not path:
                raise ValueError("cross-shard read_diff requires a path")
            return retrieval.read_diff(path, cursor, limit)

        def grep_repo(query: str, path: str = "", cursor: int = 0, limit: int = 20):
            return retrieval.grep_repo(query, path, cursor, limit)

        read_diff_schema = {
            "type": "object", "properties": {
                "path": {"type": "string"}, "cursor": {"type": "integer", "minimum": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 400},
            }, "additionalProperties": False,
        }
        if cross_shard:
            read_diff_schema["required"] = ["path"]
        tool_values = [
            AgentTool(
                "search_diff",
                "Search the PR diff for an exact case-insensitive text fragment.",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                    },
                    "required": ["query"], "additionalProperties": False,
                },
                search_diff, lambda value, limit: reduce_tool_result("search_diff", value, limit),
            ),
            AgentTool(
                "changed_line",
                "Read one added line by new-file path and line number.",
                {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "line": {"type": "integer", "minimum": 1},
                    },
                    "required": ["path", "line"], "additionalProperties": False,
                },
                changed_line, lambda value, limit: reduce_tool_result("changed_line", value, limit),
            ),
            AgentTool(
                "list_changed_files",
                "List files changed by this PR.",
                {
                    "type": "object", "properties": {},
                    "additionalProperties": False,
                },
                list_changed_files, lambda value, limit: reduce_tool_result("list_changed_files", value, limit),
            ),
            AgentTool(
                "recall_memory",
                "Recall repository-scoped review experience relevant to a query.",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                    },
                    "required": ["query"], "additionalProperties": False,
                },
                recall_memory, lambda value, limit: reduce_tool_result("recall_memory", value, limit),
            ),
            AgentTool("read_file", "Read a bounded line range from the authorized repository snapshot.",
                {"type": "object", "properties": {"path": {"type": "string"}, "start_line": {"type": "integer", "minimum": 1}, "end_line": {"type": "integer", "minimum": 1}}, "required": ["path", "start_line", "end_line"], "additionalProperties": False}, read_file, lambda value, limit: reduce_tool_result("read_file", value, limit)),
            AgentTool("read_diff", "Read a bounded page of the PR diff, optionally for one changed path.",
                read_diff_schema, read_diff, lambda value, limit: reduce_tool_result("read_diff", value, limit)),
            AgentTool("grep_repo", "Search authorized snapshot files with pagination and Top-K results.",
                {"type": "object", "properties": {"query": {"type": "string"}, "path": {"type": "string"}, "cursor": {"type": "integer", "minimum": 0}, "limit": {"type": "integer", "minimum": 1, "maximum": 50}}, "required": ["query"], "additionalProperties": False}, grep_repo, lambda value, limit: reduce_tool_result("grep_repo", value, limit)),
            AgentTool("find_symbol", "Find a symbol in the authorized repository snapshot.",
                {"type": "object", "properties": {"symbol": {"type": "string"}, "cursor": {"type": "integer", "minimum": 0}, "limit": {"type": "integer", "minimum": 1, "maximum": 50}}, "required": ["symbol"], "additionalProperties": False}, lambda symbol, cursor=0, limit=20: retrieval.find_symbol(symbol, cursor, limit), lambda value, limit: reduce_tool_result("find_symbol", value, limit)),
            AgentTool("find_references", "Find references to a symbol in the authorized repository snapshot.",
                {"type": "object", "properties": {"symbol": {"type": "string"}, "cursor": {"type": "integer", "minimum": 0}, "limit": {"type": "integer", "minimum": 1, "maximum": 50}}, "required": ["symbol"], "additionalProperties": False}, lambda symbol, cursor=0, limit=20: retrieval.find_references(symbol, cursor, limit), lambda value, limit: reduce_tool_result("find_references", value, limit)),
        ]
        if assignment.reason in {"blind-challenge", "verify-findings"}:
            # The normalized claims already contain changed-line locators.
            # A verifier must gather independent snapshot evidence rather
            # than spending its stage budget rereading the diff.
            locator_only = {"recall_memory", "search_diff", "changed_line", "read_diff"}
            tool_values = [item for item in tool_values if item.name not in locator_only]
        registry = ToolRegistry(tool_values)
        registry.retrieval = retrieval
        return registry

    def _run_agent_loop(
        self, state: CollaborationState, agent: Reviewer,
        assignment: ReviewAssignment, feedback: Optional[List[str]],
        prior_execution: Optional[dict] = None,
    ) -> tuple:
        blind = getattr(agent, "agent_role", "") in {"auditor", "verifier"}
        guidance = list(feedback or [])
        if assignment.priority_hunks:
            guidance.append(
                "Shard coverage is required. Explore these hunk summaries first, then cover every "
                "required_hunk_id: %s" % json.dumps(assignment.priority_hunks, ensure_ascii=False)
            )
        memories = [] if blind else self._recall_memories(state, assignment)
        shard = (state.get("shards_by_id") or {}).get(assignment.shard_id)
        cross_shard = assignment.reason == "cross-shard"
        local_diff = "" if cross_shard else (shard.diff if shard else state["diff"])
        if assignment.reason == "coverage-repair" and shard and assignment.required_hunk_ids:
            local_diff = DiffRiskScanner().select(
                shard.diff, assignment.required_hunk_ids, assignment.shard_id,
            ) or local_diff
        local_parsed = state["parsed"] if cross_shard else (shard.parsed if shard else state["parsed"])
        if assignment.reason == "coverage-repair" and local_diff:
            local_parsed = parse_unified_diff(local_diff)
        bundle = self.context_manager.build(local_diff, assignment.to_dict(), memories)
        if bundle.omitted_files:
            state["coverage_gaps"].append({
                "assignment_id": assignment.assignment_id,
                "files": list(bundle.omitted_files),
                "reason": "local diff compression omitted files from this review context",
            })
        self._emit(
            state, "context-manager", agent.name, "context_prepared",
            bundle.metadata(), assignment.assignment_id,
        )
        tools = self._agent_tools(
            state, assignment, cross_shard=cross_shard, local_diff_override=local_diff,
        )
        working_state = {
            "confirmed_evidence": [], "rejected_hypotheses": [], "open_questions": [],
            "files_inspected": [], "pending_files": list(assignment.files), "candidate_findings": [],
        }
        if self.memory_manager and state.get("task_id") and state.get("repository"):
            self.memory_manager.remember_working_state(
                state.get("tenant_id", "default"), state["repository"], state["task_id"],
                agent.name, assignment.shard_id or "full", working_state,
            )

        def on_event(kind: str, detail: Dict[str, Any]) -> None:
            self._emit(
                state, "agent-runtime", agent.name, kind, detail,
                assignment.assignment_id,
            )
            # The event transcript and raw tool output stay in LoopContext and
            # the task collaboration log.  Only compact path/evidence refs are
            # retained in Working Memory at the checkpoint below.
            if kind == "agent_loop_observation":
                locator = detail.get("locator") if isinstance(detail.get("locator"), dict) else {}
                path = locator.get("path", "")
                if path:
                    working_state["files_inspected"].append(str(path))
                    working_state["pending_files"] = [item for item in working_state["pending_files"] if item != path]

        prior_observations = list((prior_execution or {}).get("observations") or [])
        loop_state = {
            "task_id": state.get("task_id", ""),
            "diff": local_diff, "context": bundle.text,
            "context_metadata": bundle.metadata(), "parsed": local_parsed,
            "assignment": assignment.to_dict(), "feedback": guidance,
            "inbox": [] if blind else self._bus(state).inbox(agent.name, assignment.assignment_id),
            "memories": memories, "available_tools": tools.catalog(),
            "pr_map": state["pr_map"].compact(),
            "shard_identity": shard.identity() if shard else {"id": "full", "files": assignment.files},
            "observations": prior_observations,
            "snapshot_id": state.get("source_sha", ""),
            "producer_run_id": artifact_fingerprint({
                "task": state.get("task_id", ""), "assignment": assignment.assignment_id,
                "agent": agent.name, "snapshot": state.get("source_sha", ""),
            })[:24],
            "cross_shard": cross_shard,
        }
        last_context = {"metadata": bundle.metadata()}
        observed_events: List[Dict[str, Any]] = []
        usage_totals = {
            "input_tokens": 0, "output_tokens": 0, "cached_tokens": 0,
            "usage_source": "provider", "llm_calls": 0, "physical_llm_calls": 0,
            "request_attempts": [], "tool_calls": 0,
        }

        def managed_step(loop_iteration: Dict[str, Any]) -> Dict[str, Any]:
            deadline = float(state.get("deadline_monotonic", 0) or 0)
            if deadline and time.monotonic() >= deadline:
                raise RuntimeBudgetExceeded("review wall deadline exceeded")
            context_assignment = assignment.to_dict()
            context_assignment["pr_map"] = state["pr_map"].compact()
            managed = self.context_manager.compose(
                bundle, context_assignment, feedback=guidance,
                inbox=loop_iteration.get("inbox") or [], memories=memories,
                observations=loop_iteration.get("observations") or [],
                tools=tools.catalog(), loop_context=loop_iteration.get("loop_context") or {},
                frozen_context={
                    "assignment": assignment.to_dict(), "pr_map": state["pr_map"].compact(),
                    "shard": loop_iteration.get("shard_identity"), "tools": tools.catalog(),
                    "rules": "Report only introduced, changed-line defects with evidence.",
                },
                retrieved_context=[item for item in (loop_iteration.get("observations") or [])
                                   if item.get("tool") in {"read_file", "read_diff", "grep_repo", "find_symbol", "find_references"}],
            )
            metadata = managed.metadata()
            last_context["metadata"] = metadata
            self._emit(
                state, "context-manager", agent.name, "context_window_prepared",
                metadata, assignment.assignment_id,
            )
            prepared = dict(loop_iteration)
            prepared["context"] = managed.text
            prepared["managed_context"] = managed.text
            prepared["context_metadata"] = metadata
            prepared["frozen_context"] = {
                "assignment": assignment.to_dict(), "pr_map": state["pr_map"].compact(),
                "shard": prepared["shard_identity"], "tools": tools.catalog(),
            }
            estimated = managed.estimated_tokens
            budget = state["shared_budget"]
            budgeted_model = getattr(agent, "execution_kind", "") == "agent"
            remaining_output = budget.remaining("output_tokens")
            if remaining_output <= 0:
                raise RuntimeBudgetExceeded("shared output token budget exceeded")
            per_request_output = int((state.get("execution_context") or {}).get(
                "max_output_tokens_per_request", 1200
            ) or 1200)
            prepared["_max_output_tokens"] = max(
                1, min(remaining_output, per_request_output)
            )
            stage_limit = int((state.get("execution_context") or {}).get(
                "stage_max_llm_calls", assignment.budget.get("max_loop_calls", 0)
            ) or 0)
            requests_remaining = max(0, stage_limit - usage_totals["llm_calls"]) if stage_limit else 0
            prepared["llm_requests_remaining"] = requests_remaining
            # Once only final+optional schema-repair capacity remains, tools
            # are no longer allowed to consume the final response slot.
            prepared["must_return_final"] = bool(stage_limit and requests_remaining <= 2)
            if stage_limit and usage_totals["llm_calls"] >= stage_limit:
                raise RuntimeBudgetExceeded("agent stage LLM request budget exceeded")
            stage = "review" if assignment.reason in {
                "initial-plan", "coverage-owner", "risk-domain-second-opinion", "coverage-repair",
                "cross-shard",
            } else "tail"
            if budgeted_model and (not budget.reserve("llm_calls", 1, stage=stage)
                                  or not budget.reserve("input_tokens", estimated, stage=stage)):
                raise RuntimeBudgetExceeded("shared agent budget exceeded")
            configured_timeout = int(getattr(agent, "timeout", 60) or 60)
            prepared["_request_timeout_seconds"] = (
                max(1, min(configured_timeout, int(deadline - time.monotonic())))
                if deadline else configured_timeout
            )
            try:
                action = getattr(agent, "agent_step")(prepared)
            except ModelRequestFailure as exc:
                request_attempts = [item.to_dict() for item in exc.attempts]
                for _ in request_attempts[1:]:
                    budget.reserve("llm_calls", 1)
                for request_attempt in request_attempts:
                    usage_totals["request_attempts"].append(dict(request_attempt))
                    self._emit(
                        state, "agent-runtime", agent.name, "llm_request_failed",
                        {key: value for key, value in request_attempt.items()
                         if key not in {"logical_call_id", "request_id"}},
                        assignment.assignment_id,
                    )
                actual_input = sum(int(item.get("input_tokens", 0) or 0)
                                   for item in request_attempts)
                output_tokens = sum(int(item.get("output_tokens", 0) or 0)
                                    for item in request_attempts)
                if actual_input:
                    budget.replace_estimate("input_tokens", estimated, actual_input)
                if output_tokens:
                    budget.reserve("output_tokens", output_tokens)
                usage_totals["llm_calls"] += max(1, len(request_attempts))
                usage_totals["physical_llm_calls"] += max(1, len(request_attempts))
                usage_totals["input_tokens"] += actual_input or estimated
                usage_totals["output_tokens"] += output_tokens
                usage_totals["cached_tokens"] += sum(
                    int(item.get("cached_tokens", 0) or 0) for item in request_attempts
                )
                if any(item.get("usage_source") != "provider" for item in request_attempts):
                    usage_totals["usage_source"] = "estimated"
                raise
            if isinstance(action, dict):
                if str(action.get("action", "")).lower() == "tool":
                    tool_limit = int(assignment.budget.get("tool_budget", 0) or 0)
                    if ((tool_limit and usage_totals["tool_calls"] >= tool_limit)
                            or (budgeted_model and not budget.reserve("tool_calls", 1, stage=stage))):
                        raise RuntimeBudgetExceeded("shared tool-call budget exceeded")
                    usage_totals["tool_calls"] += 1
                usage = dict(action.get("_usage") or {})
                request_attempts = list(action.get("_request_attempts") or [])
                stage_attempts_exceeded = bool(
                    stage_limit
                    and usage_totals["llm_calls"] + max(1, len(request_attempts)) > stage_limit
                )
                if len(request_attempts) > 1:
                    for _ in request_attempts[1:]:
                        if budgeted_model and not budget.reserve("llm_calls", 1, stage=stage):
                            raise RuntimeBudgetExceeded("shared LLM request budget exceeded after model fallback")
                for request_attempt in request_attempts:
                    usage_totals["request_attempts"].append(dict(request_attempt))
                    event = (
                        "llm_model_fallback" if request_attempt.get("fallback_reason")
                        else "llm_request_completed" if request_attempt.get("status") == "success"
                        else "llm_request_failed"
                    )
                    self._emit(
                        state, "agent-runtime", agent.name, event,
                        {key: value for key, value in request_attempt.items()
                         if key not in {"logical_call_id", "request_id"}},
                        assignment.assignment_id,
                    )
                actual_input = int(usage.get("input_tokens", 0) or 0)
                if not actual_input:
                    actual_input = estimated
                    usage["usage_source"] = "estimated"
                if budgeted_model:
                    budget.replace_estimate("input_tokens", estimated, actual_input)
                output_tokens = int(usage.get("output_tokens", 0) or 0)
                if budgeted_model and output_tokens and not budget.reserve("output_tokens", output_tokens):
                    raise RuntimeBudgetExceeded("shared output token budget exceeded")
                usage_totals["llm_calls"] += max(1, len(request_attempts))
                if not action.get("_physical_request_replayed"):
                    usage_totals["physical_llm_calls"] += max(1, len(request_attempts))
                for field, value in (("input_tokens", actual_input),
                                     ("output_tokens", output_tokens),
                                     ("cached_tokens", int(usage.get("cached_tokens", 0) or 0))):
                    usage_totals[field] += value
                if usage.get("usage_source") != "provider":
                    usage_totals["usage_source"] = "estimated"
                action["_context_tokens"] = managed.estimated_tokens
                action["_context_max_tokens"] = self.context_manager.max_tokens
                if stage_attempts_exceeded:
                    raise RuntimeBudgetExceeded(
                        "agent stage LLM request budget exceeded after fallback"
                    )
            return action

        def traced_event(kind: str, detail: Dict[str, Any]) -> None:
            on_event(kind, detail)
            if kind == "agent_loop_observation":
                observed_events.append(dict(detail))

        try:
            local_loop = AgentLoop(
                max_steps=int(assignment.budget.get("max_steps", self.agent_loop.max_steps)),
                timeout_seconds=self.agent_loop.timeout_seconds,
                active_rounds=self.agent_loop.active_rounds,
                soft_compact_ratio=self.agent_loop.soft_compact_ratio,
                hard_compact_ratio=self.agent_loop.hard_compact_ratio,
                context_budget=self.agent_loop.context_budget,
            )
            result = local_loop.run(
                managed_step, tools, loop_state, traced_event,
            )
        except Exception as exc:
            # Keep the successful work that preceded a terminal failure so the
            # summary can expose an auditable failed AgentRun and model trace.
            exc.execution = {
                "loop_steps": int(usage_totals["llm_calls"]),
                "loop_stop_reason": "failed",
                "context": last_context["metadata"],
                "tool_calls": int(usage_totals["tool_calls"]),
                "repo_tool_calls": tools.retrieval.calls,
                "retrieved_context_tokens": tools.retrieval.retrieved_bytes // 4,
                "observations": list(observed_events),
                "usage": dict(usage_totals),
            }
            raise
        findings = list(result.output or [])
        expected_findings = assignment.reason not in {
            "blind-challenge", "verify-findings", "challenge-requested-revision",
        }
        if expected_findings and not all(isinstance(item, Finding) for item in findings):
            raise TypeError("agent loop final output must contain Finding objects")
        if not expected_findings and not all(isinstance(item, dict) for item in findings):
            raise TypeError("challenge/revision output must contain objects")
        working_state["candidate_findings"] = [
            finding_key(item) for item in findings if isinstance(item, Finding)
        ]
        working_state["confirmed_evidence"] = [
            str(item.get("id", "")) for item in result.loop_context.get("pinned_evidence", []) if item.get("id")
        ]
        if self.memory_manager and state.get("task_id") and state.get("repository"):
            self.memory_manager.remember_working_state(
                state.get("tenant_id", "default"), state["repository"], state["task_id"],
                agent.name, assignment.shard_id or "full", working_state,
            )
        return findings, {
            "loop_steps": result.steps, "loop_stop_reason": result.stop_reason,
            "context": last_context["metadata"], "memories_recalled": len(memories),
            "tools_available": len(tools.names()),
            "tool_calls": max(0, len(result.observations) - len(prior_observations)),
            "repo_tool_calls": tools.retrieval.calls,
            "retrieved_context_tokens": tools.retrieval.retrieved_bytes // 4,
            "context_compaction_count": result.loop_context.get("compaction_count", 0),
            "compressed_rounds": result.loop_context.get("compressed_rounds", 0),
            "pinned_evidence_count": len(result.loop_context.get("pinned_evidence", [])),
            "pinned_evidence": result.loop_context.get("pinned_evidence", []),
            "observations": list(result.observations),
            "dropped_observations": int(last_context["metadata"].get("dropped_observations", 0)),
            "usage": usage_totals,
        }

    def _invoke_agent(
        self, state: CollaborationState, agent: Reviewer,
        assignment: ReviewAssignment, feedback: Optional[List[str]] = None,
        prior_execution: Optional[dict] = None,
    ) -> tuple:
        last_error = None
        failed_request_attempts = []
        for attempt in range(1, self.agent_retries + 2):
            self._emit(
                state, "coordinator", agent.name, "attempt_started",
                {"attempt": attempt, "round": assignment.round}, assignment.assignment_id,
            )
            try:
                loop_stepper = getattr(agent, "agent_step", None)
                execution = {"loop_steps": 0, "loop_stop_reason": "one-shot"}
                if loop_stepper:
                    findings, execution = self._run_agent_loop(
                        state, agent, assignment, feedback, prior_execution
                    )
                else:
                    local_shard = (state.get("shards_by_id") or {}).get(assignment.shard_id)
                    local_diff = local_shard.diff if local_shard else state["diff"]
                    bundle = self.context_manager.build(local_diff, assignment.to_dict())
                    execution.update({"context": {**bundle.metadata(), "estimated_tokens": bundle.final_tokens}})
                    if bundle.omitted_files:
                        state["coverage_gaps"].append({
                            "assignment_id": assignment.assignment_id,
                            "files": list(bundle.omitted_files),
                            "reason": "local diff compression omitted files from this review context",
                        })
                    collaborative = getattr(agent, "review_assignment", None)
                    if collaborative:
                        findings = collaborative(
                            local_diff, local_shard.parsed if local_shard else state["parsed"], assignment.to_dict(),
                            list(feedback or []),
                            self._bus(state).inbox(agent.name, assignment.assignment_id),
                        )
                    else:
                        findings = agent.review(local_diff, local_shard.parsed if local_shard else state["parsed"])
                self._emit(
                    state, agent.name, self.critic.name, "specialist_evidence",
                    {
                        "attempt": attempt, "round": assignment.round,
                        "findings": [
                            {key: (value.to_dict() if hasattr(value, "to_dict") else value)
                             for key, value in item.items()}
                            if isinstance(item, dict) else item.to_dict()
                            for item in findings
                        ],
                        "execution": execution,
                    }, assignment.assignment_id,
                )
                return findings, attempt, "", execution
            except Exception as exc:
                last_error = str(exc)
                failed_execution = dict(getattr(exc, "execution", {}) or {})
                failed_request_attempts.extend(
                    item.to_dict() if hasattr(item, "to_dict") else dict(item)
                    for item in list(getattr(exc, "attempts", ()) or ())
                )
                self._emit(
                    state, agent.name, self.planner.name, "agent_failure",
                    {"attempt": attempt, "error": last_error[:1000]},
                    assignment.assignment_id,
                )
                if isinstance(exc, RuntimeBudgetExceeded):
                    return (
                        [], attempt, last_error,
                        failed_execution or {
                            "loop_steps": 0, "loop_stop_reason": "budget-exhausted",
                            "usage": {"llm_calls": len(failed_request_attempts),
                                      "input_tokens": 0, "output_tokens": 0,
                                      "cached_tokens": 0, "usage_source": "estimated",
                                      "request_attempts": failed_request_attempts},
                        },
                    )
                if attempt <= self.agent_retries:
                    self._emit(
                        state, self.planner.name, agent.name, "retry_request",
                        {"next_attempt": attempt + 1, "reason": last_error[:500]},
                        assignment.assignment_id,
                    )
        return (
            [], self.agent_retries + 1, last_error or "unknown agent failure",
            {"loop_steps": 0, "loop_stop_reason": "failed",
             "request_attempts": failed_request_attempts,
             "usage": {"llm_calls": len(failed_request_attempts),
                       "input_tokens": 0, "output_tokens": 0, "cached_tokens": 0,
                       "usage_source": "estimated",
                       "request_attempts": failed_request_attempts}},
        )

    def _replacement_candidates(self, failed_agent: Reviewer) -> List[Reviewer]:
        failed_role = str(getattr(failed_agent, "agent_role", ""))
        values = [
            item for item in self.agents if item is not failed_agent
            and getattr(item, "agent_role", "") != "auditor"
            and (not failed_role or getattr(item, "agent_role", "") == failed_role)
        ]
        if failed_role in {"", "primary"} and all(
            item.name != self.fallback_agent.name for item in values
        ):
            values.append(self.fallback_agent)
        return values

    def _run_assignment(
        self, state: CollaborationState, assignment: ReviewAssignment,
        agent: Reviewer, feedback: Optional[List[str]] = None,
    ) -> dict:
        checkpoint = self._restore_assignment_checkpoint(state, assignment, agent)
        if checkpoint is not None:
            return checkpoint
        stage = "review" if assignment.reason in {
            "initial-plan", "coverage-owner", "risk-domain-second-opinion", "coverage-repair",
            "cross-shard",
        } else "tail"
        if (getattr(agent, "execution_kind", "") == "agent"
                and not state["shared_budget"].reserve("agent_runs", 1, stage=stage)):
            receipt = self._coverage_receipt(assignment, "budget-exhausted", ["global budget/tail reserve"])
            return {
                "agent": agent.name, "assignment_id": assignment.assignment_id,
                "attempts": 0, "status": "budget_exhausted", "findings": [],
                "error": "shared agent run budget exceeded", "substituted_for": "",
                "assignment": assignment,
                "execution": {"loop_steps": 0, "loop_stop_reason": "shared-budget"},
                "coverage_receipt": receipt,
            }
        findings, attempts, error, execution = self._invoke_agent(
            state, agent, assignment, feedback
        )
        result = {
            "agent": agent.name, "assignment_id": assignment.assignment_id,
            "attempts": attempts, "status": "completed" if not error else (
                "timed_out" if "budget" in error.lower() or "timeout" in error.lower() else "failed"
            ),
            "findings": findings, "error": error, "substituted_for": "",
            "assignment": assignment, "execution": execution,
            "coverage_receipt": self._coverage_receipt(
                assignment, "complete" if not error else "budget-exhausted" if "budget" in error.lower() else "partial",
                [error] if error else [], execution,
            ),
        }
        if not error:
            self._save_assignment_checkpoint(state, result)
            return result
        if callable(getattr(agent, "agent_step", None)):
            state.setdefault("llm_failures", []).append({
                "assignment_id": assignment.assignment_id, "shard_id": assignment.shard_id,
                "agent": agent.name, "error": error[:500],
            })
        if "budget" in error.lower():
            # A hard budget is terminal for this assignment. Handing it to a
            # substitute would hide the failed semantic run and overspend.
            return result
        if assignment.reason == "coverage-repair":
            # A repair is deliberately scoped to the original agent's missing
            # hunks; a general fallback would falsely broaden that coverage.
            return result
        replacement = self.planner.replan(
            assignment, self._replacement_candidates(agent), error
        )
        if replacement is None:
            return result
        substitute = next(
            item for item in self._replacement_candidates(agent)
            if item.name == replacement.agent
        )
        self._emit(
            state, self.planner.name, substitute.name, "assignment_handoff",
            {
                "from": agent.name, "reason": error[:500],
                "assignment": replacement.to_dict(),
            }, assignment.assignment_id,
        )
        findings, replacement_attempts, replacement_error, replacement_execution = self._invoke_agent(
            state, substitute, replacement, ["Take over after %s failed: %s" % (agent.name, error)]
        )
        return {
            "agent": substitute.name, "assignment_id": assignment.assignment_id,
            "attempts": attempts + replacement_attempts,
            "status": ("fallback" if not replacement_error and substitute.name == self.fallback_agent.name
                       else "completed") if not replacement_error else (
                "timed_out" if "budget" in replacement_error.lower() or "timeout" in replacement_error.lower() else "failed"
            ),
            "findings": findings, "error": replacement_error,
            "substituted_for": agent.name,
            "assignment": replacement, "execution": replacement_execution,
            "coverage_receipt": self._coverage_receipt(
                replacement, "complete" if not replacement_error else "partial",
                [replacement_error] if replacement_error else [], replacement_execution,
            ),
        }

    @staticmethod
    def _coverage_receipt(
        assignment: ReviewAssignment, status: str, gaps: List[str] = None, execution: dict = None,
    ) -> dict:
        execution = execution or {}
        inspected = {
            str((item.get("locator") or {}).get("path", ""))
            for item in execution.get("observations", []) if item.get("ok")
        }
        # Reading the assigned diff is coverage even if no repository snapshot
        # was supplied; a budget-exhausted assignment remains explicitly open.
        covered = list(assignment.files) if status == "complete" else sorted(inspected.intersection(assignment.files))
        uncovered = sorted(set(assignment.files).difference(covered))
        required_hunks = list(assignment.required_hunk_ids)
        priority_by_id = {
            str(item.get("hunk_id", "")): item for item in assignment.priority_hunks
            if isinstance(item, dict) and item.get("hunk_id")
        }
        covered_hunks = list(required_hunks) if status == "complete" else []
        if status != "complete":
            for hunk_id in required_hunks:
                hunk = priority_by_id.get(hunk_id, {})
                for observation in execution.get("observations", []):
                    locator = observation.get("locator") or {}
                    if not observation.get("ok") or locator.get("path") != hunk.get("path"):
                        continue
                    start = int(locator.get("start_line", locator.get("line", 0)) or 0)
                    end = int(locator.get("end_line", locator.get("line", start)) or start)
                    if start and start <= int(hunk.get("end_line", 0)) and end >= int(hunk.get("start_line", 0)):
                        covered_hunks.append(hunk_id)
                        break
        covered_hunks = sorted(set(covered_hunks))
        uncovered_hunks = [item for item in required_hunks if item not in set(covered_hunks)]
        return CoverageReceipt(
            assignment_id=assignment.assignment_id, shard_id=assignment.shard_id,
            covered_files=covered, uncovered_files=uncovered,
            required_hunks=required_hunks, covered_hunks=covered_hunks,
            uncovered_hunks=uncovered_hunks,
            open_questions=list((execution.get("loop_context") or {}).get("open_questions", []))[:8],
            budget_gaps=[item for item in (gaps or []) if item], status=status,
        ).to_dict()

    def _assignment_fingerprint(
        self, state: CollaborationState, assignment: ReviewAssignment, agent: Reviewer,
    ) -> str:
        return artifact_fingerprint({
            "source_sha": state.get("source_sha", ""),
            "assignment": assignment.to_dict(),
            "agent": self._agent_spec(agent, assignment.assignment_id).to_dict(),
            "review_mode": self.review_mode, "challenge_strategy": self.challenge_strategy,
            "context_architecture": self.context_architecture,
            "specialist_activation": self.specialist_activation,
            "budget": state["shared_budget"].snapshot()["limits"],
            "decision_policy": self.decision_policy.VERSION,
        })

    def _restore_assignment_checkpoint(
        self, state: CollaborationState, assignment: ReviewAssignment, agent: Reviewer,
    ) -> Optional[dict]:
        if (not self.store or not state.get("task_id")
                or getattr(agent, "execution_kind", "") != "agent"):
            return None
        node = "collaboration.specialist.%s" % assignment.assignment_id
        item = self.store.load_checkpoints(state["task_id"]).get(node)
        fingerprint = self._assignment_fingerprint(state, assignment, agent)
        if not item or item.get("status") != "completed" or item.get("fingerprint") != fingerprint:
            return None
        saved = item.get("state") or {}
        findings = []
        for value in saved.get("findings", []):
            value = dict(value)
            value["severity"] = Severity(value["severity"])
            findings.append(Finding(**value))
        return {
            "agent": saved.get("agent", agent.name),
            "assignment_id": assignment.assignment_id,
            "attempts": int(saved.get("attempts", 1)), "status": "completed",
            "findings": findings, "error": "", "substituted_for": "",
            "assignment": assignment, "execution": dict(saved.get("execution") or {}),
            "coverage_receipt": self._coverage_receipt(
                assignment, "complete", execution=dict(saved.get("execution") or {}),
            ),
            "checkpoint_restored": True,
        }

    def _save_assignment_checkpoint(self, state: CollaborationState, outcome: dict) -> None:
        agent = self._agent_by_name(outcome["agent"])
        assignment = outcome["assignment"]
        if (not agent or not self.store or not state.get("task_id")
                or getattr(agent, "execution_kind", "") != "agent"):
            return
        context = state.get("execution_context") or {}
        self.store.save_checkpoint(
            state["task_id"], "collaboration.specialist.%s" % assignment.assignment_id,
            {
                "schema_version": 1, "agent": outcome["agent"],
                "assignment": assignment.to_dict(),
                "findings": [item.to_dict() for item in outcome["findings"]],
                "attempts": outcome["attempts"], "execution": outcome.get("execution") or {},
            },
            fingerprint=self._assignment_fingerprint(state, assignment, agent),
            run_token=str(context.get("run_token", "")),
            claim_token=str(context.get("claim_token", "")),
        )

    def _specialist_node(
        self, state: CollaborationState, worker_limit: int = None,
    ) -> Dict[str, Any]:
        outcomes = []
        by_name = {item.name: item for item in self.agents}
        by_name.setdefault(self.fallback_agent.name, self.fallback_agent)
        assignments = state["plan"].assignments
        with ThreadPoolExecutor(
            max_workers=min(worker_limit or self.max_workers, max(1, len(assignments)))
        ) as pool:
            futures = {
                pool.submit(self._run_assignment, state, assignment, by_name[assignment.agent]): assignment
                for assignment in assignments
            }
            for future in as_completed(futures):
                outcomes.append(future.result())
        findings = []
        sources: Dict[str, List[str]] = {}
        assignment_map = {item.assignment_id: item for item in state["plan"].assignments}
        for outcome in outcomes:
            receipt = outcome.get("coverage_receipt")
            if receipt:
                state.setdefault("coverage_receipts", []).append(receipt)
                if (receipt.get("budget_gaps") or receipt.get("uncovered_files")
                        or receipt.get("uncovered_hunks")):
                    state.setdefault("budget_gaps", []).append(receipt)
            owner_assignment = assignment_map.get(outcome["assignment_id"], outcome["assignment"])
            # Tool evidence must exist before an LLM's stable evidence_refs
            # can be linked to its proposed findings.
            state["evidence_ledger"].record((outcome.get("execution") or {}).get("pinned_evidence", []))
            for finding in outcome["findings"]:
                key = finding_key(finding)
                finding.evidence_refs = state["evidence_ledger"].link(
                    key, finding.evidence_refs, finding.path, finding.line,
                )
                sources.setdefault(key, []).append(outcome["assignment_id"])
                findings.append(finding)
            if (outcome["status"] not in {"completed", "fallback"}
                    and owner_assignment.reason == "coverage-owner"
                    and self.review_pipeline == "classic"):
                state["coverage_gaps"].append({
                    "assignment_id": outcome["assignment_id"],
                    "shard_id": owner_assignment.shard_id,
                    "files": list(owner_assignment.files),
                    "reason": "responsible shard review %s" % outcome["status"],
                    "error": str(outcome.get("error", ""))[:500],
                })
        return {
            "specialist_findings": findings,
            "finding_sources": sources,
            "agent_outcomes": outcomes,
            "assignments_by_agent": assignment_map,
        }

    def _coverage_repair_node(self, state: CollaborationState) -> Dict[str, Any]:
        """Spend only review-stage headroom on hunk coverage left incomplete."""
        if self.review_pipeline != "bounded-v2" or self.review_mode != "adaptive_multi_agent":
            return {"coverage_repair_outcomes": []}
        by_name = {item.name: item for item in self.agents}
        original_outcomes = list(state.get("agent_outcomes", []))
        repairs = []
        for outcome in original_outcomes:
            receipt = outcome.get("coverage_receipt") or {}
            missing = list(receipt.get("uncovered_hunks") or [])
            assignment = outcome.get("assignment")
            agent = by_name.get(getattr(assignment, "agent", ""))
            if not missing or assignment is None or agent is None:
                continue
            priorities = [item for item in assignment.priority_hunks
                          if item.get("hunk_id") in set(missing)]
            repairs.append((
                replace(
                    assignment, assignment_id=assignment.assignment_id + ".coverage-repair",
                    reason="coverage-repair", round=assignment.round + 1,
                    required_hunk_ids=missing, priority_hunks=priorities,
                ),
                agent,
            ))
        if not repairs:
            return {"coverage_repair_outcomes": []}
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(repairs))) as pool:
            futures = [pool.submit(
                self._run_assignment, state, assignment, agent,
                ["Re-review only these previously uncovered hunk IDs: %s" % assignment.required_hunk_ids],
            ) for assignment, agent in repairs]
            outcomes = [future.result() for future in futures]

        findings = list(state.get("specialist_findings", []))
        sources = {key: list(value) for key, value in (state.get("finding_sources") or {}).items()}
        assignments = dict(state.get("assignments_by_agent") or {})
        for outcome in outcomes:
            assignment = outcome["assignment"]
            assignments[outcome["assignment_id"]] = assignment
            receipt = outcome.get("coverage_receipt") or {}
            if receipt:
                state.setdefault("coverage_receipts", []).append(receipt)
                if receipt.get("budget_gaps") or receipt.get("uncovered_hunks"):
                    state.setdefault("budget_gaps", []).append(receipt)
            state["evidence_ledger"].record((outcome.get("execution") or {}).get("pinned_evidence", []))
            for finding in outcome.get("findings", []):
                key = finding_key(finding)
                if any(finding_key(existing) == key for existing in findings):
                    continue
                finding.evidence_refs = state["evidence_ledger"].link(
                    key, finding.evidence_refs, finding.path, finding.line,
                )
                findings.append(finding)
                sources.setdefault(key, []).append(outcome["assignment_id"])
            if receipt.get("uncovered_hunks"):
                state.setdefault("coverage_gaps", []).append({
                    "assignment_id": outcome["assignment_id"], "shard_id": assignment.shard_id,
                    "files": list(assignment.files), "uncovered_hunks": list(receipt["uncovered_hunks"]),
                    "reason": "required hunk coverage incomplete after repair",
                    "error": str(outcome.get("error", ""))[:500],
                })
        return {
            "specialist_findings": findings, "finding_sources": sources,
            "agent_outcomes": original_outcomes + outcomes,
            "assignments_by_agent": assignments, "coverage_repair_outcomes": outcomes,
        }

    def _run_cross_shard_seed(self, state: CollaborationState) -> Optional[dict]:
        """Run the tracer from PR relations only, independently of specialists."""
        if self.review_mode == "rules_only" or len(state.get("shards") or []) <= 1:
            return None
        agent = next((item for item in self.agents
                      if callable(getattr(item, "agent_step", None))
                      and getattr(item, "agent_role", "") == "cross_shard"), None)
        if agent is None:
            agent = next((item for item in self.agents
                          if callable(getattr(item, "agent_step", None))
                          and getattr(item, "agent_role", "") == "primary"), None)
        if agent is None:
            return None
        relation_map = {
            "symbols": {item.path: item.symbols for item in state["pr_map"].files if item.symbols},
            "imports": {item.path: item.imports for item in state["pr_map"].files if item.imports},
            "shards": [item.identity() for item in state.get("shards", [])],
        }
        assignment = ReviewAssignment(
            agent=agent.name, objective=self.cross_shard_tracer.objective(),
            files=list(state["parsed"].files), risk_domains=list(getattr(agent, "domains", ())),
            assignment_id="cross-shard-tracer", reason="cross-shard", coverage_scope="cross-shard",
            budget=self.budget_policy.for_shard(sum(item.changed_lines for item in state.get("shards", []))).to_dict(),
        )
        outcome = self._run_assignment(state, assignment, agent, [
            "Structured PR relation seeds: %s" % json.dumps(relation_map, ensure_ascii=False, sort_keys=True),
            "Trace only caller/callee, API, schema, and producer/consumer contract compatibility.",
        ])
        return {"agent": agent, "assignment": assignment, "outcome": outcome}

    def _merge_cross_shard_seed(self, state: CollaborationState, seed: Optional[dict]) -> Dict[str, Any]:
        if seed is None:
            return {"cross_shard_findings": [], "cross_shard_execution": {}, "cross_shard_outcomes": []}
        agent, assignment, outcome = seed["agent"], seed["assignment"], seed["outcome"]
        execution = outcome.get("execution") or {}
        findings = []
        state.setdefault("assignments_by_agent", {})[assignment.assignment_id] = assignment
        if outcome.get("coverage_receipt"):
            state.setdefault("coverage_receipts", []).append(outcome["coverage_receipt"])
        if outcome.get("status") not in {"completed", "fallback"}:
            state.setdefault("coverage_gaps", []).append({
                "shard_id": "cross-shard", "files": [], "reason": "cross-shard tracer incomplete",
                "agent": agent.name, "error": str(outcome.get("error", ""))[:500],
            })
        else:
            state["evidence_ledger"].record(execution.get("pinned_evidence", []))
            for finding in outcome.get("findings", []):
                finding.evidence_refs = state["evidence_ledger"].link(
                    finding_key(finding), finding.evidence_refs, finding.path, finding.line,
                )
                findings.append(finding)
                state.setdefault("finding_sources", {}).setdefault(finding_key(finding), []).append(assignment.assignment_id)
        state.setdefault("specialist_findings", []).extend(findings)
        self._emit(state, "context-orchestrator", "specialists", "cross_shard_review", {
            "agents": [self.cross_shard_tracer.name], "findings": [item.to_dict() for item in findings],
        })
        return {
            "cross_shard_findings": findings, "cross_shard_outcomes": [outcome],
            "cross_shard_execution": {
                "repo_tool_calls": int(execution.get("repo_tool_calls", 0)),
                "retrieved_context_tokens": int(execution.get("retrieved_context_tokens", 0)),
                "llm_calls": int((execution.get("usage") or {}).get("llm_calls", 0)),
            },
        }

    def _investigation_node(self, state: CollaborationState) -> Dict[str, Any]:
        """Fan out shard experts and the independent relation tracer together."""
        if self.max_workers <= 1:
            specialists = self._specialist_node(state)
            state.update(specialists)
            tracer = self._merge_cross_shard_seed(state, self._run_cross_shard_seed(state))
        else:
            with ThreadPoolExecutor(max_workers=2) as pool:
                specialist_future = pool.submit(self._specialist_node, state, self.max_workers - 1)
                tracer_future = pool.submit(self._run_cross_shard_seed, state)
                specialists = specialist_future.result()
                # Merge only after specialists are complete. The seed itself
                # had no access to their findings or evidence.
                state.update(specialists)
                tracer = self._merge_cross_shard_seed(state, tracer_future.result())
        # `_merge_cross_shard_seed` appends linked findings to the live state.
        # Preserve that merged collection in the runtime-node result: otherwise
        # the specialist snapshot would overwrite it when AgentRuntime applies
        # this node's output.
        merged = dict(specialists)
        merged["specialist_findings"] = list(state.get("specialist_findings", []))
        merged["finding_sources"] = dict(state.get("finding_sources", {}))
        merged["assignments_by_agent"] = dict(state.get("assignments_by_agent", {}))
        return {**merged, **tracer}

    def _legacy_cross_shard_node(self, state: CollaborationState) -> Dict[str, Any]:
        """Run a compact, bounded second pass without ever restoring the full Diff."""
        if self.review_mode == "rules_only":
            return {"cross_shard_findings": [], "cross_shard_execution": {}}
        shards = state.get("shards") or []
        if len(shards) <= 1:
            return {"cross_shard_findings": []}
        summary = {
            "pr_map": state["pr_map"].compact(),
            "shards": [item.identity() for item in shards],
            "findings": [item.to_dict() for item in state.get("specialist_findings", [])],
            "coverage_gaps": list(state.get("coverage_gaps") or []),
            "llm_failures": list(state.get("llm_failures") or []),
            "changed_symbols": {
                item.path: item.symbols for item in state["pr_map"].files if item.symbols
            },
            "evidence": [
                evidence for evidence in state["evidence_ledger"].pinned()
            ],
        }
        findings = []
        cross_calls = 0
        cross_bytes = 0
        cross_llm_calls = 0
        activated = set(state.get("activated_domains") or [])
        for agent in self.agents:
            stepper = getattr(agent, "agent_step", None)
            role = getattr(agent, "agent_role", "")
            if (not stepper or role == "auditor"
                    or (role not in {"", "primary"} and role not in activated)):
                continue
            try:
                text = json.dumps(summary, ensure_ascii=False, sort_keys=True)
                bundle = ContextBundle(
                    text, False, self.context_manager.estimate_tokens(text),
                    self.context_manager.estimate_tokens(text), strategy="cross-shard-summary",
                )
                assignment = {"agent": agent.name, "objective": (
                    "Check cross-shard API signatures, callers, schemas, configuration and data flow."
                ), "shard_id": "cross-shard", "coverage_scope": "cross-shard"}
                tools = self._agent_tools(state, ReviewAssignment(
                    agent=agent.name, objective=assignment["objective"], files=list(state["parsed"].files),
                    risk_domains=list(getattr(agent, "domains", ())), assignment_id="cross-shard",
                    shard_id="", coverage_scope="cross-shard",
                ), cross_shard=True)

                def step(loop_state):
                    managed = self.context_manager.compose(
                        bundle, assignment, loop_context=loop_state.get("loop_context") or {},
                        frozen_context={"assignment": assignment, "pr_map": summary["pr_map"],
                                        "rules": "Return only findings located on changed lines."},
                        tools=tools.catalog(),
                    )
                    prepared = dict(loop_state)
                    prepared.update({"cross_shard": True, "cross_shard_context": summary,
                                     "managed_context": managed.text, "context": managed.text,
                                     "parsed": state["parsed"]})
                    action = stepper(prepared)
                    if isinstance(action, dict):
                        action["_context_tokens"] = managed.estimated_tokens
                        action["_context_max_tokens"] = self.context_manager.max_tokens
                    return action

                result = self.agent_loop.run(step, tools, {
                    "cross_shard": True, "cross_shard_context": summary,
                    "available_tools": tools.catalog(), "observations": [], "parsed": state["parsed"],
                    "assignment": assignment, "shard_identity": {"id": "cross-shard", "files": []},
                })
                valid_locations = {(item.path, item.line) for item in state["parsed"].added_lines}
                findings.extend(item for item in list(result.output or []) if isinstance(item, Finding)
                                and (item.path, item.line) in valid_locations)
                state["evidence_ledger"].record(result.loop_context.get("pinned_evidence", []))
                cross_calls += tools.retrieval.calls
                cross_bytes += tools.retrieval.retrieved_bytes
                cross_llm_calls += result.steps
            except Exception as exc:
                state["coverage_gaps"].append({
                    "shard_id": "cross-shard", "files": [], "reason": "cross-shard review failed",
                    "agent": agent.name, "error": str(exc)[:500],
                })
        if findings:
            state["specialist_findings"].extend(findings)
            for item in findings:
                state["finding_sources"].setdefault(finding_key(item), []).append("cross-shard")
        self._emit(state, "context-orchestrator", "specialists", "cross_shard_review", {
            "shard_count": len(shards), "findings": [item.to_dict() for item in findings],
        })
        return {"cross_shard_findings": findings, "cross_shard_execution": {
            "repo_tool_calls": cross_calls, "retrieved_context_tokens": cross_bytes // 4,
            "llm_calls": cross_llm_calls,
        }}

    def _cross_shard_node(self, state: CollaborationState) -> Dict[str, Any]:
        if self.review_pipeline == "classic":
            return self._legacy_cross_shard_node(state)
        return self._merge_cross_shard_seed(state, self._run_cross_shard_seed(state))

    def _agent_by_name(self, name: str) -> Optional[Reviewer]:
        return next((item for item in self.agents if item.name == name), None)

    @staticmethod
    def _agent_spec(agent: Reviewer, assignment_id: str) -> AgentSpec:
        provider = str(getattr(agent, "provider", ""))
        model = str(getattr(agent, "model", ""))
        prompt_hash = str(getattr(agent, "prompt_hash", ""))
        role = str(getattr(agent, "agent_role", "scanner"))
        group = artifact_fingerprint({
            "provider": provider, "model": model,
        })
        return AgentSpec(
            agent.name, role, provider, model, prompt_hash,
            ("read_file", "read_diff", "grep_repo", "find_symbol", "find_references")
            if getattr(agent, "execution_kind", "") == "agent" else (),
            assignment_id, group, 0, role == "auditor",
            model, provider, prompt_hash,
        )

    @staticmethod
    def _request_trace(execution: dict) -> tuple:
        raw = list((execution.get("usage") or {}).get("request_attempts") or
                   execution.get("request_attempts") or [])
        attempts = tuple(ModelRequestAttempt(**item) for item in raw if isinstance(item, dict))
        providers = tuple(dict.fromkeys(item.provider for item in attempts))
        models = tuple(dict.fromkeys(item.model for item in attempts))
        switches = sum(
            1 for previous, current in zip(attempts, attempts[1:])
            if (previous.provider, previous.model) != (current.provider, current.model)
        )
        return attempts, providers, models, switches

    @staticmethod
    def _cross_file_signature_candidates(state: CollaborationState) -> List[tuple]:
        """Derive auditable Python arity candidates from a pinned snapshot.

        This scanner handles only the closed, deterministic case: one added
        simple call and exactly one repository function definition whose
        positional arity is provably incompatible. It does not infer dynamic
        dispatch, overloads, decorators or runtime monkey-patching.
        """
        snapshot = state.get("snapshot_provider")
        if snapshot is None or not snapshot.available():
            return []
        definitions = {}
        for path in snapshot.paths():
            if not str(path).endswith(".py"):
                continue
            content = snapshot.read(path) or ""
            try:
                tree = ast.parse(content)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                positional = list(node.args.posonlyargs) + list(node.args.args)
                definitions.setdefault(node.name, []).append({
                    "path": path, "line": node.lineno,
                    "required": max(0, len(positional) - len(node.args.defaults)),
                    "maximum": None if node.args.vararg else len(positional),
                    "signature": ast.get_source_segment(content, node) or (
                        "def %s(%s)" % (node.name, ", ".join(item.arg for item in positional))
                    ),
                })
        values = []
        scanner_run_id = artifact_fingerprint({
            "task": state.get("task_id", ""), "scanner": "python-signature-v1",
            "snapshot": state.get("source_sha", ""),
        })[:24]
        for changed in state["parsed"].added_lines:
            if not changed.path.endswith(".py"):
                continue
            try:
                # Parse the added statement in function scope so `return
                # callee(...)` and other indented statements remain valid.
                expression = ast.parse("def __evoagent_probe__():\n    " + changed.content.strip())
            except SyntaxError:
                continue
            for node in ast.walk(expression):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                    continue
                matches = definitions.get(node.func.id, [])
                if len(matches) != 1 or any(isinstance(arg, ast.Starred) for arg in node.args):
                    continue
                definition = matches[0]
                count = len(node.args)
                maximum = definition["maximum"]
                if count >= definition["required"] and (maximum is None or count <= maximum):
                    continue
                locator = evidence_from_record({
                    "tool": "changed-line", "path": changed.path, "line": changed.line,
                    "excerpt": changed.content, "agent": "python-signature-scanner",
                }, state.get("source_sha", ""), scanner_run_id)
                structural = evidence_from_record({
                    "tool": "find_symbol", "path": definition["path"],
                    "line": definition["line"], "result": {
                        "symbol": node.func.id, "signature": definition["signature"],
                        "required": definition["required"], "maximum": maximum,
                    }, "agent": "python-signature-scanner",
                }, state.get("source_sha", ""), scanner_run_id)
                finding = Finding(
                    "COR-API-ARITY", Severity.HIGH, "Caller/callee argument mismatch",
                    "%s is called with %d positional arguments, but its unique pinned-snapshot "
                    "definition accepts %s." % (
                        node.func.id, count,
                        "%d..%s" % (definition["required"], maximum),
                    ),
                    changed.path, changed.line, changed.content,
                    "Align the call with the callee signature or update the API contract.",
                    "Exercise this call path and assert it does not raise TypeError.", .99,
                    [locator.evidence_id, structural.evidence_id],
                )
                values.append((finding, scanner_run_id, [locator, structural]))
        return values

    def _claim_node(self, state: CollaborationState) -> Dict[str, Any]:
        assignments = state.get("assignments_by_agent", {})
        all_outcomes = list(state.get("agent_outcomes", [])) + list(
            state.get("cross_shard_outcomes", [])
        )
        outcomes_by_id = {
            item.get("assignment_id"): item for item in all_outcomes
        }
        claim_findings = {}
        claims = []
        run_claims: Dict[str, List[str]] = {}
        evidence_by_claim: Dict[str, List[Any]] = {}
        source_queues = {
            key: list(values) for key, values in state.get("finding_sources", {}).items()
        }
        for finding in state.get("specialist_findings", []):
            source_ids = source_queues.get(finding_key(finding), [])
            source_id = source_ids.pop(0) if source_ids else "cross-shard"
            assignment = assignments.get(source_id)
            source_agent = self._agent_by_name(getattr(assignment, "agent", "")) if assignment else None
            source_kind = getattr(source_agent, "execution_kind", "agent") if source_agent else "agent"
            run_id = artifact_fingerprint({
                "task": state.get("task_id", ""), "assignment": source_id,
                "agent": getattr(source_agent, "name", source_id), "snapshot": state.get("source_sha", ""),
            })[:24]
            # Bind only ledger records explicitly linked to this Finding.  A
            # previous implementation attached every tool observation from an
            # AgentRun to every Claim emitted by that run, allowing an unrelated
            # read_file result to satisfy semantic evidence policy.
            records = [
                item for item in state["evidence_ledger"].compact()
                if item.get("evidence_id") in set(finding.evidence_refs)
            ]
            tool_evidence = [
                evidence_from_record(item, state.get("source_sha", ""), run_id)
                for item in records
            ]
            locator = evidence_from_record({
                "tool": "changed-line", "path": finding.path, "line": finding.line,
                "excerpt": finding.evidence, "agent": getattr(source_agent, "name", ""),
            }, state.get("source_sha", ""), run_id)
            finding.evidence_refs = sorted(
                item.evidence_id for item in tool_evidence + [locator]
            )
            claim = Claim.from_finding(finding, run_id, source_kind)
            claims.append(claim)
            claim_findings[claim.claim_id] = finding
            run_claims.setdefault(source_id, []).append(claim.claim_id)
            evidence_by_claim[claim.claim_id] = [
                replace(item, claim_ids=(claim.claim_id,))
                for item in tool_evidence + [locator]
            ]
        known = {(item.path, item.line, canonical_rule_id(item.rule_id)) for item in claims}
        for finding, run_id, evidence in self._cross_file_signature_candidates(state):
            identity = (finding.path, finding.line, canonical_rule_id(finding.rule_id))
            if identity in known:
                continue
            claim = Claim.from_finding(finding, run_id, "deterministic-candidate")
            bound = [replace(item, claim_ids=(claim.claim_id,)) for item in evidence]
            claims.append(claim)
            claim_findings[claim.claim_id] = finding
            evidence_by_claim[claim.claim_id] = bound
            known.add(identity)
        runs = []
        for outcome in all_outcomes:
            assignment = outcome["assignment"]
            agent = self._agent_by_name(outcome["agent"]) or self.fallback_agent
            if getattr(agent, "execution_kind", "deterministic-checker") != "agent":
                continue
            spec = self._agent_spec(agent, assignment.assignment_id)
            execution = outcome.get("execution") or {}
            usage = execution.get("usage") or {}
            request_trace = self._request_trace(execution)
            runs.append(AgentRun(
                artifact_fingerprint({
                    "task": state.get("task_id", ""), "assignment": assignment.assignment_id,
                    "agent": outcome["agent"], "snapshot": state.get("source_sha", ""),
                })[:24], spec, assignment.assignment_id, state.get("source_sha", ""),
                outcome["status"], tuple(run_claims.get(assignment.assignment_id, [])),
                int(execution.get("tool_calls", 0)), int(usage.get("llm_calls", 0)),
                int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0)),
                int(usage.get("cached_tokens", 0)), str(usage.get("usage_source", "estimated")),
                str(outcome.get("error", ""))[:500],
                *request_trace,
            ))
        return {
            "claims": claims, "claim_findings": claim_findings,
            "claim_evidence": evidence_by_claim, "agent_runs": runs,
        }

    def _auditor_node(self, state: CollaborationState) -> Dict[str, Any]:
        return self.auditor_stage.run(self, state)

    def _run_shard_audits(
        self, state: CollaborationState, stop_policy: AuditStopPolicy,
    ) -> Dict[str, Any]:
        """Reverse-audit each original shard in parallel, rounds serially."""
        if self.review_mode == "rules_only":
            return {"audit_outcomes": [], "audit_rounds": []}
        auditor = next((item for item in self.agents
                        if getattr(item, "agent_role", "") == "auditor"
                        and callable(getattr(item, "agent_step", None))), None)
        if auditor is None:
            return {"audit_outcomes": [], "audit_rounds": []}
        cumulative = [state["claim_findings"][claim.claim_id].to_dict()
                      for claim in state.get("claims", []) if claim.claim_id in state.get("claim_findings", {})]
        all_outcomes, round_summaries, dry_rounds = [], [], 0
        max_rounds = int(getattr(state.get("plan"), "audit_rounds", stop_policy.max_rounds))
        dry_limit = int(getattr(state.get("plan"), "dry_round_limit", stop_policy.consecutive_dry_rounds))
        for round_number in range(1, max_rounds + 1):
            deadline = float(state.get("deadline_monotonic", 0) or 0)
            if deadline and time.monotonic() >= deadline:
                state.setdefault("coverage_gaps", []).append({"reason": "auditor deadline gate", "files": []})
                break
            assignments = [ReviewAssignment(
                agent=auditor.name,
                objective="Find defects missed by prior review in this shard only; do not repeat confirmed findings.",
                files=list(shard.files), risk_domains=list(getattr(auditor, "domains", ())),
                assignment_id="audit-%s-r%d" % (shard.shard_id, round_number), round=round_number,
                reason="reverse-audit", shard_id=shard.shard_id, shard_files=list(shard.files),
                coverage_scope="shard", budget=self.budget_policy.for_shard(shard.changed_lines).to_dict(),
            ) for shard in state.get("shards", [])]
            if not assignments:
                break
            guidance = ["Confirmed findings (do not duplicate): %s" % json.dumps(cumulative, ensure_ascii=False)]
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(assignments))) as pool:
                futures = [pool.submit(self._run_assignment, state, assignment, auditor, guidance)
                           for assignment in assignments]
                outcomes = [future.result() for future in futures]
            all_outcomes.extend(outcomes)
            new_findings = []
            complete = True
            for outcome in outcomes:
                state.setdefault("assignments_by_agent", {})[outcome["assignment_id"]] = outcome["assignment"]
                receipt = outcome.get("coverage_receipt")
                if receipt:
                    state.setdefault("coverage_receipts", []).append(receipt)
                if outcome.get("status") not in {"completed", "fallback"}:
                    complete = False
                    state.setdefault("coverage_gaps", []).append({
                        "assignment_id": outcome["assignment_id"], "shard_id": outcome["assignment"].shard_id,
                        "files": list(outcome["assignment"].files), "reason": "reverse audit incomplete",
                    })
                    continue
                execution = outcome.get("execution") or {}
                state["evidence_ledger"].record(execution.get("pinned_evidence", []))
                for finding in outcome.get("findings", []):
                    key = finding_key(finding)
                    if any(finding_key(existing) == key for existing in state.get("specialist_findings", []) + new_findings):
                        continue
                    finding.evidence_refs = state["evidence_ledger"].link(key, finding.evidence_refs, finding.path, finding.line)
                    new_findings.append(finding)
                    state.setdefault("finding_sources", {}).setdefault(key, []).append(outcome["assignment_id"])
            state.setdefault("specialist_findings", []).extend(new_findings)
            cumulative.extend(item.to_dict() for item in new_findings)
            dry = complete and not new_findings
            dry_rounds = dry_rounds + 1 if dry else 0
            round_summaries.append({"round": round_number, "new_findings": len(new_findings),
                                    "dry": dry, "outcomes": len(outcomes)})
            if stop_policy.should_stop(round_number, dry_rounds) or dry_rounds >= dry_limit:
                break
        if round_summaries and len(round_summaries) >= max_rounds and dry_rounds < dry_limit:
            state.setdefault("coverage_gaps", []).append({"reason": "reverse audit hard round cap", "files": []})
        state.setdefault("agent_outcomes", []).extend(all_outcomes)
        return {"audit_outcomes": all_outcomes, "audit_rounds": round_summaries}

    def _legacy_adaptive_challenge_node(self, state: CollaborationState) -> Dict[str, Any]:
        if self.review_mode == "rules_only":
            return {"challenges": [], "challenge_outcomes": []}
        context = state.get("execution_context") or {}
        claims = list(state.get("claims", []))
        has_locator_only = any(
            claim.source_kind == "agent" and not any(
                item.kind in self.decision_policy.SEMANTIC_TOOLS
                for item in state.get("claim_evidence", {}).get(claim.claim_id, [])
            ) for claim in claims
        )
        challenge_required = bool(
            context.get("force_challenge")
            or any(claim.severity in {"high", "critical"} for claim in claims)
            or has_locator_only
            or (not claims and max((state.get("risk_scores") or {}).values(), default=0) >= 2)
        )
        if not challenge_required:
            return {"challenges": [], "challenge_outcomes": []}
        primary = next((item for item in self.agents
                        if getattr(item, "agent_role", "") == "primary"), None)
        auditor = (
            primary if self.challenge_strategy == "self_reflect" else
            next((item for item in self.agents
                  if getattr(item, "agent_role", "") == "auditor"), None)
        )
        if auditor is None:
            return {"challenges": [], "challenge_outcomes": []}
        if (self.challenge_strategy == "independent_auditor"
                and not state["shared_budget"].reserve("agent_runs", 1)):
            return {"challenges": [], "challenge_outcomes": [{
                "status": "budget_exhausted", "agent": auditor.name,
            }]}
        blind_claims = []
        for claim in sorted(state.get("claims", []), key=lambda item: item.claim_id):
            value = claim.to_dict()
            value.pop("proposer_run_id", None)
            blind_claims.append(value)
        assignment = ReviewAssignment(
            agent=auditor.name,
            objective="Blindly verify or refute candidate claims and find at most one missed high-risk defect.",
            files=list(state["parsed"].files), risk_domains=["security", "reliability", "correctness"],
            assignment_id="challenge-" + artifact_fingerprint({
                "task": state.get("task_id", ""), "claims": blind_claims,
                "strategy": self.challenge_strategy,
            })[:12], reason="blind-challenge", coverage_scope="full-pr",
        )
        guidance = [
            "Candidate claims (author identity and confidence intentionally removed): %s"
            % json.dumps(blind_claims, ensure_ascii=False, sort_keys=True)
        ]
        prior_execution = None
        if self.challenge_strategy == "self_reflect":
            prior_execution = next((
                item.get("execution") for item in state.get("agent_outcomes", [])
                if item.get("agent") == auditor.name
            ), None)
        findings, attempts, error, execution = self._invoke_agent(
            state, auditor, assignment, guidance, prior_execution,
        )
        challenge_run_id = artifact_fingerprint({
            "task": state.get("task_id", ""), "assignment": assignment.assignment_id,
            "agent": auditor.name, "snapshot": state.get("source_sha", ""),
        })[:24]
        if self.challenge_strategy == "self_reflect":
            primary_run = next((item for item in state.get("agent_runs", [])
                                if item.spec.role == "primary"), None)
            if primary_run is not None:
                challenge_run_id = primary_run.run_id
        raw_evidence = list(execution.get("pinned_evidence") or [])
        for observation in execution.get("observations") or []:
            if not observation.get("ok"):
                continue
            result = observation.get("result")
            locator = observation.get("locator") or {}
            record = {
                "tool": observation.get("tool", ""),
                "path": locator.get("path", ""),
                "line": locator.get("line", locator.get("start_line", 0)),
                "result": result, "agent": auditor.name, "shard_id": "challenge",
            }
            raw_evidence.append(record)
        evidence = [evidence_from_record(
            item, state.get("source_sha", ""), challenge_run_id,
        ) for item in raw_evidence]
        by_location = {(item.path, item.line, canonical_rule_id(item.rule_id)): item for item in findings}
        challenges = []
        for claim in state.get("claims", []):
            supported = by_location.get((claim.path, claim.line, canonical_rule_id(claim.rule_id)))
            existing_refs = tuple(item.evidence_id for item in state.get("claim_evidence", {}).get(claim.claim_id, []))
            refs = tuple(item.evidence_id for item in evidence) or (
                existing_refs if self.challenge_strategy == "self_reflect" else ()
            )
            verdict = "support" if supported and refs else "insufficient"
            rationale = (
                "Independent tool evidence supports the claim."
                if verdict == "support" else
                "The challenge pass did not produce independent semantic evidence for this claim."
            )
            challenges.append(Challenge(claim.claim_id, verdict, rationale, challenge_run_id, refs))
            state.setdefault("claim_evidence", {}).setdefault(claim.claim_id, []).extend(
                item for item in evidence if item.evidence_id in refs
            )
        known = {(item.path, item.line, canonical_rule_id(item.rule_id)) for item in state.get("claim_findings", {}).values()}
        new_findings = [item for item in findings
                        if (item.path, item.line, canonical_rule_id(item.rule_id)) not in known][:1]
        for finding in new_findings:
            claim = Claim.from_finding(finding, challenge_run_id, "agent")
            state["claims"].append(claim)
            state["claim_findings"][claim.claim_id] = finding
            state.setdefault("claim_evidence", {})[claim.claim_id] = list(evidence)
            challenges.append(Challenge(
                claim.claim_id, "new_claim", "Auditor found a previously omitted claim.",
                challenge_run_id, tuple(item.evidence_id for item in evidence),
            ))
        revision = {}
        revision_run = None
        if (self.challenge_strategy == "independent_auditor" and primary
                and any(item.verdict in {"refute", "insufficient", "new_claim"} for item in challenges)):
            revision_assignment = ReviewAssignment(
                agent=primary.name,
                objective="Revise all challenged claims in one batch; discard unsupported hypotheses.",
                files=list(state["parsed"].files), risk_domains=["correctness", "security", "reliability"],
                assignment_id="revision-" + assignment.assignment_id,
                reason="challenge-requested-revision", coverage_scope="full-pr",
            )
            revision_guidance = [
                "Challenges to address in one batch: %s" % json.dumps(
                    [item.to_dict() for item in challenges], ensure_ascii=False, sort_keys=True,
                )
            ]
            revised, revision_attempts, revision_error, revision_execution = self._invoke_agent(
                state, primary, revision_assignment, revision_guidance,
            )
            revised_by_cluster = {
                (item.path, item.line, canonical_rule_id(item.rule_id)): item for item in revised
            }
            for claim in state.get("claims", []):
                replacement = revised_by_cluster.get((claim.path, claim.line, canonical_rule_id(claim.rule_id)))
                if replacement is not None:
                    state["claim_findings"][claim.claim_id] = replacement
            revision = {
                "agent": primary.name, "status": "completed" if not revision_error else "failed",
                "attempts": revision_attempts, "error": revision_error,
                "execution": revision_execution, "batched": True,
            }
            revision_usage = revision_execution.get("usage") or {}
            revision_run_id = artifact_fingerprint({
                "task": state.get("task_id", ""), "assignment": revision_assignment.assignment_id,
                "agent": primary.name, "snapshot": state.get("source_sha", ""),
            })[:24]
            revision_run = AgentRun(
                revision_run_id, self._agent_spec(primary, revision_assignment.assignment_id),
                revision_assignment.assignment_id, state.get("source_sha", ""),
                revision["status"], (), int(revision_execution.get("tool_calls", 0)),
                int(revision_usage.get("llm_calls", 0)), int(revision_usage.get("input_tokens", 0)),
                int(revision_usage.get("output_tokens", 0)), int(revision_usage.get("cached_tokens", 0)),
                str(revision_usage.get("usage_source", "estimated")), revision_error[:500],
                *self._request_trace(revision_execution),
            )
        usage = execution.get("usage") or {}
        challenge_run = AgentRun(
            challenge_run_id, self._agent_spec(auditor, assignment.assignment_id),
            assignment.assignment_id, state.get("source_sha", ""),
            "completed" if not error else "failed",
            tuple(item.claim_id for item in challenges), int(execution.get("tool_calls", 0)),
            int(usage.get("llm_calls", 0)), int(usage.get("input_tokens", 0)),
            int(usage.get("output_tokens", 0)), int(usage.get("cached_tokens", 0)),
            str(usage.get("usage_source", "estimated")), error[:500],
            *self._request_trace(execution),
        )
        runs = list(state.get("agent_runs", []))
        if self.challenge_strategy == "independent_auditor":
            runs.append(challenge_run)
        else:
            for index, run in enumerate(runs):
                if run.spec.role == "primary":
                    runs[index] = replace(
                        run,
                        tool_calls=run.tool_calls + challenge_run.tool_calls,
                        llm_calls=run.llm_calls + challenge_run.llm_calls,
                        input_tokens=run.input_tokens + challenge_run.input_tokens,
                        output_tokens=run.output_tokens + challenge_run.output_tokens,
                        cached_tokens=run.cached_tokens + challenge_run.cached_tokens,
                        usage_source=("provider" if run.usage_source == challenge_run.usage_source == "provider" else "estimated"),
                        error=challenge_run.error or run.error,
                        request_attempts=run.request_attempts + challenge_run.request_attempts,
                        effective_providers=tuple(dict.fromkeys(run.effective_providers + challenge_run.effective_providers)),
                        effective_models=tuple(dict.fromkeys(run.effective_models + challenge_run.effective_models)),
                        model_switch_count=run.model_switch_count + challenge_run.model_switch_count,
                    )
                    break
        if revision_run is not None:
            # Directed revision belongs to the original proposer AgentRun; it
            # is a new phase, not an independent participant/context vote.
            for index, run in enumerate(runs):
                if run.spec.role == "primary":
                    runs[index] = replace(
                        run,
                        status="failed" if revision_run.status == "failed" else run.status,
                        tool_calls=run.tool_calls + revision_run.tool_calls,
                        llm_calls=run.llm_calls + revision_run.llm_calls,
                        input_tokens=run.input_tokens + revision_run.input_tokens,
                        output_tokens=run.output_tokens + revision_run.output_tokens,
                        cached_tokens=run.cached_tokens + revision_run.cached_tokens,
                        usage_source=("provider" if run.usage_source == revision_run.usage_source == "provider" else "estimated"),
                        error=revision_run.error or run.error,
                        request_attempts=run.request_attempts + revision_run.request_attempts,
                        effective_providers=tuple(dict.fromkeys(run.effective_providers + revision_run.effective_providers)),
                        effective_models=tuple(dict.fromkeys(run.effective_models + revision_run.effective_models)),
                        model_switch_count=run.model_switch_count + revision_run.model_switch_count,
                    )
                    break
        outcomes = [{
            "agent": auditor.name, "status": "completed" if not error else "failed",
            "attempts": attempts, "error": error, "execution": execution,
            "blind_context": True,
        }]
        if revision:
            outcomes.append(revision)
        return {
            "challenges": challenges,
            "challenge_outcomes": outcomes,
            "agent_runs": runs,
        }

    def _challenge_node(self, state: CollaborationState) -> Dict[str, Any]:
        # Verification is deliberately a fan-out: each worker receives at
        # most eight normalized findings and independently re-collects proof.
        # The older monolithic challenge implementation remains below for
        # compatibility with legacy callers, but adaptive graph execution uses
        # this bounded verifier path.
        if self.review_pipeline == "bounded-v2" and self.review_mode == "adaptive_multi_agent":
            return self.verifier_stage.run(self, state)
        if self.review_mode == "rules_only":
            return {"challenges": [], "challenge_outcomes": [],
                    "challenge_activation": {"triggered": False, "reasons": ["rules-only"]}}
        context = state.get("execution_context") or {}
        claims = list(state.get("claims", []))
        risk_score = max((state.get("risk_scores") or {}).values(), default=0)
        admitted = []
        for claim in claims:
            finding = state.get("claim_findings", {}).get(claim.claim_id)
            if (claim.source_kind == "agent" and finding is not None
                    and not self.decision_policy.claim_admission_reasons(
                        claim, finding, state["parsed"]
                    )):
                admitted.append(claim)
        relevant_semantic = {
            claim.claim_id: any(
                item.kind in self.decision_policy.SEMANTIC_TOOLS
                and item.result_nonempty
                and self.decision_policy.evidence_relevant(claim, item)
                for item in state.get("claim_evidence", {}).get(claim.claim_id, [])
            ) for claim in admitted
        }
        cross_file_claims = [
            claim for claim in admitted if (
                canonical_rule_id(claim.rule_id) == "CWE-628"
                or len(state["parsed"].files) > 1
                or any(
                    item.path and item.path != claim.path
                    and self.decision_policy.evidence_relevant(claim, item)
                    for item in state.get("claim_evidence", {}).get(claim.claim_id, [])
                )
            )
        ]
        by_location = {}
        for claim in admitted:
            by_location.setdefault((claim.path, claim.line), set()).add(
                canonical_rule_id(claim.rule_id)
            )
        divergent = any(len(values) > 1 for values in by_location.values())
        reasons = []
        if context.get("force_challenge"):
            reasons.append("forced-experiment")
        else:
            if risk_score >= 1 and any(
                claim.severity in {"high", "critical"} for claim in admitted
            ):
                reasons.append("admitted-high-risk")
            if cross_file_claims:
                reasons.append("cross-file-contract")
            if risk_score >= 2 and any(
                not relevant_semantic[claim.claim_id] for claim in admitted
            ):
                reasons.append("claim-evidence-incomplete")
            if divergent:
                reasons.append("agent-claim-divergence")
            if not admitted and not claims and risk_score >= 2:
                reasons.append("risk-activated-clean-review")
        activation = {
            "triggered": bool(reasons), "reasons": reasons,
            "risk_score": risk_score,
            "eligible_claim_ids": [claim.claim_id for claim in admitted],
        }
        if not reasons:
            return {"challenges": [], "challenge_outcomes": [],
                    "challenge_activation": activation}

        primary = next((item for item in self.agents
                        if getattr(item, "agent_role", "") == "primary"), None)
        auditor = (
            primary if self.challenge_strategy == "self_reflect" else
            next((item for item in self.agents
                  if getattr(item, "agent_role", "") == "auditor"), None)
        )
        if auditor is None:
            activation["reasons"].append("auditor-unavailable")
            return {"challenges": [], "challenge_outcomes": [],
                    "challenge_activation": activation}
        if (self.challenge_strategy == "independent_auditor"
                and not state["shared_budget"].reserve("agent_runs", 1)):
            return {"challenges": [], "challenge_outcomes": [{
                "status": "budget_exhausted", "agent": auditor.name,
            }], "challenge_activation": activation}

        # For the narrow, structurally-adapted CWE families, multiple agents
        # can independently describe the same sink at the same added line.
        # Keep every immutable Claim for audit, but challenge one representative
        # and propagate the evidence verdict to its semantic aliases.  Unknown
        # families never reach this point, so distinct unmodelled hypotheses
        # are not collapsed by location alone.
        challenge_groups = {}
        for claim in admitted:
            challenge_groups.setdefault(
                (claim.path, claim.line, canonical_rule_id(claim.rule_id)), []
            ).append(claim)
        challenge_claims = [values[0] for values in challenge_groups.values()]
        aliases_by_representative = {
            values[0].claim_id: values for values in challenge_groups.values()
        }
        blind_claims = []
        for claim in challenge_claims:
            value = claim.to_dict()
            value.pop("proposer_run_id", None)
            blind_claims.append(value)
        assignment = ReviewAssignment(
            agent=auditor.name,
            objective="Return an explicit support, refute or insufficient verdict for every claim.",
            files=list(state["parsed"].files),
            risk_domains=["security", "reliability", "correctness"],
            assignment_id="challenge-" + artifact_fingerprint({
                "task": state.get("task_id", ""), "claims": blind_claims,
                "strategy": self.challenge_strategy,
            })[:12], reason="blind-challenge", coverage_scope="full-pr",
        )
        guidance = [
            "Candidate claims (identity and confidence removed): %s" %
            json.dumps(blind_claims, ensure_ascii=False, sort_keys=True),
            "Every verdict must cite only evidence IDs actually available in this run.",
        ]
        if not blind_claims:
            guidance.append(
                "Primary returned no claims despite deterministic risk activation. Independently inspect the "
                "highest-risk changed behavior and return new_claim with tool evidence when a high-risk defect "
                "was missed; otherwise return an empty challenges list after recording counterevidence. "
                "For a changed function call, use find_symbol on the callee to verify its signature."
            )
        prior_execution = None
        if self.challenge_strategy == "self_reflect":
            prior_execution = next((
                item.get("execution") for item in state.get("agent_outcomes", [])
                if item.get("agent") == auditor.name
            ), None)
        outputs, attempts, error, execution = self._invoke_agent(
            state, auditor, assignment, guidance, prior_execution,
        )
        challenge_run_id = artifact_fingerprint({
            "task": state.get("task_id", ""), "assignment": assignment.assignment_id,
            "agent": auditor.name, "snapshot": state.get("source_sha", ""),
        })[:24]
        if self.challenge_strategy == "self_reflect":
            original = next((item for item in state.get("agent_runs", [])
                             if item.spec.role == "primary"), None)
            if original is not None:
                challenge_run_id = original.run_id

        raw_records = list(execution.get("pinned_evidence") or [])
        for observation in execution.get("observations") or []:
            if not observation.get("ok"):
                continue
            locator = observation.get("locator") or {}
            record = {
                "id": "R%d" % int(observation.get("step", 0) or 0),
                "tool": observation.get("tool", ""),
                "path": locator.get("path", ""),
                "line": locator.get("line", locator.get("start_line", 0)),
                "result": observation.get("result"), "agent": auditor.name,
                "shard_id": "challenge",
                "snapshot_id": observation.get("snapshot_id", ""),
                "producer_run_id": observation.get("producer_run_id", ""),
            }
            locations = list(locator.get("locations") or [])
            if not record["path"] and len(locations) == 1:
                record["path"] = str(locations[0].get("path", ""))
                record["line"] = int(locations[0].get("line", 0) or 0)
            raw_records.append(record)
        challenge_evidence = {}
        evidence_aliases = {}
        for record in raw_records:
            item = evidence_from_record(
                record, state.get("source_sha", ""), challenge_run_id
            )
            if not item.result_nonempty:
                continue
            challenge_evidence[item.evidence_id] = item
            for alias in (record.get("id"), record.get("content_id")):
                if alias:
                    evidence_aliases[str(alias)] = item.evidence_id
        response_by_claim = {
            str(item.get("claim_id", "")): item for item in outputs
            if isinstance(item, dict) and item.get("claim_id")
        }
        challenges = []
        for claim in challenge_claims:
            response = response_by_claim.get(claim.claim_id) or {}
            verdict = str(response.get("verdict", "insufficient"))
            if verdict not in {"support", "refute", "insufficient"}:
                verdict = "insufficient"
            existing = {
                item.evidence_id: item
                for item in state.get("claim_evidence", {}).get(claim.claim_id, [])
                if item.result_nonempty
            }
            available = dict(challenge_evidence)
            if self.challenge_strategy == "self_reflect":
                available.update(existing)
            refs = tuple(dict.fromkeys(
                evidence_aliases.get(str(ref), str(ref))
                for ref in list(response.get("evidence_refs") or [])
                if evidence_aliases.get(str(ref), str(ref)) in available
                and self.decision_policy.evidence_relevant(
                    claim, available[evidence_aliases.get(str(ref), str(ref))]
                )
            ))
            if verdict in {"support", "refute"} and not refs:
                verdict = "insufficient"
            bound = [replace(available[ref], claim_ids=(claim.claim_id,),
                             direction="refute" if verdict == "refute" else "support")
                     for ref in refs]
            rationale = (
                str(response.get("rationale", ""))[:1000] or
                ("Explicit evidence challenge completed." if refs else
                 "No valid claim-bound evidence was returned.")
            )
            for alias in aliases_by_representative.get(claim.claim_id, [claim]):
                alias_bound = [replace(
                    item, claim_ids=(alias.claim_id,),
                    direction="refute" if verdict == "refute" else "support",
                ) for item in bound if self.decision_policy.evidence_relevant(alias, item)]
                alias_refs = tuple(item.evidence_id for item in alias_bound)
                alias_verdict = verdict
                if verdict in {"support", "refute"} and not alias_refs:
                    alias_verdict = "insufficient"
                state.setdefault("claim_evidence", {}).setdefault(
                    alias.claim_id, []
                ).extend(alias_bound)
                challenges.append(Challenge(
                    alias.claim_id, alias_verdict, rationale,
                    challenge_run_id, alias_refs,
                    str(response.get("counter_hypothesis", ""))[:1000],
                    str(response.get("counterexample", ""))[:1000],
                ))

        for response in outputs:
            if not isinstance(response, dict) or response.get("verdict") != "new_claim":
                continue
            finding = response.get("finding")
            if not isinstance(finding, Finding):
                continue
            claim = Claim.from_finding(finding, challenge_run_id, "agent")
            refs = tuple(
                evidence_aliases.get(str(ref), str(ref))
                for ref in response.get("evidence_refs", [])
                if evidence_aliases.get(str(ref), str(ref)) in challenge_evidence
                and self.decision_policy.evidence_relevant(
                    claim, challenge_evidence[evidence_aliases.get(str(ref), str(ref))]
                )
            )
            bound_new_evidence = [
                replace(challenge_evidence[ref], claim_ids=(claim.claim_id,)) for ref in refs
            ]
            admission = self.decision_policy.claim_admission_reasons(
                claim, finding, state["parsed"]
            )
            semantic = any(
                item.kind in self.decision_policy.SEMANTIC_TOOLS
                and item.result_nonempty for item in bound_new_evidence
            )
            in_risk_scope = risk_score >= 2 or len(state["parsed"].files) > 1
            if admission or not semantic or not in_risk_scope:
                reasons = list(admission)
                if not semantic:
                    reasons.append("claim-relevant-evidence-required")
                if not in_risk_scope:
                    reasons.append("outside-activated-risk-scope")
                challenges.append(Challenge(
                    claim.claim_id, "insufficient", ";".join(reasons),
                    challenge_run_id, refs,
                ))
                continue
            state["claims"].append(claim)
            state["claim_findings"][claim.claim_id] = finding
            state.setdefault("claim_evidence", {})[claim.claim_id] = bound_new_evidence
            challenges.append(Challenge(
                claim.claim_id, "new_claim", str(response.get("rationale", ""))[:1000],
                challenge_run_id, refs,
            ))

        usage = execution.get("usage") or {}
        challenge_run = AgentRun(
            challenge_run_id, self._agent_spec(auditor, assignment.assignment_id),
            assignment.assignment_id, state.get("source_sha", ""),
            "completed" if not error else "failed", tuple(item.claim_id for item in challenges),
            int(execution.get("tool_calls", 0)), int(usage.get("llm_calls", 0)),
            int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0)),
            int(usage.get("cached_tokens", 0)), str(usage.get("usage_source", "estimated")),
            error[:500], *self._request_trace(execution),
        )
        runs = list(state.get("agent_runs", []))

        def merge_phase(target_id: str, phase: AgentRun) -> None:
            for index, run in enumerate(runs):
                if run.run_id != target_id:
                    continue
                combined_attempts = run.request_attempts + phase.request_attempts
                providers = tuple(dict.fromkeys(item.provider for item in combined_attempts))
                models = tuple(dict.fromkeys(item.model for item in combined_attempts))
                switches = sum(
                    1 for previous, current in zip(combined_attempts, combined_attempts[1:])
                    if (previous.provider, previous.model) != (current.provider, current.model)
                )
                runs[index] = replace(
                    run, status="failed" if phase.status == "failed" else run.status,
                    tool_calls=run.tool_calls + phase.tool_calls,
                    llm_calls=run.llm_calls + phase.llm_calls,
                    input_tokens=run.input_tokens + phase.input_tokens,
                    output_tokens=run.output_tokens + phase.output_tokens,
                    cached_tokens=run.cached_tokens + phase.cached_tokens,
                    usage_source=("provider" if run.usage_source == phase.usage_source == "provider"
                                  else "estimated"),
                    error=phase.error or run.error, request_attempts=combined_attempts,
                    effective_providers=providers, effective_models=models,
                    model_switch_count=switches,
                )
                return

        if self.challenge_strategy == "independent_auditor":
            runs.append(challenge_run)
        else:
            merge_phase(challenge_run_id, challenge_run)

        revision_outcomes = []
        latest_claims = list(state.get("claims", []))
        challenges_by_claim = {item.claim_id: item for item in challenges}
        proposer_groups: Dict[str, List[Claim]] = {}
        for claim in latest_claims:
            if (claim.source_kind == "agent" and claim.claim_id in challenges_by_claim
                    and challenges_by_claim[claim.claim_id].verdict in {"refute", "insufficient"}):
                proposer_groups.setdefault(claim.proposer_run_id, []).append(claim)
        replacement_ids = {}
        withdrawn = set()
        for proposer_run_id, proposed_claims in proposer_groups.items():
            proposer_run = next((item for item in runs if item.run_id == proposer_run_id), None)
            proposer = self._agent_by_name(proposer_run.spec.agent_id) if proposer_run else None
            if proposer is None:
                continue
            revision_assignment = ReviewAssignment(
                agent=proposer.name,
                objective="Return retain, revise or withdraw for every challenged claim.",
                files=sorted({claim.path for claim in proposed_claims}),
                risk_domains=list(getattr(proposer, "domains", ())),
                assignment_id="revision-" + artifact_fingerprint({
                    "challenge": assignment.assignment_id, "proposer": proposer_run_id,
                })[:12], reason="challenge-requested-revision", coverage_scope="targeted",
            )
            revision_guidance = ["Claims and challenges: %s" % json.dumps([
                {"claim": claim.to_dict(),
                 "challenge": challenges_by_claim[claim.claim_id].to_dict()}
                for claim in proposed_claims
            ], ensure_ascii=False, sort_keys=True)]
            revisions, revision_attempts, revision_error, revision_execution = self._invoke_agent(
                state, proposer, revision_assignment, revision_guidance,
            )
            response_map = {
                str(item.get("claim_id", "")): item for item in revisions
                if isinstance(item, dict) and item.get("claim_id")
            }
            for claim in proposed_claims:
                response = response_map.get(claim.claim_id)
                if revision_error:
                    continue
                if response is None or response.get("action") == "withdraw":
                    withdrawn.add(claim.claim_id)
                    continue
                if response.get("action") != "revise" or not isinstance(
                    response.get("revised_finding"), Finding
                ):
                    continue
                revised_finding = response["revised_finding"]
                revised_claim = replace(
                    Claim.from_finding(revised_finding, claim.proposer_run_id, claim.source_kind),
                    version=claim.version + 1, supersedes_claim_id=claim.claim_id,
                )
                replacement_ids[claim.claim_id] = revised_claim.claim_id
                state["claim_findings"][revised_claim.claim_id] = revised_finding
                state["claim_evidence"][revised_claim.claim_id] = [
                    replace(item, claim_ids=(revised_claim.claim_id,))
                    for item in state["claim_evidence"].get(claim.claim_id, [])
                ]
                latest_claims.append(revised_claim)
            revision_usage = revision_execution.get("usage") or {}
            phase = AgentRun(
                proposer_run_id, proposer_run.spec, revision_assignment.assignment_id,
                state.get("source_sha", ""), "completed" if not revision_error else "failed", (),
                int(revision_execution.get("tool_calls", 0)),
                int(revision_usage.get("llm_calls", 0)), int(revision_usage.get("input_tokens", 0)),
                int(revision_usage.get("output_tokens", 0)), int(revision_usage.get("cached_tokens", 0)),
                str(revision_usage.get("usage_source", "estimated")), revision_error[:500],
                *self._request_trace(revision_execution),
            )
            merge_phase(proposer_run_id, phase)
            revision_outcomes.append({
                "agent": proposer.name, "proposer_run_id": proposer_run_id,
                "status": "completed" if not revision_error else "failed",
                "attempts": revision_attempts, "error": revision_error,
                "execution": revision_execution, "batched": True,
            })

        state["claims"] = [
            claim for claim in latest_claims
            if claim.claim_id not in withdrawn and claim.claim_id not in replacement_ids
        ]
        state["challenges"] = [
            replace(item, claim_id=replacement_ids.get(item.claim_id, item.claim_id))
            for item in challenges
        ]
        outcomes = [{
            "agent": auditor.name, "status": "completed" if not error else "failed",
            "attempts": attempts, "error": error, "execution": execution,
            "blind_context": self.challenge_strategy == "independent_auditor",
        }] + revision_outcomes
        return {"challenges": state["challenges"], "challenge_outcomes": outcomes,
                "agent_runs": runs, "claims": state["claims"],
                "challenge_activation": activation}

    def _run_batched_verifier(self, state: CollaborationState, batch_builder) -> Dict[str, Any]:
        verifier = next((item for item in self.agents
                        if getattr(item, "agent_role", "") == "verifier"
                        and callable(getattr(item, "agent_step", None))), None)
        if verifier is None:
            return {"challenges": [], "challenge_outcomes": [], "verifier_batches": []}
        claims = list(state.get("claims", []))
        # Normalize display aliases before batching, keeping their claims so
        # deterministic decision provenance is never discarded.
        representatives = {}
        for claim in claims:
            representatives.setdefault((claim.path, claim.line, canonical_rule_id(claim.rule_id)), []).append(claim)
        groups = list(representatives.values())
        batches = batch_builder(groups)
        if not batches:
            return {"challenges": [], "challenge_outcomes": [], "verifier_batches": []}

        def run_batch(index: int, groups_in_batch: List[List[Claim]]) -> dict:
            batch_claims = [group[0] for group in groups_in_batch]
            public = []
            for claim in batch_claims:
                value = claim.to_dict()
                value.pop("proposer_run_id", None)
                public.append(value)
            assignment = ReviewAssignment(
                agent=verifier.name, objective="Independently verify each candidate failure scenario.",
                files=sorted({claim.path for claim in batch_claims}),
                risk_domains=["security", "reliability", "correctness"],
                assignment_id="verifier-%02d" % (index + 1), reason="verify-findings",
                coverage_scope="verification",
                budget=self.budget_policy.for_shard(sum(
                    1 for line in state["parsed"].added_lines if line.path in {claim.path for claim in batch_claims}
                )).to_dict(),
            )
            outcome = self._run_assignment(state, assignment, verifier, [
                "Candidate claims (max 8): %s" % json.dumps(public, ensure_ascii=False, sort_keys=True),
                "Return support, refute, or insufficient with fresh tool evidence for every claim.",
            ])
            outcome["claim_groups"] = groups_in_batch
            return outcome

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(batches))) as pool:
            futures = [pool.submit(run_batch, batch.index - 1, batch.claims) for batch in batches]
            outcomes = [future.result() for future in futures]
        challenges, runs = [], list(state.get("agent_runs", []))
        for outcome in outcomes:
            execution = outcome.get("execution") or {}
            receipt = outcome.get("coverage_receipt")
            if receipt:
                state.setdefault("coverage_receipts", []).append(receipt)
            if outcome.get("status") not in {"completed", "fallback"}:
                for group in outcome["claim_groups"]:
                    for claim in group:
                        challenges.append(Challenge(claim.claim_id, "insufficient", "verifier budget or execution gap", "", ()))
                state.setdefault("coverage_gaps", []).append({
                    "assignment_id": outcome["assignment_id"], "files": list(outcome["assignment"].files),
                    "reason": "verifier incomplete",
                })
                continue
            raw = list(execution.get("pinned_evidence") or [])
            raw.extend({"tool": item.get("tool", ""), "path": (item.get("locator") or {}).get("path", ""),
                        "line": (item.get("locator") or {}).get("line", 0), "result": item.get("result", "")}
                       for item in execution.get("observations", []) if item.get("ok"))
            run_id = artifact_fingerprint({"task": state.get("task_id", ""), "assignment": outcome["assignment_id"],
                                           "snapshot": state.get("source_sha", "")})[:24]
            evidence = [evidence_from_record(item, state.get("source_sha", ""), run_id) for item in raw]
            by_claim = {str(item.get("claim_id", "")): item for item in outcome.get("findings", []) if isinstance(item, dict)}
            for group in outcome["claim_groups"]:
                representative = group[0]
                response = by_claim.get(representative.claim_id, {})
                verdict = str(response.get("verdict", "insufficient")).lower()
                if verdict not in {"support", "refute", "insufficient"}:
                    verdict = "insufficient"
                refs = tuple(item.evidence_id for item in evidence)
                for claim in group:
                    challenges.append(Challenge(claim.claim_id, verdict,
                                                str(response.get("rationale", "fresh verifier evidence"))[:500],
                                                run_id, refs))
                    state.setdefault("claim_evidence", {}).setdefault(claim.claim_id, []).extend(
                        [replace(item, claim_ids=(claim.claim_id,)) for item in evidence]
                    )
            usage = execution.get("usage") or {}
            runs.append(AgentRun(run_id, self._agent_spec(verifier, outcome["assignment_id"]),
                                 outcome["assignment_id"], state.get("source_sha", ""), "completed",
                                 tuple(claim.claim_id for group in outcome["claim_groups"] for claim in group),
                                 int(execution.get("tool_calls", 0)), int(usage.get("llm_calls", 0)),
                                 int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0)),
                                 int(usage.get("cached_tokens", 0)), str(usage.get("usage_source", "estimated")), "",
                                 *self._request_trace(execution)))
        return {"challenges": challenges, "challenge_outcomes": outcomes, "agent_runs": runs,
                "verifier_batches": [{"size": sum(len(group) for group in batch.claims)} for batch in batches],
                "challenge_activation": {"triggered": True, "reasons": ["batched-verifier"]}}

    def _adaptive_decision_node(self, state: CollaborationState) -> Dict[str, Any]:
        decisions = []
        accepted = []
        escalations = []
        rejected = []
        for claim in sorted(state.get("claims", []), key=lambda item: item.claim_id):
            finding = state["claim_findings"][claim.claim_id]
            decision = self.decision_policy.decide(
                claim, finding, state["parsed"],
                state.get("claim_evidence", {}).get(claim.claim_id, []),
                state.get("challenges", []),
                state.get("source_sha", ""),
            )
            decisions.append(decision)
            if decision.outcome == "accept":
                accepted.append((claim, finding))
            elif decision.outcome == "escalate":
                escalations.append({
                    "claim": claim.to_dict(), "decision": decision.to_dict(),
                })
            else:
                rejected.append({
                    "claim_id": claim.claim_id, "reason_codes": list(decision.reason_codes),
                })
        if (max((state.get("risk_scores") or {}).values(), default=0) >= 2
                and (state.get("llm_failures") or not any(
                    run.spec.role == "primary" and run.status in {"completed", "fallback"}
                    for run in state.get("agent_runs", [])
                ))):
            escalations.append({
                "kind": "operational-risk-escalation",
                "risk_scores": dict(state.get("risk_scores") or {}),
                "reason_codes": ["semantic-review-incomplete"],
                "llm_failures": list(state.get("llm_failures") or []),
            })
        coverage_incomplete = bool(state.get("coverage_gaps"))
        if coverage_incomplete:
            escalations.append({
                "kind": "coverage-incomplete",
                "reason_codes": ["required-hunk-coverage-incomplete"],
                "coverage_gaps": list(state.get("coverage_gaps") or []),
                "verdict_cap": "needs-human-review",
            })
        # Keep every immutable Claim and Decision above, but collapse aliases
        # at the user-facing Finding boundary.  A semantic Agent may call the
        # same defect CWE-703 while the rule scanner calls it
        # REL-EMPTY-EXCEPT; emitting both is duplicate noise, not independent
        # support.  Location + canonical rule is deliberately only a display
        # key and never an identity key for claims or evidence.
        merged = {}
        source_rank = {"deterministic-checker": 2, "deterministic-candidate": 1, "agent": 0}
        for claim, finding in accepted:
            display_key = (finding.path, finding.line, canonical_rule_id(finding.rule_id))
            current = merged.get(display_key)
            candidate_rank = (
                source_rank.get(claim.source_kind, 0), finding.confidence,
                len(finding.explanation), finding.rule_id,
            )
            if current is None or candidate_rank > current[0]:
                merged[display_key] = (candidate_rank, finding)
        order = {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2, Severity.LOW: 3}
        verified = sorted(
            (value[1] for value in merged.values()),
            key=lambda item: (order[item.severity], item.path, item.line),
        )
        self._emit(state, "adaptive-decision-policy", "review-report", "adaptive_decision", {
            "approved_findings": [item.to_dict() for item in verified],
            "rejected_findings": rejected, "escalations": escalations,
        })
        self._emit(state, self.arbiter.name, "review-report", "arbitration_decision", {
            "approved_findings": [item.to_dict() for item in verified],
            "rejected_findings": rejected, "escalations": escalations,
            "compatibility_event": True,
        })
        return {
            "verified": verified, "adaptive_decisions": decisions,
            "escalations": escalations, "adaptive_rejected": rejected,
            "coverage_gate": "needs-human-review" if coverage_incomplete else "clear",
        }

    def _deliberation_node(self, state: CollaborationState) -> Dict[str, Any]:
        findings = list(state["specialist_findings"])
        critiques: Dict[str, Critique] = {}
        rounds_completed = 0
        agents = {item.name: item for item in self.agents}
        agents.setdefault(self.fallback_agent.name, self.fallback_agent)
        assignments = state["assignments_by_agent"]
        for round_number in range(1, self.collaboration_rounds + 1):
            rounds_completed = round_number
            revisions = []
            for finding in findings:
                key = finding_key(finding)
                critique = self.critic.challenge(
                    finding, state["parsed"], state["finding_sources"].get(key, []),
                    round_number,
                )
                critiques[key] = critique
                reflection = self.reflection_agent.reflect(critique)
                self._emit(
                    state, self.critic.name, self.reflection_agent.name,
                    "critique_for_reflection", asdict(critique), key,
                )
                recipients = state["finding_sources"].get(key, [])
                for recipient in recipients:
                    recipient_assignment = assignments.get(recipient)
                    recipient_agent = recipient_assignment.agent if recipient_assignment else "specialists"
                    self._emit(
                        state, self.critic.name, recipient_agent, "peer_challenge",
                        asdict(critique), key,
                    )
                    self._emit(
                        state, self.reflection_agent.name, recipient_agent,
                        "reflection_guidance", asdict(reflection), key,
                    )
                if reflection.revision_needed:
                    if recipients:
                        revisions.append((finding, critique, reflection, recipients[0]))
            if not revisions or round_number >= self.collaboration_rounds:
                break
            revised_by_key = {}
            for original, critique, reflection, source in revisions:
                assignment = assignments.get(source)
                agent = agents.get(assignment.agent) if assignment else None
                if not assignment or not agent:
                    continue
                revised_assignment = ReviewAssignment(
                    agent=assignment.agent, objective=assignment.objective,
                    files=list(assignment.files), risk_domains=list(assignment.risk_domains),
                    assignment_id=assignment.assignment_id, round=round_number + 1,
                    reason="critic-requested-revision",
                    shard_id=assignment.shard_id, shard_files=list(assignment.shard_files),
                    coverage_scope=assignment.coverage_scope,
                )
                self._emit(
                    state, self.critic.name, assignment.agent, "revision_request",
                    {"objections": critique.objections, "guidance": reflection.guidance},
                    finding_key(original),
                )
                revised, _attempts, error, _execution = self._invoke_agent(
                    state, agent, revised_assignment,
                    reflection.guidance,
                )
                match = next(
                    (item for item in revised if finding_key(item) == finding_key(original)), None
                )
                if match is not None:
                    revised_by_key[finding_key(original)] = match
                    self._emit(
                        state, assignment.agent, self.critic.name, "revision_response",
                        {"finding": match.to_dict(), "resolved": True}, finding_key(original),
                    )
                elif error:
                    self._emit(
                        state, assignment.agent, self.critic.name, "revision_response",
                        {"resolved": False, "error": error[:500]}, finding_key(original),
                    )
            findings = [revised_by_key.get(finding_key(item), item) for item in findings]
        return {
            "specialist_findings": findings,
            "critiques": critiques,
            "rounds_completed": rounds_completed,
        }

    def _evidence_node(self, state: CollaborationState) -> Dict[str, Any]:
        reproductions = {}
        for finding in state["specialist_findings"]:
            reproduction = self.evidence_agent.reproduce(finding, state["parsed"])
            reproductions[reproduction.finding_key] = reproduction
            self._emit(
                state, self.evidence_agent.name, self.verifier.name, "evidence_report",
                asdict(reproduction), reproduction.finding_key,
            )
        return {"reproductions": reproductions}

    def _verify_node(self, state: CollaborationState) -> Dict[str, Any]:
        fix_ready = {}
        decisions = {}
        for finding in state["specialist_findings"]:
            key = finding_key(finding)
            source_line = next((item.content for item in state["parsed"].added_lines
                                if item.path == finding.path and item.line == finding.line), "")
            if not finding.evidence_refs and source_line:
                finding.evidence_refs = [state["evidence_ledger"].record_changed_line(
                    key, finding.path, finding.line, source_line,
                )]
            else:
                finding.evidence_refs = state["evidence_ledger"].link(
                    key, finding.evidence_refs, finding.path, finding.line,
                )
            ready = self.fix_agent.assess(finding)
            fix_ready[key] = ready
            decision = self.verifier.verify(
                finding, state["critiques"][key], state["reproductions"][key], ready,
                state["evidence_ledger"].valid_for(key, finding.path, finding.line),
            )
            decisions[key] = decision
            self._emit(
                state, self.verifier.name, self.arbiter.name, "verification_decision",
                asdict(decision), key,
            )
        return {"fix_ready": fix_ready, "decisions": decisions}

    def _arbitrate_node(self, state: CollaborationState) -> Dict[str, Any]:
        verified = self.arbiter.decide(
            state["specialist_findings"], state["decisions"]
        )
        rejected = [
            {"finding_key": key, "reasons": decision.reasons}
            for key, decision in state["decisions"].items() if not decision.approved
        ]
        self._emit(
            state, self.arbiter.name, "review-report", "arbitration_decision",
            {
                "approved_findings": [item.to_dict() for item in verified],
                "rejected_findings": rejected,
            },
        )
        if self.memory_manager and state.get("repository"):
            approved_keys = {finding_key(item) for item in verified}
            for finding in state["specialist_findings"]:
                key = finding_key(finding)
                decision = state["decisions"][key]
                self.memory_manager.remember_finding(
                    state.get("tenant_id", "default"), state["repository"],
                    state.get("task_id", ""), finding.to_dict(),
                    key in approved_keys, decision.reasons,
                    pr_number=state.get("pull_request"), source_sha=state.get("source_sha", ""),
                    source_agents=[getattr(state.get("assignments_by_agent", {}).get(source), "agent", "")
                                   for source in state.get("finding_sources", {}).get(key, [])
                                   if getattr(state.get("assignments_by_agent", {}).get(source), "agent", "")],
                    shard_ids=[getattr(state.get("assignments_by_agent", {}).get(source), "shard_id", "")
                               for source in state.get("finding_sources", {}).get(key, [])
                               if getattr(state.get("assignments_by_agent", {}).get(source), "shard_id", "")],
                    cross_shard="cross-shard" in state.get("finding_sources", {}).get(key, []),
                )
            self.memory_manager.compute_finding_delta(
                state.get("tenant_id", "default"), state["repository"], state.get("pull_request"),
                state.get("source_sha", ""), [item.to_dict() for item in verified], state["parsed"].files,
            )
            if state.get("task_id"):
                outcomes = state.get("agent_outcomes", [])
                memory_summary = {
                    "proposed_findings": len(state.get("specialist_findings", [])),
                    "approved_findings": len(verified),
                    "rejected_findings": len(rejected),
                    "dialogue_rounds": state.get("rounds_completed", 0),
                    "agent_loop_steps": sum(
                        int((item.get("execution") or {}).get("loop_steps", 0))
                        for item in outcomes
                    ),
                    "tool_calls": sum(
                        int((item.get("execution") or {}).get("tool_calls", 0))
                        for item in outcomes
                    ),
                    "shard_count": len(state.get("shards") or []),
                    "reviewed_files": sorted({path for item in outcomes if item.get("status") in {"completed", "fallback"}
                                             for path in item.get("assignment").files}),
                    "coverage_ratio": round(len({path for item in outcomes if item.get("status") in {"completed", "fallback"}
                                                for path in item.get("assignment").files}) / max(1, len(state["parsed"].files)), 4),
                    "coverage_gaps": list(state.get("coverage_gaps") or []),
                    "cross_shard_findings": len(state.get("cross_shard_findings") or []),
                    "repo_tool_calls": sum(int((item.get("execution") or {}).get("repo_tool_calls", 0)) for item in outcomes),
                    "context_compaction_count": sum(int((item.get("execution") or {}).get("context_compaction_count", 0)) for item in outcomes),
                    "context_compactions": sum(int((item.get("execution") or {}).get("context_compaction_count", 0)) for item in outcomes),
                }
                archived = self.memory_manager.consolidate_task(
                    state.get("tenant_id", "default"), state["repository"],
                    state["task_id"], memory_summary,
                )
                self._emit(
                    state, "memory-manager", "agent-runtime", "memory_consolidated",
                    {
                        "task_id": state["task_id"],
                        "summary_memory_id": (archived or {}).get("id", ""),
                        "working_memory_released": True,
                    },
                )
        return {"verified": verified}

    def _make_summary(self, state: CollaborationState) -> dict:
        outcomes = list(state.get("agent_outcomes", [])) + list(
            state.get("cross_shard_outcomes", [])
        )
        decisions = state.get("decisions", {})
        adaptive = self.review_mode != "legacy"
        agent_runs = state.get("agent_runs", [])
        omitted = {
            path for gap in state.get("coverage_gaps") or []
            if "omitted" in str(gap.get("reason", ""))
            for path in gap.get("files", [])
        }
        planned = {item.assignment_id: item for item in (state.get("plan").assignments if state.get("plan") else [])}
        reviewed = {
            path for item in outcomes if item["status"] in {"completed", "fallback"}
            and planned.get(item["assignment_id"], item["assignment"]).reason == "coverage-owner"
            for path in planned.get(item["assignment_id"], item["assignment"]).files
            if path not in omitted
        }
        shard_hunks = {item.shard_id: item.hunk_count for item in state.get("shards") or []}
        omitted_by_shard = {}
        for outcome in outcomes:
            assignment = outcome["assignment"]
            metadata = (outcome.get("execution") or {}).get("context") or {}
            omitted_hunks = int(metadata.get("diff", metadata).get("omitted_hunks", 0))
            previous = omitted_by_shard.get(assignment.shard_id)
            omitted_by_shard[assignment.shard_id] = (
                omitted_hunks if previous is None else min(previous, omitted_hunks)
            )
        total_hunks = sum(shard_hunks.values())
        visible_hunks = sum(max(0, count - omitted_by_shard.get(shard_id, 0))
                            for shard_id, count in shard_hunks.items())
        total_added = len(state["parsed"].added_lines)
        omitted_added = sum(
            int((((item.get("execution") or {}).get("context") or {}).get("diff") or {}).get("omitted_added_lines", 0))
            for item in outcomes if planned.get(item["assignment_id"], item["assignment"]).reason == "coverage-owner"
        )
        visible_added = max(0, total_added - omitted_added)
        assignment_matrix = []
        for shard in state.get("shards") or []:
            entries = [item for item in (state.get("plan").assignments if state.get("plan") else [])
                       if item.shard_id == shard.shard_id]
            owner = next((item.agent for item in entries if item.reason == "coverage-owner"), "")
            assignment_matrix.append({
                "shard_id": shard.shard_id, "files": list(shard.files), "coverage_owner": owner,
                "supplemental_reviewers": [item.agent for item in entries if item.reason != "coverage-owner"],
                "risk_tags": sorted({tag for path in shard.files for tag in next(
                    (entry.risk_tags for entry in state["pr_map"].files if entry.path == path), [])}),
            })
        cross_execution = state.get("cross_shard_execution") or {}
        loop_capable = {item.name for item in self.agents if callable(getattr(item, "agent_step", None))}
        summary = {
            "protocol": "plan-challenge-revise-evidence-verify-arbitrate",
            "roles": [
                self.planner.name, "specialists", self.critic.name,
                self.reflection_agent.name,
                self.evidence_agent.name, self.verifier.name, self.arbiter.name,
            ],
            "planned_assignments": len(state.get("plan").assignments) if state.get("plan") else 0,
            "dialogue_rounds": state.get("rounds_completed", 0),
            "messages": self._bus(state).count(),
            "retries": self._bus(state).count("retry_request"),
            "handoffs": self._bus(state).count("assignment_handoff"),
            "agents": [
                {
                    "agent": item["agent"], "status": item["status"],
                    "attempts": item["attempts"],
                    "substituted_for": item.get("substituted_for", ""),
                    "loop_steps": (item.get("execution") or {}).get("loop_steps", 0),
                    "loop_stop_reason": (
                        item.get("execution") or {}
                    ).get("loop_stop_reason", "one-shot"),
                    "context_compressed": bool(
                        ((item.get("execution") or {}).get("context") or {}).get("compressed")
                    ),
                    "memories_recalled": (
                        item.get("execution") or {}
                    ).get("memories_recalled", 0),
                }
                for item in outcomes
            ],
            "agent_loop_steps": sum(
                int((item.get("execution") or {}).get("loop_steps", 0))
                for item in outcomes
            ),
            "tool_calls": (sum(item.tool_calls for item in agent_runs) if adaptive else sum(
                int((item.get("execution") or {}).get("tool_calls", 0))
                for item in outcomes
            )),
            "llm_calls": (sum(item.llm_calls for item in agent_runs) if adaptive else sum(
                int((item.get("execution") or {}).get("loop_steps", 0))
                for item in outcomes if item.get("agent") in loop_capable
            ) + int(cross_execution.get("llm_calls", 0))),
            "physical_llm_calls": sum(
                int(((item.get("execution") or {}).get("usage") or {}).get(
                    "physical_llm_calls", 0
                ))
                for item in outcomes + list(state.get("challenge_outcomes", []))
            ),
            "context_compressions": sum(
                bool(((item.get("execution") or {}).get("context") or {}).get("compressed"))
                for item in outcomes
            ),
            "context_compaction_count": sum(
                int((item.get("execution") or {}).get("context_compaction_count", 0))
                for item in outcomes
            ),
            "compressed_rounds": sum(
                int((item.get("execution") or {}).get("compressed_rounds", 0))
                for item in outcomes
            ),
            "pinned_evidence_count": len(state["evidence_ledger"].pinned()),
            "memories_recalled": sum(
                int((item.get("execution") or {}).get("memories_recalled", 0))
                for item in outcomes
            ),
            "proposed_findings": len(state.get("claims", [])) if adaptive else len(state.get("specialist_findings", [])),
            "total_input_tokens": (sum(item.input_tokens for item in agent_runs) if adaptive else sum(int(((item.get("execution") or {}).get("context") or {}).get("estimated_tokens", 0)) for item in outcomes)),
            "max_input_tokens": (max([item.input_tokens for item in agent_runs] or [0]) if adaptive else max([int(((item.get("execution") or {}).get("context") or {}).get("estimated_tokens", 0)) for item in outcomes] or [0])),
            "approved_findings": len(state.get("verified", [])) if adaptive else sum(1 for item in decisions.values() if item.approved),
            "rejected_findings": len(state.get("adaptive_rejected", [])) if adaptive else sum(1 for item in decisions.values() if not item.approved),
            "shard_count": len(state.get("shards") or []),
            "reviewed_files": sorted(reviewed),
            "unreviewed_files": sorted(set(state["parsed"].files).difference(reviewed)),
            "coverage_ratio": round(len(reviewed) / max(1, len(state["parsed"].files)), 4),
            "changed_hunks": total_hunks,
            "visible_hunks": visible_hunks,
            "changed_hunk_coverage": round(visible_added / total_added, 4) if total_added else 1.0,
            "changed_added_lines": total_added,
            "visible_added_lines": visible_added,
            "omitted_added_lines": omitted_added,
            "coverage_gaps": list(state.get("coverage_gaps") or []),
            "coverage_receipts": list(state.get("coverage_receipts") or []),
            "budget_gaps": list(state.get("budget_gaps") or []),
            "global_budget": state["shared_budget"].snapshot(),
            "verifier_batches": list(state.get("verifier_batches") or []),
            "auditor_rounds": list(state.get("audit_rounds") or []),
            "coverage_repairs": [
                {"assignment_id": item["assignment_id"],
                 "required_hunk_ids": list(item["assignment"].required_hunk_ids)}
                for item in state.get("coverage_repair_outcomes", [])
            ],
            "coverage_status": "complete" if not state.get("coverage_gaps") and len(reviewed) == len(state["parsed"].files) else "partial",
            "review_complete": not state.get("coverage_gaps") and len(reviewed) == len(state["parsed"].files),
            "verdict_cap": state.get("coverage_gate", "clear"),
            "specialist_activation": self.specialist_activation,
            "shard_assignments": assignment_matrix,
            "llm_failures": list(state.get("llm_failures") or []),
            "cross_shard_findings": len(state.get("cross_shard_findings") or []),
            "cross_shard_execution": dict(cross_execution),
            "repo_tool_calls": sum(int((item.get("execution") or {}).get("repo_tool_calls", 0)) for item in outcomes) + int(cross_execution.get("repo_tool_calls", 0)),
            "retrieved_context_tokens": sum(int((item.get("execution") or {}).get("retrieved_context_tokens", 0)) for item in outcomes) + int(cross_execution.get("retrieved_context_tokens", 0)),
            "dropped_observations": sum(int((item.get("execution") or {}).get("dropped_observations", 0)) for item in outcomes),
        }
        participants = {
            item.name: (
                "agent" if getattr(item, "execution_kind", "") == "agent"
                else "deterministic-checker"
            ) for item in self.agents
        }
        participants.update({
            self.planner.name: "deterministic-stage", self.critic.name: "deterministic-stage",
            self.reflection_agent.name: "deterministic-stage",
            self.evidence_agent.name: "deterministic-stage",
            self.verifier.name: "deterministic-stage", self.arbiter.name: "deterministic-stage",
        })
        # Evidence IDs address immutable tool content, so one observation can
        # legitimately support more than one Claim.  Do not serialize the
        # per-Claim bindings as duplicate IDs: dict-based report consumers
        # would otherwise keep only the last Claim and report false provenance
        # failures.  Direction remains meaningful per decision-time binding;
        # the summary marks a cross-Claim mixture explicitly.
        merged_evidence = {}
        for values in state.get("claim_evidence", {}).values():
            for item in values:
                current = merged_evidence.get(item.evidence_id)
                if current is None:
                    merged_evidence[item.evidence_id] = item
                    continue
                direction = current.direction if current.direction == item.direction else "mixed"
                merged_evidence[item.evidence_id] = replace(
                    current,
                    claim_ids=tuple(sorted(set(current.claim_ids).union(item.claim_ids))),
                    direction=direction,
                )
        summary.update({
            "schema_version": 2,
            "architecture": (
                "legacy-collaboration" if not adaptive
                else "agent-specialists-with-deterministic-gates"
            ),
            "review_mode": self.review_mode,
            "challenge_strategy": self.challenge_strategy,
            "participant_kinds": participants,
            "participants": [{"name": key, "kind": value} for key, value in sorted(participants.items())],
            "risk_scores": dict(state.get("risk_scores") or {}),
            "activated_domains": list(state.get("activated_domains") or []),
            "challenge_activation": dict(state.get("challenge_activation") or {}),
            "agent_runs": [item.to_dict() for item in agent_runs],
            "agent_count": len(agent_runs),
            "task_execution_success": True,
            "semantic_review_complete": (
                True if self.review_mode == "rules_only" else
                bool(agent_runs) and not state.get("llm_failures") and all(
                    item.status in {"completed", "fallback"} for item in agent_runs
                )
            ),
            "budget_compliant": not state["shared_budget"].snapshot()["violations"],
            "model_trace": [
                attempt.to_dict() for run in agent_runs for attempt in run.request_attempts
            ],
            "effective_providers": list(dict.fromkeys(
                provider for run in agent_runs for provider in run.effective_providers
            )),
            "effective_models": list(dict.fromkeys(
                model for run in agent_runs for model in run.effective_models
            )),
            "model_switch_count": sum(item.model_switch_count for item in agent_runs),
            "claims": [item.to_dict() for item in state.get("claims", [])],
            "evidence": [merged_evidence[key].to_dict() for key in sorted(merged_evidence)],
            "challenges": [item.to_dict() for item in state.get("challenges", [])],
            "decisions": [item.to_dict() for item in state.get("adaptive_decisions", [])],
            "escalations": list(state.get("escalations", [])),
            "budget": state["shared_budget"].snapshot(),
            "usage_source": (
                "provider" if agent_runs and all(item.usage_source == "provider" for item in agent_runs)
                else "estimated"
            ),
            "total_output_tokens": sum(item.output_tokens for item in agent_runs),
            "total_cached_tokens": sum(item.cached_tokens for item in agent_runs),
            "fingerprints": {
                "source": state.get("source_sha", ""),
                "decision_policy": artifact_fingerprint(self.decision_policy.VERSION),
                "agent_specs": artifact_fingerprint([item.spec.to_dict() for item in agent_runs]),
                "budget": artifact_fingerprint(state["shared_budget"].snapshot()["limits"]),
            },
        })
        return summary
