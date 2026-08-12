"""Multi-agent review with planning, dialogue, verification and arbitration.

The coordinator implements a bounded collaboration protocol:
plan -> specialist review -> peer challenge -> evidence revision -> independent
verification -> arbitration.  Every hand-off is persisted as an agent message
when a task store is available.  Failed specialists are retried and then
replanned to a substitute reviewer.
"""
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, TypedDict

from .context_manager import ContextBundle, ContextManager
from .context.pr_map import PRContextMap, build_pr_context_map
from .context.retrieval import RepositoryRetrieval, RepositorySnapshotProvider
from .context.evidence import EvidenceLedger
from .context.shard_planner import ReviewShard, ShardPlanner
from .diff_parser import ParsedDiff
from .memory import MemoryManager
from .models import Finding, Severity
from .reviewer import LocalRuleReviewer, Reviewer
from .runtime import AgentLoop, AgentRuntime, AgentTool, RuntimeBudgetExceeded, RuntimeNode, ToolRegistry
from .context.reducers import reduce_tool_result


@dataclass
class AgentMessage:
    sender: str
    recipient: str
    kind: str
    content: Dict[str, Any]
    correlation_id: str = ""

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
        message = AgentMessage(sender, recipient, kind, content, correlation_id)
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

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReviewPlan:
    languages: List[str]
    changed_files: List[str]
    risk_level: str
    assignments: List[ReviewAssignment]

    def to_dict(self) -> dict:
        return {
            "languages": self.languages,
            "changed_files": self.changed_files,
            "risk_level": self.risk_level,
            "assignments": [item.to_dict() for item in self.assignments],
        }


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
        agent_runtime_max_steps: int = 8, agent_runtime_timeout_seconds: int = 300,
        context_active_rounds: int = 3, context_soft_compact_ratio: float = .60,
        context_hard_compact_ratio: float = .80,
        context_architecture: str = "coverage-first",
        specialist_activation: str = "hybrid",
        shard_file_threshold: int = 12, shard_changed_line_threshold: int = 1200,
        snapshot_provider: Optional[RepositorySnapshotProvider] = None,
        snapshot_factory=None,
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
        self.runtime = AgentRuntime(
            max_steps=agent_runtime_max_steps, timeout_seconds=agent_runtime_timeout_seconds,
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
        source_sha: str = "",
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
        }
        if self.snapshot_factory:
            state["snapshot_provider"] = self.snapshot_factory(
                repository, pull_request, state["pr_map"], task_id
            )
            if not source_sha:
                head = str(getattr(state["snapshot_provider"], "ref", ""))
                if head:
                    state["source_sha"] = head
        result = self.runtime.execute(
            state,
            [
                RuntimeNode("planner", self._plan_node, checkpoint=False),
                RuntimeNode("specialists", self._specialist_node, checkpoint=False),
                RuntimeNode("cross_shard", self._cross_shard_node, checkpoint=False),
                RuntimeNode("deliberation", self._deliberation_node, checkpoint=False),
                RuntimeNode("evidence", self._evidence_node, checkpoint=False),
                RuntimeNode("verifier", self._verify_node, checkpoint=False),
                RuntimeNode("arbiter", self._arbitrate_node, checkpoint=False),
            ],
            task_id=task_id,
        )
        summary = self._make_summary(result)
        self._last_summary = summary
        if task_id:
            with self._summary_lock:
                self._summaries[task_id] = summary
        return result["verified"]

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
        cross_shard: bool = False,
    ) -> ToolRegistry:
        shard = (state.get("shards_by_id") or {}).get(assignment.shard_id)
        local_diff = shard.diff if shard else state["diff"]
        local_parsed = shard.parsed if shard else state["parsed"]
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
        registry = ToolRegistry([
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
        ])
        registry.retrieval = retrieval
        return registry

    def _run_agent_loop(
        self, state: CollaborationState, agent: Reviewer,
        assignment: ReviewAssignment, feedback: Optional[List[str]],
    ) -> tuple:
        memories = self._recall_memories(state, assignment)
        shard = (state.get("shards_by_id") or {}).get(assignment.shard_id)
        local_diff = shard.diff if shard else state["diff"]
        local_parsed = shard.parsed if shard else state["parsed"]
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
        tools = self._agent_tools(state, assignment)
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
                result = detail.get("result") if isinstance(detail.get("result"), dict) else {}
                path = result.get("path") if isinstance(result, dict) else ""
                if path:
                    working_state["files_inspected"].append(str(path))
                    working_state["pending_files"] = [item for item in working_state["pending_files"] if item != path]

        loop_state = {
            "diff": local_diff, "context": bundle.text,
            "context_metadata": bundle.metadata(), "parsed": local_parsed,
            "assignment": assignment.to_dict(), "feedback": list(feedback or []),
            "inbox": self._bus(state).inbox(agent.name, assignment.assignment_id),
            "memories": memories, "available_tools": tools.catalog(),
            "pr_map": state["pr_map"].compact(),
            "shard_identity": shard.identity() if shard else {"id": "full", "files": assignment.files},
        }
        last_context = {"metadata": bundle.metadata()}

        def managed_step(loop_iteration: Dict[str, Any]) -> Dict[str, Any]:
            context_assignment = assignment.to_dict()
            context_assignment["pr_map"] = state["pr_map"].compact()
            managed = self.context_manager.compose(
                bundle, context_assignment, feedback=list(feedback or []),
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
            action = getattr(agent, "agent_step")(prepared)
            if isinstance(action, dict):
                action["_context_tokens"] = managed.estimated_tokens
                action["_context_max_tokens"] = self.context_manager.max_tokens
            return action

        result = self.agent_loop.run(
            managed_step, tools, loop_state, on_event,
        )
        findings = list(result.output or [])
        if not all(isinstance(item, Finding) for item in findings):
            raise TypeError("agent loop final output must contain Finding objects")
        working_state["candidate_findings"] = [finding_key(item) for item in findings]
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
            "tool_calls": len(result.observations),
            "repo_tool_calls": tools.retrieval.calls,
            "retrieved_context_tokens": tools.retrieval.retrieved_bytes // 4,
            "context_compaction_count": result.loop_context.get("compaction_count", 0),
            "compressed_rounds": result.loop_context.get("compressed_rounds", 0),
            "pinned_evidence_count": len(result.loop_context.get("pinned_evidence", [])),
            "pinned_evidence": result.loop_context.get("pinned_evidence", []),
            "dropped_observations": int(last_context["metadata"].get("dropped_observations", 0)),
        }

    def _invoke_agent(
        self, state: CollaborationState, agent: Reviewer,
        assignment: ReviewAssignment, feedback: Optional[List[str]] = None,
    ) -> tuple:
        last_error = None
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
                        state, agent, assignment, feedback
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
                        "findings": [item.to_dict() for item in findings],
                        "execution": execution,
                    }, assignment.assignment_id,
                )
                return findings, attempt, "", execution
            except Exception as exc:
                last_error = str(exc)
                self._emit(
                    state, agent.name, self.planner.name, "agent_failure",
                    {"attempt": attempt, "error": last_error[:1000]},
                    assignment.assignment_id,
                )
                if attempt <= self.agent_retries:
                    self._emit(
                        state, self.planner.name, agent.name, "retry_request",
                        {"next_attempt": attempt + 1, "reason": last_error[:500]},
                        assignment.assignment_id,
                    )
        return (
            [], self.agent_retries + 1, last_error or "unknown agent failure",
            {"loop_steps": 0, "loop_stop_reason": "failed"},
        )

    def _replacement_candidates(self, failed_agent: Reviewer) -> List[Reviewer]:
        values = [item for item in self.agents if item is not failed_agent]
        if all(item.name != self.fallback_agent.name for item in values):
            values.append(self.fallback_agent)
        return values

    def _run_assignment(
        self, state: CollaborationState, assignment: ReviewAssignment,
        agent: Reviewer,
    ) -> dict:
        findings, attempts, error, execution = self._invoke_agent(
            state, agent, assignment
        )
        result = {
            "agent": agent.name, "assignment_id": assignment.assignment_id,
            "attempts": attempts, "status": "completed" if not error else (
                "timed_out" if "budget" in error.lower() or "timeout" in error.lower() else "failed"
            ),
            "findings": findings, "error": error, "substituted_for": "",
            "assignment": assignment, "execution": execution,
        }
        if not error:
            return result
        if callable(getattr(agent, "agent_step", None)):
            state.setdefault("llm_failures", []).append({
                "assignment_id": assignment.assignment_id, "shard_id": assignment.shard_id,
                "agent": agent.name, "error": error[:500],
            })
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
        }

    def _specialist_node(self, state: CollaborationState) -> Dict[str, Any]:
        outcomes = []
        by_name = {item.name: item for item in self.agents}
        by_name.setdefault(self.fallback_agent.name, self.fallback_agent)
        assignments = state["plan"].assignments
        with ThreadPoolExecutor(
            max_workers=min(self.max_workers, max(1, len(assignments)))
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
                    and owner_assignment.reason == "coverage-owner"):
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

    def _cross_shard_node(self, state: CollaborationState) -> Dict[str, Any]:
        """Run a compact, bounded second pass without ever restoring the full Diff."""
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
        for agent in self.agents:
            stepper = getattr(agent, "agent_step", None)
            if not stepper:
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
        outcomes = state.get("agent_outcomes", [])
        decisions = state.get("decisions", {})
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
        return {
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
            "tool_calls": sum(
                int((item.get("execution") or {}).get("tool_calls", 0))
                for item in outcomes
            ),
            "llm_calls": sum(
                int((item.get("execution") or {}).get("loop_steps", 0))
                for item in outcomes if item.get("agent") in loop_capable
            ) + int(cross_execution.get("llm_calls", 0)),
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
            "proposed_findings": len(state.get("specialist_findings", [])),
            "total_input_tokens": sum(int(((item.get("execution") or {}).get("context") or {}).get("estimated_tokens", 0)) for item in outcomes),
            "max_input_tokens": max([int(((item.get("execution") or {}).get("context") or {}).get("estimated_tokens", 0)) for item in outcomes] or [0]),
            "approved_findings": sum(1 for item in decisions.values() if item.approved),
            "rejected_findings": sum(1 for item in decisions.values() if not item.approved),
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
            "coverage_status": "complete" if not state.get("coverage_gaps") and len(reviewed) == len(state["parsed"].files) else "partial",
            "review_complete": not state.get("coverage_gaps") and len(reviewed) == len(state["parsed"].files),
            "specialist_activation": self.specialist_activation,
            "shard_assignments": assignment_matrix,
            "llm_failures": list(state.get("llm_failures") or []),
            "cross_shard_findings": len(state.get("cross_shard_findings") or []),
            "repo_tool_calls": sum(int((item.get("execution") or {}).get("repo_tool_calls", 0)) for item in outcomes) + int(cross_execution.get("repo_tool_calls", 0)),
            "retrieved_context_tokens": sum(int((item.get("execution") or {}).get("retrieved_context_tokens", 0)) for item in outcomes) + int(cross_execution.get("retrieved_context_tokens", 0)),
            "dropped_observations": sum(int((item.get("execution") or {}).get("dropped_observations", 0)) for item in outcomes),
        }
