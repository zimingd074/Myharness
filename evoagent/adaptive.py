"""Typed artifacts and deterministic policies for adaptive review collaboration."""
import ast
import hashlib
import io
import json
import re
import threading
import tokenize
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .diff_parser import ParsedDiff
from .models import Finding, Severity


RULE_CWE_ALIASES = {
    "SEC-EVAL": "CWE-95", "SEC-SUBPROCESS-SHELL": "CWE-78",
    "SEC-HARDCODED-SECRET": "CWE-798", "SEC-SQL-CONCAT": "CWE-89",
    "REL-EMPTY-EXCEPT": "CWE-703", "REL-DEBUG-PRINT": "CWE-532",
    "SEC-PATH-TRAVERSAL": "CWE-22", "SEC-YAML-LOAD": "CWE-502",
    "SEC-PICKLE-LOAD": "CWE-502", "SEC-WEAK-HASH": "CWE-328",
    "SEC-INSECURE-TEMPFILE": "CWE-377", "SEC-WEAK-RANDOM": "CWE-330",
    "REL-UNBOUNDED-RETRY": "CWE-835", "SEC-ASSERT-AUTH": "CWE-617",
    "SEC-INSECURE-COOKIE": "CWE-614", "REL-FLOAT-MONEY": "CWE-682",
    "REL-NAIVE-DATETIME": "CWE-367", "REL-BLOCKING-ASYNC": "CWE-400",
    "REL-NONATOMIC-WRITE": "CWE-362", "SEC-OPEN-REDIRECT": "CWE-601",
    "SEC-LOG-FORGING": "CWE-117", "BUSINESS-NEGATIVE-BALANCE": "CWE-840",
    "SEC-AUTHZ-BYPASS": "CWE-863", "COR-API-ARITY": "CWE-628",
}
SUPPORTED_AGENT_CWES = frozenset(RULE_CWE_ALIASES.values())

# These terms are deliberately narrow.  They do not prove a defect; they stop
# a model from labelling an unrelated causal story with a familiar CWE merely
# to pass the rule manifest gate.
CAUSAL_FAMILY_TERMS = {
    "CWE-95": ("eval", "exec", "dynamic execution", "code execution", "动态执行", "代码执行"),
    "CWE-89": ("sql", "query", "execute", "injection", "查询", "注入"),
    "CWE-703": ("exception", "error", "raise", "swallow", "failure", "异常", "错误", "故障"),
    "CWE-863": ("authoriz", "permission", "tenant", "ownership", "access control", "授权", "权限", "租户", "所有权"),
    "CWE-628": ("argument", "arity", "signature", "caller", "callee", "typeerror", "参数", "签名", "调用"),
}


def canonical_rule_id(rule_id: str) -> str:
    value = str(rule_id).strip().upper()
    return RULE_CWE_ALIASES.get(value, value)


def _digest(value: Any, length: int = 20) -> str:
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:length]


@dataclass(frozen=True)
class ModelEndpoint:
    provider: str
    model: str
    base_url: str
    headers: Tuple[Tuple[str, str], ...] = ()

    def to_dict(self) -> dict:
        return {
            "provider": self.provider, "model": self.model,
            "base_url": self.base_url,
            "header_names": [key for key, _value in self.headers],
        }


@dataclass(frozen=True)
class ModelRequestAttempt:
    request_id: str
    logical_call_id: str
    attempt_index: int
    provider: str
    model: str
    endpoint_host: str
    status: str
    http_status: int = 0
    error_code: str = ""
    fallback_reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    usage_source: str = "estimated"
    finish_reason: str = ""
    latency_ms: int = 0

    def __post_init__(self) -> None:
        if self.status not in {"success", "quota_exhausted", "failed"}:
            raise ValueError("invalid model request status: %s" % self.status)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AgentSpec:
    agent_id: str
    role: str
    provider: str = ""
    model: str = ""
    prompt_hash: str = ""
    capabilities: Tuple[str, ...] = ()
    context_lineage: str = ""
    independence_group: str = ""
    max_steps: int = 0
    context_independent: bool = False
    model_family: str = ""
    provider_family: str = ""
    prompt_lineage: str = ""

    def to_dict(self) -> dict:
        value = asdict(self)
        value["capabilities"] = list(self.capabilities)
        return value


@dataclass(frozen=True)
class AgentRun:
    run_id: str
    spec: AgentSpec
    assignment_id: str
    snapshot_id: str
    status: str
    findings: Tuple[str, ...] = ()
    tool_calls: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    usage_source: str = "estimated"
    error: str = ""
    request_attempts: Tuple[ModelRequestAttempt, ...] = ()
    effective_providers: Tuple[str, ...] = ()
    effective_models: Tuple[str, ...] = ()
    model_switch_count: int = 0

    def to_dict(self) -> dict:
        value = asdict(self)
        value["spec"] = self.spec.to_dict()
        value["findings"] = list(self.findings)
        value["request_attempts"] = [item.to_dict() for item in self.request_attempts]
        value["effective_providers"] = list(self.effective_providers)
        value["effective_models"] = list(self.effective_models)
        return value


@dataclass(frozen=True)
class Claim:
    claim_id: str
    cluster_key: str
    rule_id: str
    severity: str
    path: str
    line: int
    causal_hypothesis: str
    preconditions: str
    impact: str
    remediation: str
    test_plan: str
    proposer_run_id: str
    source_kind: str
    evidence_refs: Tuple[str, ...] = ()
    version: int = 1
    supersedes_claim_id: str = ""

    @classmethod
    def from_finding(
        cls, finding: Finding, proposer_run_id: str, source_kind: str,
    ) -> "Claim":
        cluster_key = _digest({
            "path": finding.path, "line": finding.line,
            "rule_id": canonical_rule_id(finding.rule_id),
            "hypothesis": (
                finding.explanation.strip().lower()
                if source_kind != "deterministic-checker" else ""
            ),
        }, 16)
        claim_id = _digest({
            "cluster": cluster_key, "hypothesis": finding.explanation.strip().lower(),
            "proposer": proposer_run_id,
        })
        return cls(
            claim_id, cluster_key, finding.rule_id, finding.severity.value,
            finding.path, finding.line, finding.explanation, "",
            finding.title, finding.fix, finding.test, proposer_run_id,
            source_kind, tuple(finding.evidence_refs),
        )

    def to_dict(self) -> dict:
        value = asdict(self)
        value["evidence_refs"] = list(self.evidence_refs)
        return value


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    snapshot_id: str
    path: str
    line: int
    kind: str
    source_tool: str
    producer_run_id: str
    result_digest: str
    trust_level: str
    claim_ids: Tuple[str, ...] = ()
    direction: str = "support"
    result_nonempty: bool = False
    tool_version: str = "1"

    def to_dict(self) -> dict:
        value = asdict(self)
        value["claim_ids"] = list(self.claim_ids)
        return value


@dataclass(frozen=True)
class Challenge:
    claim_id: str
    verdict: str
    rationale: str
    auditor_run_id: str
    evidence_refs: Tuple[str, ...] = ()
    counter_hypothesis: str = ""
    counterexample: str = ""

    def __post_init__(self) -> None:
        if self.verdict not in {"support", "refute", "insufficient", "new_claim"}:
            raise ValueError("invalid challenge verdict: %s" % self.verdict)

    def to_dict(self) -> dict:
        value = asdict(self)
        value["evidence_refs"] = list(self.evidence_refs)
        return value


@dataclass(frozen=True)
class ChallengeResponse:
    claim_id: str
    verdict: str
    evidence_refs: Tuple[str, ...] = ()
    rationale: str = ""
    counterexample: str = ""
    counter_hypothesis: str = ""

    def __post_init__(self) -> None:
        if self.verdict not in {"support", "refute", "insufficient", "new_claim"}:
            raise ValueError("invalid challenge verdict: %s" % self.verdict)

    def to_dict(self) -> dict:
        value = asdict(self)
        value["evidence_refs"] = list(self.evidence_refs)
        return value


@dataclass(frozen=True)
class RevisionResponse:
    claim_id: str
    action: str
    evidence_refs: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.action not in {"retain", "revise", "withdraw"}:
            raise ValueError("invalid revision action: %s" % self.action)

    def to_dict(self) -> dict:
        value = asdict(self)
        value["evidence_refs"] = list(self.evidence_refs)
        return value


@dataclass(frozen=True)
class Decision:
    claim_id: str
    outcome: str
    reason_codes: Tuple[str, ...]
    policy_version: str = "adaptive-decision-v2"
    evidence_refs: Tuple[str, ...] = ()
    supporting_run_ids: Tuple[str, ...] = ()
    refuting_run_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.outcome not in {"accept", "reject", "escalate"}:
            raise ValueError("invalid decision outcome: %s" % self.outcome)

    def to_dict(self) -> dict:
        value = asdict(self)
        for key in ("reason_codes", "evidence_refs", "supporting_run_ids", "refuting_run_ids"):
            value[key] = list(value[key])
        return value


class SharedAgentBudget:
    """Thread-safe hard budget shared by every agent in one review."""

    def __init__(
        self, max_agent_runs: int = 4, max_llm_calls: int = 12,
        max_tool_calls: int = 24, max_input_tokens: int = 60000,
        max_output_tokens: int = 12000, tail_reserve: Dict[str, int] = None,
    ):
        self.limits = {
            "agent_runs": max_agent_runs, "llm_calls": max_llm_calls,
            "tool_calls": max_tool_calls, "input_tokens": max_input_tokens,
            "output_tokens": max_output_tokens,
        }
        self.used = {key: 0 for key in self.limits}
        self.tail_reserve = {
            key: max(0, min(int(value), self.limits.get(key, 0)))
            for key, value in dict(tail_reserve or {}).items() if key in self.limits
        }
        self.violations: List[str] = []
        self._lock = threading.Lock()

    def reserve(self, field: str, amount: int = 1, stage: str = "tail") -> bool:
        amount = max(0, int(amount))
        with self._lock:
            protected = self.tail_reserve.get(field, 0) if stage == "review" else 0
            if self.used[field] + amount > self.limits[field] - protected:
                if field not in self.violations:
                    self.violations.append(field)
                return False
            self.used[field] += amount
            return True

    def replace_estimate(self, field: str, estimated: int, actual: int) -> None:
        with self._lock:
            self.used[field] = max(0, self.used[field] - max(0, estimated)) + max(0, actual)
            if self.used[field] > self.limits[field] and field not in self.violations:
                self.violations.append(field)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "limits": dict(self.limits), "used": dict(self.used),
                "tail_reserve": dict(self.tail_reserve),
                "violations": list(self.violations),
            }

    def remaining(self, field: str) -> int:
        with self._lock:
            return max(0, self.limits[field] - self.used[field])


class DeterministicRiskRouter:
    """Scores security and reliability risk without model involvement."""

    SECURITY = re.compile(
        r"(?i)auth|tenant|permission|token|secret|password|cookie|sql|eval|exec|shell|"
        r"subprocess|pickle|yaml|deserialize|encrypt|crypto|path|redirect"
    )
    RELIABILITY = re.compile(
        r"(?i)retry|timeout|transaction|lock|async|await|queue|cache|except|error|"
        r"rollback|commit|idempot|resource|close|concurr|migration|config"
    )
    CROSS_FILE = re.compile(r"(?i)def |class |import |config|schema|migration|transaction|commit|api")
    AUTH_GUARD = re.compile(r"(?i)tenant|permission|authori[sz]|ownership|is_admin|access[_ ]?control")
    RULE_DOMAINS = {
        "SEC-EVAL": "security", "SEC-SUBPROCESS-SHELL": "security",
        "SEC-HARDCODED-SECRET": "security", "SEC-SQL-CONCAT": "security",
        "SEC-PATH-TRAVERSAL": "security", "SEC-YAML-LOAD": "security",
        "SEC-WEAK-HASH": "security", "SEC-INSECURE-TEMPFILE": "security",
        "SEC-WEAK-RANDOM": "security", "SEC-ASSERT-AUTH": "security",
        "SEC-INSECURE-COOKIE": "security", "SEC-AUTHZ-BYPASS": "security",
        "REL-EMPTY-EXCEPT": "reliability", "REL-DEBUG-PRINT": "reliability",
        "REL-UNBOUNDED-RETRY": "reliability", "COR-API-ARITY": "reliability",
    }

    def score(self, diff: str, parsed: ParsedDiff, scanner_findings: Iterable[Finding]) -> Dict[str, int]:
        scores = {"security": 0, "reliability": 0}
        rules = {finding.rule_id for finding in scanner_findings}
        domains = {self.RULE_DOMAINS.get(rule, "") for rule in rules}
        if "security" in domains:
            scores["security"] += 2
        if "reliability" in domains:
            scores["reliability"] += 2
        executable = "\n".join(
            item.content for item in parsed.added_lines
            if item.content.strip() and not item.content.lstrip().startswith(("#", "//"))
        )
        paths_and_diff = "\n".join(parsed.files) + "\n" + executable
        if self.SECURITY.search(paths_and_diff):
            scores["security"] += 1
        if self.RELIABILITY.search(paths_and_diff):
            scores["reliability"] += 1
        deleted = "\n".join(
            raw[1:] for raw in diff.splitlines()
            if raw.startswith("-") and not raw.startswith("---")
            and raw[1:].strip() and not raw[1:].lstrip().startswith(("#", "//"))
        )
        # A removed ownership/authorization predicate is a deterministic trust
        # boundary signal even when the replacement merely says
        # `is_authenticated`.  It activates investigation; it is not itself a
        # vulnerability verdict.
        if self.AUTH_GUARD.search(deleted) and self.SECURITY.search(paths_and_diff):
            scores["security"] += 1
        if len(parsed.files) > 1 and self.CROSS_FILE.search(executable):
            if self.SECURITY.search(paths_and_diff):
                scores["security"] += 1
            if self.RELIABILITY.search(paths_and_diff):
                scores["reliability"] += 1
        return scores

    def activated(self, scores: Dict[str, int]) -> List[str]:
        return [domain for domain in ("security", "reliability") if scores.get(domain, 0) >= 2]


class AdaptiveDecisionPolicy:
    VERSION = "adaptive-decision-v2"
    SEMANTIC_TOOLS = {
        "read_file", "grep_repo", "find_symbol", "find_references",
        "static-analysis", "test", "runtime",
    }

    @staticmethod
    def _line(parsed: ParsedDiff, finding: Finding) -> str:
        return next((item.content for item in parsed.added_lines
                     if item.path == finding.path and item.line == finding.line), "")

    @staticmethod
    def _python_tokens(line: str) -> List[Tuple[int, str]]:
        try:
            return [(item.type, item.string) for item in tokenize.generate_tokens(io.StringIO(line).readline)]
        except (tokenize.TokenError, IndentationError):
            return []

    @staticmethod
    def _python_ast(line: str) -> Optional[ast.AST]:
        try:
            return ast.parse(line.strip())
        except (SyntaxError, ValueError):
            return None

    def structurally_valid(
        self, finding: Finding, parsed: ParsedDiff, source_kind: str = "",
    ) -> bool:
        line = self._line(parsed, finding)
        if not line:
            return False
        if line.lstrip().startswith("#"):
            # Comment-oriented deterministic policies are legitimate.  Other
            # rules (especially semantic/model claims) cannot prove an
            # executable defect with a comment locator.
            return finding.rule_id in {"QUALITY-UNFINISHED"}
        rule = str(finding.rule_id).strip().upper()
        family = canonical_rule_id(rule)
        if family == "CWE-95":
            tokens = self._python_tokens(line)
            return any(kind == tokenize.NAME and value in {"eval", "exec"}
                       for kind, value in tokens)
        if family == "CWE-863":
            normalized = re.sub(r"\s+", "", line)
            # A same-tenant ownership guard directly contradicts the claimed
            # authorization bypass. More complex policies remain semantic.
            if ("user.tenant_id==invoice.tenant_id" in normalized
                    or "invoice.tenant_id==user.tenant_id" in normalized):
                return False
        tree = self._python_ast(line)
        if family == "CWE-78":
            return bool(tree) and any(
                isinstance(node, ast.Call) and any(
                    keyword.arg == "shell" and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True for keyword in node.keywords
                ) for node in ast.walk(tree)
            )
        if family == "CWE-798":
            sensitive = re.compile(r"(?i)password|passwd|api[_-]?key|secret|token")
            return bool(tree) and any(
                isinstance(node, (ast.Assign, ast.AnnAssign))
                and any(sensitive.search(getattr(target, "id", "")) for target in (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                ))
                and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
                for node in ast.walk(tree)
            )
        if family == "CWE-22":
            return bool(tree) and any(
                isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "open" and node.args
                and isinstance(node.args[0], ast.BinOp) and isinstance(node.args[0].op, ast.Div)
                for node in ast.walk(tree)
            )
        if rule == "SEC-YAML-LOAD":
            return bool(tree) and any(
                isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "load" and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "yaml" for node in ast.walk(tree)
            )
        if family == "CWE-703":
            following = [item.content.strip() for item in parsed.added_lines
                         if item.path == finding.path and finding.line < item.line <= finding.line + 3]
            is_handler = line.lstrip().startswith("except ") or line.lstrip().startswith("except:")
            return is_handler and not any(
                value == "raise" or value.startswith("raise ") for value in following
            )
        if family == "CWE-89" and "column" in line:
            joined = "\n".join(item.content for item in parsed.added_lines if item.path == finding.path)
            allowlisted = (
                re.search(r"\{[^}]+:[^}]+\}\s*\[", joined)
                or re.search(
                    r"\bcolumn\s*=\s*\{[^}]+:[^}]+\}\.get\([^,]+,\s*['\"][^'\"]+['\"]\s*\)",
                    joined,
                )
            )
            if re.search(r"\{[^}]*column[^}]*\}", line) and allowlisted:
                return False
        if family == "CWE-89":
            return bool(tree) and any(
                isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "execute" and node.args
                and isinstance(node.args[0], (ast.JoinedStr, ast.BinOp))
                for node in ast.walk(tree)
            )
        if family == "CWE-628":
            return bool(tree) and any(isinstance(node, ast.Call) for node in ast.walk(tree))
        # Existing deterministic rule packs remain backward compatible.  A
        # semantic Agent, however, may not auto-approve an unknown CWE merely
        # because its changed-line locator exists.
        return source_kind != "agent" or family in SUPPORTED_AGENT_CWES

    def claim_admission_reasons(
        self, claim: Claim, finding: Finding, parsed: ParsedDiff,
    ) -> Tuple[str, ...]:
        family = canonical_rule_id(claim.rule_id)
        if claim.source_kind == "agent" and family not in SUPPORTED_AGENT_CWES:
            return ("unsupported-semantic-family",)
        if not self.structurally_valid(finding, parsed, claim.source_kind):
            return ("not-executable-structure",)
        if claim.source_kind != "agent":
            return ()
        terms = CAUSAL_FAMILY_TERMS.get(family)
        hypothesis = " ".join((claim.causal_hypothesis, claim.impact)).lower()
        if terms and not any(term in hypothesis for term in terms):
            return ("causal-family-mismatch",)
        return ()

    @staticmethod
    def evidence_relevant(claim: Claim, evidence: Evidence) -> bool:
        if evidence.kind in {"static-analysis", "test", "runtime"}:
            return True
        if evidence.path == claim.path:
            return True
        # Cross-file contract evidence is useful only for tools whose result is
        # explicitly a symbol/reference relation, not an arbitrary file read.
        return (
            canonical_rule_id(claim.rule_id) == "CWE-628"
            and evidence.kind in {"find_symbol", "find_references"}
            and bool(evidence.path)
        )

    def decide(
        self, claim: Claim, finding: Finding, parsed: ParsedDiff,
        evidence: Iterable[Evidence], challenges: Iterable[Challenge],
        snapshot_id: str = "",
    ) -> Decision:
        evidence = list(evidence)
        refs = tuple(sorted({item.evidence_id for item in evidence}))
        challenge_values = [item for item in challenges if item.claim_id == claim.claim_id]
        supporting = tuple(sorted({item.auditor_run_id for item in challenge_values
                                   if item.verdict == "support"}))
        refuting = tuple(sorted({item.auditor_run_id for item in challenge_values
                                if item.verdict == "refute"}))
        admission_reasons = self.claim_admission_reasons(claim, finding, parsed)
        if admission_reasons:
            outcome = (
                "escalate" if "unsupported-semantic-family" in admission_reasons
                else "reject"
            )
            return Decision(claim.claim_id, outcome, admission_reasons,
                            self.VERSION, refs, supporting, refuting)
        if any(item.snapshot_id != snapshot_id for item in evidence):
            return Decision(claim.claim_id, "reject", ("stale-snapshot-evidence",),
                            self.VERSION, refs, supporting, refuting)
        if refuting and supporting:
            return Decision(claim.claim_id, "escalate", ("conflicting-credible-evidence",),
                            self.VERSION, refs, supporting, refuting)
        if refuting:
            return Decision(claim.claim_id, "reject", ("credible-counterevidence",),
                            self.VERSION, refs, supporting, refuting)
        if claim.source_kind == "deterministic-checker":
            return Decision(claim.claim_id, "accept", ("deterministic-rule-and-structure",),
                            self.VERSION, refs, supporting, refuting)
        semantic = any(
            item.kind in self.SEMANTIC_TOOLS and item.result_nonempty
            and claim.claim_id in item.claim_ids and item.direction == "support"
            and self.evidence_relevant(claim, item)
            for item in evidence
        )
        if claim.source_kind == "deterministic-candidate":
            outcome = "accept" if semantic else "escalate"
            reasons = (
                ("deterministic-structure-and-semantic-evidence",)
                if semantic else ("structural-evidence-required",)
            )
            return Decision(claim.claim_id, outcome, reasons, self.VERSION,
                            refs, supporting, refuting)
        high = claim.severity in {"high", "critical"}
        if high and (not semantic or not supporting):
            reasons = []
            if not semantic:
                reasons.append("semantic-evidence-required")
            if not supporting:
                reasons.append("challenge-support-required")
            return Decision(claim.claim_id, "escalate", tuple(reasons), self.VERSION,
                            refs, supporting, refuting)
        if not semantic:
            return Decision(claim.claim_id, "escalate", ("structural-evidence-required",),
                            self.VERSION, refs, supporting, refuting)
        if any(item.verdict == "insufficient" for item in challenge_values) and high:
            return Decision(claim.claim_id, "escalate", ("auditor-insufficient",),
                            self.VERSION, refs, supporting, refuting)
        return Decision(claim.claim_id, "accept", ("evidence-policy-satisfied",),
                        self.VERSION, refs, supporting, refuting)


def evidence_from_record(record: dict, snapshot_id: str, producer_run_id: str) -> Evidence:
    tool = str(record.get("tool") or record.get("source_tool") or "changed-line")
    kind = "locator" if tool in {"changed_line", "changed-line", "search_diff", "read_diff"} else tool
    result = record.get("excerpt", record.get("result", ""))
    actual_snapshot = str(record.get("snapshot_id") or snapshot_id)
    actual_producer = str(record.get("producer_run_id") or producer_run_id)
    digest = _digest({
        "snapshot": actual_snapshot, "path": record.get("path", ""),
        "line": int(record.get("line", 0) or 0), "tool": tool,
        "excerpt": result,
        "producer": actual_producer,
    }, 24)
    return Evidence(
        "E:" + digest, actual_snapshot, str(record.get("path", "")),
        int(record.get("line", 0) or 0), kind, tool, actual_producer,
        digest, "tool" if kind != "locator" else "locator",
        result_nonempty=bool(str(result).strip()),
        tool_version=str(record.get("tool_version", "1")),
    )


def artifact_fingerprint(value: Any) -> str:
    return _digest(value, 64)
