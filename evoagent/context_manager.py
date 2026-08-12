"""Token-aware deterministic context construction for PR review agents."""
from dataclasses import asdict, dataclass, field
import hashlib
import json
import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .context.budget import ContextBudget, TokenCounter, Utf8TokenCounter


RISK_TERMS = {
    "eval", "exec", "shell", "subprocess", "password", "secret", "token",
    "auth", "permission", "sql", "query", "except", "error", "migration",
    "payment", "deserialize", "pickle", "yaml.load", "chmod", "admin",
}


@dataclass(frozen=True)
class ContextBundle:
    text: str
    compressed: bool
    original_tokens: int
    final_tokens: int
    omitted_files: List[str] = field(default_factory=list)
    omitted_hunks: int = 0
    strategy: str = "full-diff"
    source_sha256: str = ""
    omitted_added_lines: int = 0
    omitted_context_lines: int = 0

    def metadata(self) -> Dict[str, Any]:
        value = asdict(self)
        value.pop("text", None)
        return value


@dataclass(frozen=True)
class ManagedContext:
    """The complete dynamic context presented to one agent-loop iteration."""

    text: str
    estimated_tokens: int
    compressed: bool
    diff: Dict[str, Any]
    kept_feedback: int = 0
    kept_memories: int = 0
    kept_observations: int = 0
    dropped_feedback: int = 0
    dropped_memories: int = 0
    dropped_observations: int = 0
    budget: Dict[str, int] = field(default_factory=dict)

    def metadata(self) -> Dict[str, Any]:
        value = asdict(self)
        value.pop("text", None)
        return value


@dataclass
class _Hunk:
    index: int
    path: str
    header: str
    text: str
    score: float


class ContextManager:
    """Build a bounded LLM context while preserving changed-line evidence."""

    def __init__(
        self, max_tokens: int = 12000, reserved_tokens: int = 2500,
        token_counter: TokenCounter = None, soft_compact_ratio: float = .60,
        hard_compact_ratio: float = .80,
    ):
        if max_tokens < 512:
            raise ValueError("context max_tokens must be at least 512")
        if reserved_tokens < 0 or reserved_tokens >= max_tokens:
            raise ValueError("context reserved_tokens must be within the context budget")
        self.max_tokens = max_tokens
        self.reserved_tokens = reserved_tokens
        self.token_counter = token_counter or Utf8TokenCounter()
        self.budget = ContextBudget(
            max_tokens, reserved_tokens, soft_compact_ratio, hard_compact_ratio,
        )

    def estimate_tokens(self, text: str) -> int:
        # A conservative dependency-free estimate for mixed source code and text.
        return self.token_counter.count(text)

    def build(
        self, diff: str, assignment: Dict[str, Any] = None,
        memories: Sequence[Dict[str, Any]] = (),
    ) -> ContextBundle:
        original_tokens = self.estimate_tokens(diff)
        digest = hashlib.sha256(diff.encode("utf-8")).hexdigest()
        available_tokens = self.max_tokens - self.reserved_tokens
        if original_tokens <= available_tokens:
            return ContextBundle(
                diff, False, original_tokens, original_tokens,
                strategy="full-diff", source_sha256=digest,
            )

        assignment = assignment or {}
        terms = self._priority_terms(assignment, memories)
        file_headers, hunks = self._parse(diff, terms)
        byte_budget = max(1024, available_tokens * 4)
        selected: List[_Hunk] = []
        used = 0
        included_headers = set()

        for hunk in sorted(hunks, key=lambda item: (-item.score, item.index)):
            header = file_headers.get(hunk.path, "")
            header_cost = len(header.encode("utf-8")) if hunk.path not in included_headers else 0
            hunk_cost = len(hunk.text.encode("utf-8"))
            if used + header_cost + hunk_cost <= byte_budget:
                selected.append(hunk)
                used += header_cost + hunk_cost
                included_headers.add(hunk.path)
                continue
            remaining = byte_budget - used - header_cost
            compact = self._compact_hunk(hunk, remaining, terms)
            if compact:
                selected.append(_Hunk(
                    hunk.index, hunk.path, hunk.header, compact, hunk.score
                ))
                used += header_cost + len(compact.encode("utf-8"))
                included_headers.add(hunk.path)
            if used >= byte_budget:
                break

        if not selected and hunks:
            first = max(hunks, key=lambda item: (item.score, -item.index))
            selected = [_Hunk(
                first.index, first.path, first.header,
                self._compact_hunk(first, byte_budget, terms) or first.text[:byte_budget],
                first.score,
            )]
            included_headers.add(first.path)

        pieces = []
        current_path = None
        for hunk in sorted(selected, key=lambda item: item.index):
            if hunk.path != current_path:
                pieces.append(file_headers.get(hunk.path, ""))
                current_path = hunk.path
            pieces.append(hunk.text)
        compressed = "".join(pieces).strip() + "\n"
        all_paths = list(file_headers)
        omitted_files = [path for path in all_paths if path not in included_headers]
        omitted_hunks = max(0, len(hunks) - len(selected))
        selected_text = "\n".join(item.text for item in selected)
        omitted_added_lines = max(0, sum(
            1 for line in diff.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        ) - sum(
            1 for line in selected_text.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        ))
        omitted_context_lines = max(0, sum(
            1 for line in diff.splitlines()
            if line.startswith(" ")
        ) - sum(1 for line in selected_text.splitlines() if line.startswith(" ")))
        final_tokens = self.estimate_tokens(compressed)
        return ContextBundle(
            compressed, True, original_tokens, final_tokens,
            omitted_files=omitted_files, omitted_hunks=omitted_hunks,
            strategy="risk-ranked-hunk-compression", source_sha256=digest,
            omitted_added_lines=omitted_added_lines,
            omitted_context_lines=omitted_context_lines,
        )

    def compose(
        self, diff_bundle: ContextBundle, assignment: Dict[str, Any],
        feedback: Sequence[Any] = (), inbox: Sequence[Dict[str, Any]] = (),
        memories: Sequence[Dict[str, Any]] = (),
        observations: Sequence[Dict[str, Any]] = (),
        tools: Sequence[Dict[str, Any]] = (),
        loop_context: Dict[str, Any] = None,
        frozen_context: Dict[str, Any] = None,
        retrieved_context: Sequence[Dict[str, Any]] = (),
    ) -> ManagedContext:
        """Fit all changing loop state into one deterministic token budget.

        The diff owns ``max_tokens - reserved_tokens``. Assignment, tool schemas,
        critique, recalled memories and tool observations share the reserved
        portion. Lower-priority records are dropped instead of silently growing
        the model request on every loop iteration.
        """
        # A small shard releases unused diff capacity to runtime.  This keeps
        # the legacy 9500/2500 default as the minimum guarantee, not a wall.
        allocations = self.budget.allocations(diff_bundle.final_tokens)
        runtime_bytes = max(128, allocations["runtime"] * 4)
        parts: List[str] = []
        used = 0
        was_truncated = False

        section_used: Dict[str, int] = {}
        def append(label: str, value: Any, optional: bool = False, section: str = "shared") -> bool:
            nonlocal used, was_truncated
            rendered = value if isinstance(value, str) else json.dumps(
                value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            line = "%s: %s\n" % (label, rendered.replace("\x00", ""))
            encoded = line.encode("utf-8")
            remaining = runtime_bytes - used
            section_limit = (allocations.get(section, 0) + allocations.get("shared", 0)) * 4
            if section != "shared":
                remaining = min(remaining, max(0, section_limit - section_used.get(section, 0)))
            if len(encoded) <= remaining:
                parts.append(line)
                used += len(encoded)
                section_used[section] = section_used.get(section, 0) + len(encoded)
                return True
            if optional or remaining < 48:
                was_truncated = True
                return False
            clipped = self._truncate_utf8(encoded, remaining)
            if clipped:
                parts.append(clipped.decode("utf-8", errors="ignore") + "\n")
                used += len(clipped) + 1
                section_used[section] = section_used.get(section, 0) + len(clipped) + 1
            was_truncated = True
            return bool(clipped)

        compact_assignment = {
            key: assignment.get(key) for key in (
                "agent", "objective", "files", "risk_domains", "round", "reason",
                "shard_id", "shard_files", "coverage_scope", "pr_map",
            ) if assignment.get(key) not in (None, "", [])
        }
        frozen = dict(frozen_context or {})
        frozen.setdefault("assignment", compact_assignment)
        append("FROZEN_CONTEXT", frozen, section="frozen")
        for tool in tools:
            append("TOOL", {
                "name": tool.get("name"),
                "description": str(tool.get("description", ""))[:240],
                "parameters": tool.get("parameters") or {},
            }, optional=True)

        # Keep only mailbox routing metadata. Message bodies are represented by
        # critique and observations, avoiding duplicated prompt content.
        if inbox:
            append("COLLABORATION", {
                "count": len(inbox),
                "kinds": sorted({str(item.get("kind", "")) for item in inbox}),
                "senders": sorted({str(item.get("sender", "")) for item in inbox}),
            }, optional=True)

        loop_state = dict(loop_context or {})
        active_rounds = list(loop_state.get("active_rounds") or [])
        compressed_history = loop_state.get("compressed_history") or {}
        pinned_evidence = loop_state.get("pinned_evidence") or []
        kept_observations = 0
        for item in active_rounds:
            if append("ACTIVE_ROUND", item, optional=True, section="active"):
                kept_observations += 1
        if compressed_history:
            append("COMPRESSED_HISTORY", compressed_history, optional=True, section="compressed")
        if pinned_evidence:
            append("PINNED_EVIDENCE", pinned_evidence, optional=True, section="frozen")
        # Compatibility path for callers which do not yet supply LoopContext.
        if not loop_context:
            for item in reversed(observations):
                compact = {key: item.get(key) for key in ("step", "tool", "ok", "result", "error")
                           if item.get(key) is not None}
                if append("OBSERVATION", compact, optional=True):
                    kept_observations += 1
        for item in retrieved_context:
            append("RETRIEVED_CONTEXT", item, optional=True, section="retrieved")

        kept_feedback = 0
        for item in feedback:
            if append("CRITIC_FEEDBACK", str(item)[:1200], optional=True):
                kept_feedback += 1

        kept_memories = 0
        for item in memories:
            compact = {
                "scope": item.get("scope"), "kind": item.get("kind"),
                "content": str(item.get("content", ""))[:1200],
                "score": item.get("recall_score"),
            }
            if append("MEMORY", compact, optional=True):
                kept_memories += 1

        runtime_text = "".join(parts)
        text = runtime_text + "DIFF_CONTEXT:\n" + diff_bundle.text
        # ``build`` and the reserved budget should already guarantee this. The
        # final guard protects callers that construct ContextBundle themselves.
        maximum_bytes = self.max_tokens * 4
        encoded = text.encode("utf-8")
        if len(encoded) > maximum_bytes:
            text = self._truncate_utf8(encoded, maximum_bytes).decode(
                "utf-8", errors="ignore"
            )
            was_truncated = True
        final_tokens = self.estimate_tokens(text)
        return ManagedContext(
            text=text, estimated_tokens=final_tokens,
            compressed=bool(diff_bundle.compressed or was_truncated),
            diff=diff_bundle.metadata(),
            kept_feedback=kept_feedback, kept_memories=kept_memories,
            kept_observations=kept_observations,
            dropped_feedback=max(0, len(feedback) - kept_feedback),
            dropped_memories=max(0, len(memories) - kept_memories),
            dropped_observations=max(0, len(observations) - kept_observations),
            budget={name: value for name, value in allocations.items()},
        )

    @staticmethod
    def _truncate_utf8(value: bytes, limit: int) -> bytes:
        if limit <= 0:
            return b""
        clipped = value[:limit]
        while clipped:
            try:
                clipped.decode("utf-8")
                return clipped
            except UnicodeDecodeError:
                clipped = clipped[:-1]
        return b""

    @staticmethod
    def _priority_terms(
        assignment: Dict[str, Any], memories: Sequence[Dict[str, Any]],
    ) -> set:
        values = list(RISK_TERMS)
        values.extend(str(item) for item in assignment.get("risk_domains", []))
        values.extend(str(assignment.get("objective", "")).lower().split())
        for memory in memories[:20]:
            values.extend(str(memory.get("content", "")).lower().split()[:40])
            values.extend(str(memory.get("kind", "")).lower().split())
        return {
            re.sub(r"[^a-z0-9_.-]", "", value.lower())
            for value in values if len(value) >= 3
        }

    def _parse(self, diff: str, terms: set) -> Tuple[Dict[str, str], List[_Hunk]]:
        lines = diff.splitlines(True)
        files: List[Tuple[str, List[str]]] = []
        current: List[str] = []
        current_path = "unknown"
        for line in lines:
            if line.startswith("--- ") and current:
                files.append((current_path, current))
                current = []
                current_path = "unknown"
            current.append(line)
            if line.startswith("+++ "):
                raw = line[4:].strip()
                current_path = raw[2:] if raw.startswith("b/") else raw
        if current:
            files.append((current_path, current))

        headers: Dict[str, str] = {}
        hunks: List[_Hunk] = []
        index = 0
        for path, block in files:
            positions = [i for i, line in enumerate(block) if line.startswith("@@")]
            if not positions:
                headers[path] = "".join(block[:2])
                text = "".join(block)
                hunks.append(_Hunk(index, path, "", text, self._score(path, text, terms)))
                index += 1
                continue
            headers[path] = "".join(block[:positions[0]])
            for pos_index, start in enumerate(positions):
                end = positions[pos_index + 1] if pos_index + 1 < len(positions) else len(block)
                text = "".join(block[start:end])
                hunks.append(_Hunk(
                    index, path, block[start].rstrip(), text,
                    self._score(path, text, terms),
                ))
                index += 1
        return headers, hunks

    @staticmethod
    def _score(path: str, text: str, terms: set) -> float:
        lowered = (path + "\n" + text).lower()
        score = sum(2.0 for term in terms if term and term in lowered)
        score += min(8.0, sum(
            1 for line in text.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        ) * 0.25)
        if any(token in path.lower() for token in ("auth", "security", "payment", "migration")):
            score += 6.0
        return score

    @staticmethod
    def _compact_hunk(hunk: _Hunk, byte_budget: int, terms: set) -> str:
        if byte_budget < 128:
            return ""
        lines = hunk.text.splitlines(True)
        if not lines:
            return ""
        priority = {0}
        optional = set()
        for index, line in enumerate(lines):
            lowered = line.lower()
            risky = any(term in lowered for term in terms if term)
            added = line.startswith("+") and not line.startswith("+++")
            if risky:
                priority.update({
                    max(0, index - 1), index, min(len(lines) - 1, index + 1)
                })
            elif added:
                optional.add(index)
        selected = set(priority)
        estimated = sum(len(lines[index].encode("utf-8")) for index in selected)
        for index in sorted(optional):
            cost = len(lines[index].encode("utf-8"))
            if estimated + cost > max(128, int(byte_budget * 0.85)):
                break
            selected.add(index)
            estimated += cost
        output = []
        previous = -2
        used = 0
        for index in sorted(selected):
            if index - previous > 1 and output:
                marker = " ... [lower-priority diff content omitted; changed lines may be included] ...\n"
                if used + len(marker.encode("utf-8")) <= byte_budget:
                    output.append(marker)
                    used += len(marker.encode("utf-8"))
            encoded = lines[index].encode("utf-8")
            if used + len(encoded) > byte_budget:
                break
            output.append(lines[index])
            used += len(encoded)
            previous = index
        if selected and max(selected) < len(lines) - 1:
            marker = " ... [lower-priority diff content omitted; changed lines may be included] ...\n"
            if used + len(marker.encode("utf-8")) <= byte_budget:
                output.append(marker)
        return "".join(output)


def render_memories(memories: Iterable[Dict[str, Any]], max_chars: int = 5000) -> str:
    """Render recalled memory as untrusted, compact runtime context."""
    lines = []
    used = 0
    for item in memories:
        status = (item.get("metadata") or {}).get("status", "")
        state = " status=%s" % status if status and status != "active" else ""
        line = "[%s/%s%s] %s" % (
            item.get("scope", "memory"), item.get("kind", "note"), state,
            str(item.get("content", "")).replace("\n", " ")[:1000],
        )
        if used + len(line) + 1 > max_chars:
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)
