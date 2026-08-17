"""Fast, deterministic hunk priorities for shard-local exploration."""
from dataclasses import asdict, dataclass
import re
from typing import Dict, List


HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
SYMBOL = re.compile(r"^[+]\s*(?:async\s+def|def|class|function|interface|type)\s+([A-Za-z_]\w*)")
SECURITY = re.compile(
    r"(?i)auth|permission|access|token|secret|password|sql|eval|exec|shell|"
    r"subprocess|pickle|yaml|deserialize|redirect"
)
RELIABILITY = re.compile(
    r"(?i)retry|timeout|transaction|lock|async|await|queue|cache|except|error|"
    r"rollback|idempot"
)
GUARD = re.compile(r"(?i)auth|permission|access|guard|validate|check")
CONTRACT = re.compile(r"(?i)\b(?:def|function|interface|type|schema|route|export|import)\b")


@dataclass(frozen=True)
class RiskHunk:
    """Metadata only: source text stays in the owning ``ReviewShard``."""

    hunk_id: str
    path: str
    start_line: int
    end_line: int
    tags: List[str]
    score: int
    symbols: List[str]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class DiffRiskScanner:
    """One linear diff pass; never emits findings or calls a model."""

    def scan(self, diff: str, shard_id: str = "") -> List[RiskHunk]:
        values = []
        path, start, length, body, ordinal = "", 0, 0, [], 0

        def finish() -> None:
            if not path or not body:
                return
            added = "\n".join(line[1:] for line in body if line.startswith("+") and not line.startswith("+++"))
            deleted = "\n".join(line[1:] for line in body if line.startswith("-") and not line.startswith("---"))
            text = added + "\n" + deleted
            tags, score = [], 0
            if deleted and GUARD.search(deleted):
                tags.append("removed_guard")
                score += 5
            if SECURITY.search(text):
                tags.append("security")
                score += 3
            if RELIABILITY.search(text):
                tags.append("reliability")
                score += 2
            if CONTRACT.search(added):
                tags.append("contract_change")
                score += 2
            symbols = []
            for line in body:
                match = SYMBOL.match(line)
                if match and match.group(1) not in symbols:
                    symbols.append(match.group(1))
            if len(tags) > 1:
                score += 1
            end = start + max(0, length - 1)
            values.append(RiskHunk(
                "%s:%s:%d" % (shard_id or "diff", path, ordinal), path, start, end,
                tags, score, symbols,
            ))

        for raw in diff.splitlines():
            if raw.startswith("+++ "):
                finish()
                path = raw[4:].strip()
                path = path[2:] if path.startswith("b/") else path
                if path == "/dev/null":
                    path = ""
                start, length, body, ordinal = 0, 0, [], 0
                continue
            match = HUNK.match(raw)
            if match:
                finish()
                ordinal += 1
                start = int(match.group(1))
                length = int(match.group(2) or 1)
                body = [raw]
            elif body:
                body.append(raw)
        finish()
        return values

    def select(self, diff: str, hunk_ids: List[str], shard_id: str = "") -> str:
        """Return only selected original hunk text for a coverage repair."""
        wanted = set(hunk_ids)
        values, headers, body = [], [], []
        path, ordinal = "", 0

        def finish() -> None:
            if body and "%s:%s:%d" % (shard_id or "diff", path, ordinal) in wanted:
                values.append("".join(headers + body))

        for raw in diff.splitlines(True):
            if raw.startswith("--- "):
                finish()
                headers, body, path, ordinal = [raw], [], "", 0
            elif raw.startswith("+++ "):
                finish()
                headers.append(raw)
                path = raw[4:].strip()
                path = path[2:] if path.startswith("b/") else path
                if path == "/dev/null":
                    path = ""
                body = []
            elif raw.startswith("@@ "):
                finish()
                ordinal += 1
                body = [raw]
            elif body:
                body.append(raw)
        finish()
        return "".join(values)
