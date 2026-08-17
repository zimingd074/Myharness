"""Compact deterministic PR maps built from unified diffs."""
from dataclasses import asdict, dataclass, field
import re
from typing import Dict, Iterable, List

from ..diff_parser import ParsedDiff


SYMBOL = re.compile(
    r"^\+\s*(?:async\s+def|def|class|function|interface|type)\s+([A-Za-z_][\w.]*)"
)
IMPORT = re.compile(r"^\+\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+)|require\(['\"]([^'\"]+))")
RISK = (
    "auth", "permission", "token", "secret", "password", "payment", "migration", "sql",
    "eval", "exec", "shell", "subprocess", "pickle", "yaml", "deserialize",
    "except", "print",
)


@dataclass(frozen=True)
class PRFileMap:
    path: str
    added_lines: int
    deleted_lines: int
    symbols: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    risk_tags: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class PRContextMap:
    files: List[PRFileMap]
    total_added_lines: int
    total_deleted_lines: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def compact(self) -> Dict[str, object]:
        return {
            "files": [
                {"path": item.path, "added": item.added_lines, "deleted": item.deleted_lines,
                 "symbols": item.symbols[:8], "imports": item.imports[:5],
                 "risk": item.risk_tags}
                for item in self.files
            ],
            "totals": {"added": self.total_added_lines, "deleted": self.total_deleted_lines},
        }


def build_pr_context_map(diff: str, parsed: ParsedDiff = None) -> PRContextMap:
    """Extract structural hints only; no model call or source checkout is required."""
    values: Dict[str, Dict[str, object]] = {}
    current = ""
    for raw in diff.splitlines():
        if raw.startswith("+++ "):
            current = raw[4:].strip()
            if current.startswith("b/"):
                current = current[2:]
            if current == "/dev/null":
                current = ""
            if current:
                values.setdefault(current, {"added": 0, "deleted": 0, "symbols": [], "imports": [], "risk": []})
            continue
        if not current:
            continue
        entry = values[current]
        if raw.startswith("+") and not raw.startswith("+++"):
            entry["added"] = int(entry["added"]) + 1
            match = SYMBOL.match(raw)
            if match and match.group(1) not in entry["symbols"]:
                entry["symbols"].append(match.group(1))
            imported = IMPORT.match(raw)
            if imported:
                name = next((part for part in imported.groups() if part), "")
                if name and name not in entry["imports"]:
                    entry["imports"].append(name)
            lowered = raw.lower()
            for tag in RISK:
                if tag in lowered and tag not in entry["risk"]:
                    entry["risk"].append(tag)
        elif raw.startswith("-") and not raw.startswith("---"):
            entry["deleted"] = int(entry["deleted"]) + 1

    # ParsedDiff preserves files that contain unusual headers but no recognizable body.
    for path in (parsed.files if parsed else []):
        values.setdefault(path, {"added": 0, "deleted": 0, "symbols": [], "imports": [], "risk": []})
    files = []
    for path, entry in values.items():
        lowered = path.lower()
        tags = sorted(set(entry.get("risk", [])) | {term for term in RISK if term in lowered})
        files.append(PRFileMap(
            path=path, added_lines=int(entry["added"]), deleted_lines=int(entry["deleted"]),
            symbols=list(entry["symbols"]), imports=list(entry["imports"]), risk_tags=tags,
        ))
    files.sort(key=lambda item: item.path)
    return PRContextMap(
        files=files,
        total_added_lines=sum(item.added_lines for item in files),
        total_deleted_lines=sum(item.deleted_lines for item in files),
    )
