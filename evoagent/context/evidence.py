"""Task-scoped, traceable evidence shared by collaboration stages."""
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List


@dataclass
class LedgerEvidence:
    evidence_id: str
    path: str
    line: int
    excerpt: str
    source_tool: str
    agent: str
    shard_id: str
    importance: float = .5
    pinned: bool = False
    finding_ids: List[str] = field(default_factory=list)

    def compact(self) -> dict:
        value = asdict(self)
        value["excerpt"] = self.excerpt[:300]
        return value


class EvidenceLedger:
    """Keeps stable locators, not unbounded raw observations, for one review task."""

    def __init__(self):
        self._items: Dict[str, LedgerEvidence] = {}

    def record(self, values: Iterable[dict]) -> List[str]:
        added = []
        for value in values:
            evidence_id = str(value.get("id") or value.get("evidence_id") or "")
            if not evidence_id:
                continue
            record = LedgerEvidence(
                evidence_id=evidence_id, path=str(value.get("path", "")),
                line=int(value.get("line", 0) or 0), excerpt=str(value.get("excerpt", "")),
                source_tool=str(value.get("tool") or value.get("source_tool") or ""),
                agent=str(value.get("agent", "")), shard_id=str(value.get("shard_id", "")),
                importance=float(value.get("importance", .5) or .5), pinned=bool(value.get("pinned")),
                finding_ids=list(value.get("finding_ids") or value.get("referenced_by_finding") or []),
            )
            self._items[evidence_id] = record
            added.append(evidence_id)
        return added

    def link(self, finding_id: str, refs: Iterable[str], path: str = "", line: int = 0) -> List[str]:
        selected = [ref for ref in refs if ref in self._items]
        if not selected and path and line:
            selected = [item.evidence_id for item in self._items.values()
                        if item.path == path and item.line == line]
        for evidence_id in selected:
            item = self._items[evidence_id]
            item.pinned = True
            if finding_id not in item.finding_ids:
                item.finding_ids.append(finding_id)
        return selected

    def valid_for(self, finding_id: str, path: str, line: int) -> bool:
        return any(item.pinned and finding_id in item.finding_ids and item.path == path and item.line == line
                   for item in self._items.values())

    def record_changed_line(self, finding_id: str, path: str, line: int, excerpt: str) -> str:
        evidence_id = "F:" + finding_id
        self.record([{"id": evidence_id, "path": path, "line": line, "excerpt": excerpt,
                      "tool": "changed-line", "agent": "verification", "shard_id": ""}])
        self.link(finding_id, [evidence_id], path, line)
        return evidence_id

    def compact(self) -> List[dict]:
        return [item.compact() for item in self._items.values()]

    def pinned(self) -> List[dict]:
        return [item.compact() for item in self._items.values() if item.pinned]
