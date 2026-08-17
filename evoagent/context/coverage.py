"""Structured coverage receipts emitted by bounded review assignments."""
from dataclasses import asdict, dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class CoverageReceipt:
    assignment_id: str
    shard_id: str
    covered_files: List[str]
    uncovered_files: List[str] = field(default_factory=list)
    required_hunks: List[str] = field(default_factory=list)
    covered_hunks: List[str] = field(default_factory=list)
    uncovered_hunks: List[str] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    budget_gaps: List[str] = field(default_factory=list)
    status: str = "complete"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)
