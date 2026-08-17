"""Size-based, centrally configured review budgets.

The numbers deliberately live here rather than in prompts or individual
agents: they are soft per-assignment limits and can be tuned by evaluation
without changing the graph topology.
"""
from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class ShardBudget:
    changed_lines: int
    tool_budget: int
    max_steps: int
    max_loop_calls: int

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


class BudgetPolicy:
    """Qwen-style ``base + floor(lines / K)`` assignment budget policy."""

    # A roughly 400-line territory is the large-diff review unit.  The limits
    # are intentionally modest: a budget overrun must return a receipt, not
    # consume the verifier/auditor tail reserve trying to finish exploration.
    TOOL_BASE = 3
    TOOL_LINES_PER_INCREMENT = 100
    TOOL_MIN = 3
    TOOL_MAX = 12
    STEP_BASE = 3
    STEP_LINES_PER_INCREMENT = 160
    STEP_MIN = 3
    STEP_MAX = 8
    LOOP_BASE = 3
    LOOP_LINES_PER_INCREMENT = 200
    LOOP_MIN = 3
    LOOP_MAX = 7
    VERIFIER_BATCH_SIZE = 8
    REVERSE_AUDIT_ROUNDS = 5
    REVERSE_AUDIT_DRY_ROUNDS = 2

    # Kept for verifier/auditor/final coverage work.  A caller can override
    # these fields through the existing ``agent_budget`` dictionary.
    TAIL_RESERVE = {"agent_runs": 2, "llm_calls": 3, "tool_calls": 4}

    @staticmethod
    def _bounded(base: int, lines: int, divisor: int, lower: int, upper: int) -> int:
        return max(lower, min(upper, base + max(0, int(lines)) // divisor))

    def for_shard(self, changed_lines: int) -> ShardBudget:
        return ShardBudget(
            changed_lines=max(0, int(changed_lines)),
            tool_budget=self._bounded(self.TOOL_BASE, changed_lines, self.TOOL_LINES_PER_INCREMENT,
                                      self.TOOL_MIN, self.TOOL_MAX),
            max_steps=self._bounded(self.STEP_BASE, changed_lines, self.STEP_LINES_PER_INCREMENT,
                                    self.STEP_MIN, self.STEP_MAX),
            max_loop_calls=self._bounded(self.LOOP_BASE, changed_lines, self.LOOP_LINES_PER_INCREMENT,
                                         self.LOOP_MIN, self.LOOP_MAX),
        )

    def tail_reserve(self, limits: Dict[str, int]) -> Dict[str, int]:
        return {
            field: min(max(0, int(limits.get(field, 0))), amount)
            for field, amount in self.TAIL_RESERVE.items()
            if field in limits
        }
