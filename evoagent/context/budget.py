"""Token estimation and deterministic context budget policy."""
from dataclasses import dataclass
from typing import Protocol


class TokenCounter(Protocol):
    """Pluggable token estimator; providers may supply an exact tokenizer later."""

    def count(self, text: str) -> int:
        ...


class Utf8TokenCounter:
    """Dependency-free conservative estimator retained for backwards compatibility."""

    def count(self, text: str) -> int:
        return max(1, (len(text.encode("utf-8")) + 3) // 4)


@dataclass(frozen=True)
class ContextBudget:
    """Minimum quotas plus an elastic pool shared by local diff and loop state."""

    max_tokens: int = 12000
    reserved_tokens: int = 2500
    soft_compact_ratio: float = 0.60
    hard_compact_ratio: float = 0.80
    frozen_min_tokens: int = 600
    active_min_tokens: int = 900
    compressed_min_tokens: int = 400
    retrieved_min_tokens: int = 400

    @property
    def diff_tokens(self) -> int:
        return self.max_tokens - self.reserved_tokens

    def runtime_tokens(self, local_diff_tokens: int) -> int:
        """Let runtime borrow unused local-diff capacity without exceeding max_tokens."""
        return max(self.reserved_tokens, self.max_tokens - max(0, local_diff_tokens))

    def compose_limit(self, local_diff_tokens: int) -> int:
        """Runtime's elastic limit after a shard has claimed its actual space."""
        return min(self.max_tokens, max(self.reserved_tokens, self.runtime_tokens(local_diff_tokens)))

    def allocations(self, local_diff_tokens: int) -> dict:
        """Minimum runtime reservations plus elastic space left by this shard."""
        runtime = self.compose_limit(local_diff_tokens)
        minimums = {
            "frozen": self.frozen_min_tokens, "active": self.active_min_tokens,
            "compressed": self.compressed_min_tokens, "retrieved": self.retrieved_min_tokens,
        }
        floor = sum(minimums.values())
        # A small context cannot honour every floor; preserve Frozen first.
        if floor > runtime:
            remaining = runtime
            for name in ("frozen", "active", "compressed", "retrieved"):
                minimums[name] = min(minimums[name], remaining)
                remaining -= minimums[name]
            return {**minimums, "shared": 0, "runtime": runtime}
        return {**minimums, "shared": runtime - floor, "runtime": runtime}

    def should_compact(self, used_tokens: int, hard: bool = False) -> bool:
        ratio = self.hard_compact_ratio if hard else self.soft_compact_ratio
        return used_tokens >= int(self.max_tokens * ratio)
