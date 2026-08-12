"""Coverage-aware, deterministic large-PR review sharding."""
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from ..diff_parser import ParsedDiff, parse_unified_diff
from .budget import TokenCounter, Utf8TokenCounter
from .pr_map import PRContextMap


@dataclass(frozen=True)
class ReviewShard:
    shard_id: str
    files: List[str]
    diff: str
    parsed: ParsedDiff
    estimated_tokens: int
    changed_lines: int
    hunk_count: int

    def identity(self) -> Dict[str, object]:
        return {"id": self.shard_id, "files": self.files, "changed_lines": self.changed_lines}


class ShardPlanner:
    def __init__(
        self, diff_budget_tokens: int = 9500, file_threshold: int = 12,
        changed_line_threshold: int = 1200, token_counter: TokenCounter = None,
    ):
        self.diff_budget_tokens = max(256, diff_budget_tokens)
        self.file_threshold = max(1, file_threshold)
        self.changed_line_threshold = max(1, changed_line_threshold)
        self.counter = token_counter or Utf8TokenCounter()

    def needs_sharding(self, diff: str, pr_map: PRContextMap) -> bool:
        return (
            self.counter.count(diff) > self.diff_budget_tokens
            or len(pr_map.files) >= self.file_threshold
            or pr_map.total_added_lines >= self.changed_line_threshold
        )

    def plan(self, diff: str, parsed: ParsedDiff, pr_map: PRContextMap) -> List[ReviewShard]:
        if not self.needs_sharding(diff, pr_map):
            return [self._make_shard("full", list(parsed.files), diff)]
        blocks = self._file_blocks(diff)
        ordered_files = [item.path for item in pr_map.files]
        # File-count-triggered sharding remains meaningful when one directory
        # or module contains many small files.  A semantic Review Unit may
        # inform ordering, but may not defeat the configured capacity bound.
        max_files = max(1, self.file_threshold // 2) if len(pr_map.files) >= self.file_threshold else 10 ** 9
        # Keep the same top-level module together whenever its combined content fits.
        units: List[List[str]] = []
        groups: Dict[str, List[str]] = {}
        for path in ordered_files:
            groups.setdefault(self._module_key(path), []).append(path)
        for paths in groups.values():
            joined = "".join(blocks.get(path, "") for path in paths)
            if self.counter.count(joined) <= self.diff_budget_tokens and len(paths) <= max_files:
                units.append(paths)
            else:
                # Preserve directory/module order while binning a large unit.
                # Individual files remain atomic so later local compression is
                # the only fallback when one file itself exceeds the budget.
                for start in range(0, len(paths), max_files):
                    units.append(paths[start:start + max_files])
        shards: List[ReviewShard] = []
        current: List[str] = []
        current_text = ""
        # File-count-triggered sharding is meaningful even when a small source
        # diff happens to fit in tokens: it limits unrelated files per review.
        for unit in units:
            text = "".join(blocks.get(path, "") for path in unit)
            if current and (self.counter.count(current_text + text) > self.diff_budget_tokens
                            or len(current) + len(unit) > max_files):
                shards.append(self._make_shard("S%02d" % (len(shards) + 1), current, current_text))
                current, current_text = [], ""
            current.extend(unit)
            current_text += text
        if current:
            shards.append(self._make_shard("S%02d" % (len(shards) + 1), current, current_text))
        # The fallback is deliberately explicit: even an odd diff file remains reviewable.
        covered = {path for shard in shards for path in shard.files}
        for path in parsed.files:
            if path not in covered:
                shards.append(self._make_shard("S%02d" % (len(shards) + 1), [path], blocks.get(path, "")))
        return shards

    def _make_shard(self, shard_id: str, files: List[str], diff: str) -> ReviewShard:
        parsed = parse_unified_diff(diff)
        return ReviewShard(
            shard_id, list(files), diff, parsed, self.counter.count(diff),
            len(parsed.added_lines), max(1, diff.count("@@ ")),
        )

    @staticmethod
    def _module_key(path: str) -> str:
        pieces = path.replace("\\", "/").split("/")
        return "/".join(pieces[:2]) if len(pieces) > 1 else pieces[0]

    @staticmethod
    def _file_blocks(diff: str) -> Dict[str, str]:
        blocks: Dict[str, List[str]] = {}
        current = ""
        for line in diff.splitlines(True):
            if line.startswith("--- "):
                current = ""
            if line.startswith("+++ "):
                raw = line[4:].strip()
                current = raw[2:] if raw.startswith("b/") else raw
                if current != "/dev/null":
                    blocks.setdefault(current, [])
            if current and current != "/dev/null":
                blocks[current].append(line)
        return {path: "".join(lines) for path, lines in blocks.items()}
