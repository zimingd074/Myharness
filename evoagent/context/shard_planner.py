"""Coverage-aware, module-first large-PR review sharding."""
from dataclasses import dataclass
import re
from typing import Dict, List

from ..diff_parser import ParsedDiff, parse_unified_diff
from .budget import TokenCounter, Utf8TokenCounter
from .pr_map import PRContextMap
from .risk_priority import DiffRiskScanner, RiskHunk


@dataclass(frozen=True)
class ReviewShard:
    shard_id: str
    files: List[str]
    diff: str
    parsed: ParsedDiff
    estimated_tokens: int
    changed_lines: int
    hunk_count: int
    risk_hunks: List[RiskHunk]

    def identity(self) -> Dict[str, object]:
        return {"id": self.shard_id, "files": self.files, "changed_lines": self.changed_lines}


class ShardPlanner:
    """Keep fitting modules intact, then split large modules on diff boundaries."""

    def __init__(
        self, diff_budget_tokens: int = 9500, file_threshold: int = 12,
        changed_line_threshold: int = 1200, token_counter: TokenCounter = None,
        module_changed_line_capacity: int = 400,
    ):
        self.diff_budget_tokens = max(256, diff_budget_tokens)
        self.file_threshold = max(1, file_threshold)
        self.changed_line_threshold = max(1, changed_line_threshold)
        self.module_changed_line_capacity = max(40, module_changed_line_capacity)
        self.counter = token_counter or Utf8TokenCounter()
        self.risk_scanner = DiffRiskScanner()

    def needs_sharding(self, diff: str, pr_map: PRContextMap) -> bool:
        return (
            self.counter.count(diff) > self.diff_budget_tokens
            or len(pr_map.files) >= self.file_threshold
            or pr_map.total_added_lines + pr_map.total_deleted_lines >= self.changed_line_threshold
        )

    def plan(self, diff: str, parsed: ParsedDiff, pr_map: PRContextMap) -> List[ReviewShard]:
        if not self.needs_sharding(diff, pr_map):
            return [self._make_shard("full", list(parsed.files), diff)]
        blocks = self._file_blocks(diff)
        paths_by_module: Dict[str, List[str]] = {}
        for item in pr_map.files:
            paths_by_module.setdefault(self._module_key(item.path), []).append(item.path)
        fragments: List[tuple] = []
        for paths in paths_by_module.values():
            module_text = "".join(blocks.get(path, "") for path in paths)
            module_file_capacity = max(1, self.file_threshold // 2)
            if (self._changed_lines(module_text) <= self.module_changed_line_capacity and self._fits(module_text)
                    and len(paths) <= module_file_capacity):
                fragments.append((paths, module_text))
                continue
            # A too-large module is partitioned within itself. File and hunk
            # boundaries are preferred; only a giant hunk needs a function
            # boundary fallback.
            for path in paths:
                fragments.extend(self._split_file(path, blocks.get(path, "")))
        shards: List[ReviewShard] = []
        current_files: List[str] = []
        current_text = ""
        for files, text in fragments:
            if not text:
                continue
            if current_text and (not self._fits(current_text + text)
                                 or self._changed_lines(current_text + text) > self.module_changed_line_capacity
                                 or len(set(current_files).union(files)) > max(1, self.file_threshold // 2)):
                shards.append(self._make_shard("S%02d" % (len(shards) + 1), current_files, current_text))
                current_files, current_text = [], ""
            current_files.extend(path for path in files if path not in current_files)
            current_text += text
        if current_text:
            shards.append(self._make_shard("S%02d" % (len(shards) + 1), current_files, current_text))
        # Coverage is an invariant, including unusual/empty file blocks.
        covered = {path for shard in shards for path in shard.files}
        for path in parsed.files:
            if path not in covered:
                shards.append(self._make_shard("S%02d" % (len(shards) + 1), [path], blocks.get(path, "")))
        return shards or [self._make_shard("full", list(parsed.files), diff)]

    def _fits(self, text: str) -> bool:
        return self.counter.count(text) <= self.diff_budget_tokens

    def _split_file(self, path: str, block: str) -> List[tuple]:
        if not block:
            return [([path], block)]
        header, hunks = self._hunks(block)
        if not hunks:
            return [([path], block)]
        fragments, current = [], ""
        for hunk in hunks:
            for piece in self._split_oversized_hunk(header, hunk):
                body = piece[len(header):] if piece.startswith(header) else piece
                candidate = current + body if current else header + body
                if current and self._changed_lines(candidate) > self.module_changed_line_capacity:
                    fragments.append(([path], current))
                    current = header + body
                else:
                    current = candidate
        if current:
            fragments.append(([path], current))
        return fragments

    def _split_oversized_hunk(self, header: str, hunk: str) -> List[str]:
        if self._changed_lines(hunk) <= self.module_changed_line_capacity:
            return [header + hunk]
        lines = hunk.splitlines(True)
        hunk_header, body = lines[:1], lines[1:]
        chunks: List[List[str]] = []
        current: List[str] = []
        changed = 0
        boundary = re.compile(r"^[+ ]\s*(?:async\s+def|def|class|function|func|export\s+(?:async\s+)?function)\b")
        for line in body:
            line_changed = int(line.startswith(("+", "-")) and not line.startswith(("+++", "---")))
            if current and changed >= self.module_changed_line_capacity and (
                # Prefer function boundaries; a single giant function has no
                # safe declaration boundary, so keep its shard at capacity.
                boundary.match(line) or changed >= self.module_changed_line_capacity
            ):
                chunks.append(current)
                current, changed = [], 0
            current.append(line)
            changed += line_changed
        if current:
            chunks.append(current)
        return [header + "".join(hunk_header + item) for item in chunks]

    @staticmethod
    def _hunks(block: str) -> tuple:
        lines = block.splitlines(True)
        starts = [index for index, line in enumerate(lines) if line.startswith("@@ ")]
        if not starts:
            return block, []
        header = "".join(lines[:starts[0]])
        return header, ["".join(lines[start:end]) for start, end in zip(starts, starts[1:] + [len(lines)])]

    @staticmethod
    def _changed_lines(text: str) -> int:
        return sum(1 for line in text.splitlines() if line.startswith(("+", "-"))
                   and not line.startswith(("+++", "---")))

    def _make_shard(self, shard_id: str, files: List[str], diff: str) -> ReviewShard:
        parsed = parse_unified_diff(diff)
        return ReviewShard(
            shard_id, list(dict.fromkeys(files)), diff, parsed, self.counter.count(diff),
            self._changed_lines(diff), max(1, diff.count("@@ ")),
            self.risk_scanner.scan(diff, shard_id),
        )

    @staticmethod
    def _module_key(path: str) -> str:
        pieces = path.replace("\\", "/").split("/")
        # Top-level directory is the stable module boundary for a diff-only
        # planner; deeper paths stay together until that module exceeds its
        # explicit capacity.
        return pieces[0]

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
