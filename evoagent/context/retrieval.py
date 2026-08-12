"""Bounded repository-context retrieval with no shell access."""
from abc import ABC, abstractmethod
import re
from typing import Dict, List, Optional


MAX_RANGE_LINES = 400
MAX_RESULT_BYTES = 16000


class RepositorySnapshotProvider(ABC):
    """Task-scoped read-only repository snapshot, never a host filesystem handle."""

    @abstractmethod
    def paths(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def read(self, path: str) -> Optional[str]:
        raise NotImplementedError

    def available(self) -> bool:
        return True


class InMemorySnapshotProvider(RepositorySnapshotProvider):
    def __init__(self, files: Dict[str, str] = None, allowed_paths: List[str] = None):
        self.files = {self._path(path): str(value) for path, value in (files or {}).items()}
        self.allowed = set(self._path(path) for path in (allowed_paths or self.files))

    def paths(self) -> List[str]:
        return sorted(self.allowed)

    def read(self, path: str) -> Optional[str]:
        safe = self._path(path)
        return self.files.get(safe) if safe in self.allowed else None

    @staticmethod
    def _path(path: str) -> str:
        value = str(path).replace("\\", "/").lstrip("/")
        if not value or ".." in value.split("/"):
            raise ValueError("invalid repository path")
        return value


class UnavailableSnapshotProvider(InMemorySnapshotProvider):
    def available(self) -> bool:
        return False


class GitHubSnapshotProvider(RepositorySnapshotProvider):
    """Lazy adapter for an authorized GitHub PR head and bounded repository index."""

    def __init__(self, client, repository: str, ref: str, allowed_paths: List[str], allow_related: bool = False):
        self.client = client
        self.repository = repository
        self.ref = ref
        self.allowed = sorted({InMemorySnapshotProvider._path(path) for path in allowed_paths})
        self.allow_related = allow_related
        self._paths_loaded = False
        self.cache: Dict[str, str] = {}

    def paths(self) -> List[str]:
        if self.allow_related and not self._paths_loaded:
            try:
                related = self.client.list_repository_paths(self.repository, self.ref)
                self.allowed = sorted(set(self.allowed).union(
                    InMemorySnapshotProvider._path(path) for path in related
                ))
            finally:
                self._paths_loaded = True
        return list(self.allowed)

    def read(self, path: str) -> Optional[str]:
        safe = InMemorySnapshotProvider._path(path)
        if safe not in self.paths():
            return None
        if safe not in self.cache:
            self.cache[safe] = str(
                self.client.get_file(self.repository, safe, self.ref).get("decoded_content", "")
            )
        return self.cache[safe]


class RepositoryRetrieval:
    def __init__(self, snapshot: RepositorySnapshotProvider = None, diff: str = ""):
        self.snapshot = snapshot or UnavailableSnapshotProvider()
        self.diff = diff
        self.calls = 0
        self.retrieved_bytes = 0

    def read_file(self, path: str, start_line: int, end_line: int) -> Dict[str, object]:
        self.calls += 1
        if not self.snapshot.available():
            return {"available": False, "reason": "repository snapshot is unavailable"}
        if end_line < start_line or end_line - start_line + 1 > MAX_RANGE_LINES:
            raise ValueError("read_file range must be positive and at most %d lines" % MAX_RANGE_LINES)
        content = self.snapshot.read(path)
        if content is None:
            return {"found": False, "path": path}
        rows = content.splitlines()
        selected = rows[start_line - 1:end_line]
        rendered = self._bound("\n".join(selected))
        self._charge(rendered)
        return {"found": True, "path": path, "start_line": start_line,
                "end_line": min(end_line, len(rows)), "content": rendered}

    def read_diff(self, path: str = "", cursor: int = 0, limit: int = 120) -> Dict[str, object]:
        self.calls += 1
        limit = max(1, min(int(limit), MAX_RANGE_LINES))
        blocks = self._diff_blocks()
        paths = [path] if path else sorted(blocks)
        rows = []
        for item in paths:
            rows.extend(blocks.get(item, "").splitlines())
        start = max(0, int(cursor))
        selected = rows[start:start + limit]
        bounded = self._bound("\n".join(selected))
        bounded_rows = bounded.splitlines()
        self._charge(bounded)
        delivered = len(bounded_rows)
        next_cursor = start + delivered
        return {"path": path or None, "cursor": start, "lines": bounded_rows,
                "truncated": len(bounded.encode("utf-8")) < len("\n".join(selected).encode("utf-8")),
                "next_cursor": next_cursor if next_cursor < len(rows) else None}

    def grep_repo(self, query: str, path: str = "", cursor: int = 0, limit: int = 20) -> Dict[str, object]:
        self.calls += 1
        if not self.snapshot.available():
            return {"available": False, "reason": "repository snapshot is unavailable"}
        needle = str(query).strip().lower()
        if not needle:
            raise ValueError("grep_repo query is required")
        candidates = [path] if path else self.snapshot.paths()
        hits = []
        for candidate in candidates:
            content = self.snapshot.read(candidate)
            if content is None:
                continue
            for line_number, line in enumerate(content.splitlines(), 1):
                if needle in line.lower():
                    hits.append({"path": candidate, "line": line_number, "content": line[:500]})
        start = max(0, int(cursor))
        selected = hits[start:start + max(1, min(int(limit), 50))]
        # A single very long source line cannot bypass the returned-byte cap.
        selected = [dict(item, content=self._bound(str(item["content"]))) for item in selected]
        self._charge(str(selected))
        next_cursor = start + len(selected)
        return {"query": query, "hits": selected, "next_cursor": next_cursor if next_cursor < len(hits) else None,
                "total_hits": len(hits)}

    def find_symbol(self, symbol: str, cursor: int = 0, limit: int = 20) -> Dict[str, object]:
        return self._symbol_hits(symbol, "definition", cursor, limit)

    def find_references(self, symbol: str, cursor: int = 0, limit: int = 20) -> Dict[str, object]:
        return self._symbol_hits(symbol, "reference", cursor, limit)

    def _symbol_hits(self, symbol: str, kind: str, cursor: int, limit: int) -> Dict[str, object]:
        self.calls += 1
        if not self.snapshot.available():
            return {"available": False, "reason": "repository snapshot is unavailable"}
        needle = str(symbol).strip()
        if not needle:
            raise ValueError("symbol is required")
        definition = re.compile(r"\b(?:def|class|function|interface|type)\s+%s\b" % re.escape(needle))
        hits = []
        for path in self.snapshot.paths():
            content = self.snapshot.read(path) or ""
            for line, text in enumerate(content.splitlines(), 1):
                is_definition = bool(definition.search(text))
                if needle in text and (is_definition if kind == "definition" else not is_definition):
                    hits.append({"path": path, "line": line, "kind": kind, "excerpt": self._bound(text[:500])})
        start = max(0, int(cursor))
        selected = hits[start:start + max(1, min(int(limit), 50))]
        self._charge(str(selected))
        next_cursor = start + len(selected)
        return {"symbol": needle, "kind": kind, "hits": selected,
                "next_cursor": next_cursor if next_cursor < len(hits) else None, "total_hits": len(hits)}

    def _charge(self, value: str) -> None:
        self.retrieved_bytes += min(MAX_RESULT_BYTES, len(value.encode("utf-8")))

    @staticmethod
    def _bound(value: str) -> str:
        raw = value.encode("utf-8")
        if len(raw) <= MAX_RESULT_BYTES:
            return value
        return raw[:MAX_RESULT_BYTES].decode("utf-8", errors="ignore") + "\n... [result byte limit reached]"

    def _diff_blocks(self) -> Dict[str, str]:
        values: Dict[str, List[str]] = {}
        current = ""
        for line in self.diff.splitlines(True):
            if line.startswith("--- "):
                current = ""
            if line.startswith("+++ "):
                raw = line[4:].strip()
                current = raw[2:] if raw.startswith("b/") else raw
                if current != "/dev/null":
                    values.setdefault(current, [])
            if current and current != "/dev/null":
                values[current].append(line)
        return {path: "".join(lines) for path, lines in values.items()}
