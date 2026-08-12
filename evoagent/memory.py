"""Scoped working, episodic and semantic memory for review agents.

Memory deliberately stores references to task evidence, never the raw tool or
evidence payload.  The loop context remains the owner of raw observations.
"""
from datetime import datetime, timedelta, timezone
from fnmatch import fnmatch
import hashlib
import json
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .store import utc_now


TOKEN = re.compile(r"[A-Za-z0-9_./:-]{2,}")
VALID_SCOPES = {"working", "episodic", "semantic", "procedural"}
FINDING_STATUSES = {"new", "open", "fixed", "suppressed", "rejected"}
WORKING_FIELDS = (
    "confirmed_evidence", "rejected_hypotheses", "open_questions",
    "files_inspected", "pending_files", "candidate_findings",
)


def _tokens(value: str) -> set:
    return {item.lower() for item in TOKEN.findall(value or "")}


def _normal(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return [str(value)] if str(value) else []


class MemoryManager:
    """Persist bounded, provenance-aware memory without a framework checkpoint."""

    def __init__(
        self, store, enabled: bool = True, recall_limit: int = 6,
        working_ttl_seconds: int = 86400, promotion_support_threshold: int = 2,
    ):
        self.store = store
        self.enabled = enabled
        self.recall_limit = max(1, recall_limit)
        self.working_ttl_seconds = max(60, working_ttl_seconds)
        self.promotion_support_threshold = max(1, int(promotion_support_threshold))

    def _save(self, record: Dict[str, Any]) -> Dict[str, Any]:
        record["keywords"] = sorted(set(record.get("keywords") or []))
        return self.store.save_agent_memory(record)

    def remember(
        self, tenant_id: str, repository: str, scope: str, kind: str,
        content: str, metadata: Optional[Dict[str, Any]] = None,
        task_id: str = "", agent: str = "", importance: float = 0.5,
        ttl_seconds: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Backward-compatible generic persistence entry point."""
        if not self.enabled or not content.strip():
            return None
        if scope not in VALID_SCOPES:
            raise ValueError("unsupported memory scope: %s" % scope)
        importance = max(0.0, min(1.0, float(importance)))
        metadata = dict(metadata or {})
        normalized = content.strip()[:8000]
        fingerprint = json.dumps({
            "tenant": tenant_id, "repository": repository, "scope": scope,
            "kind": kind, "content": normalized, "metadata": metadata,
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        ttl = self.working_ttl_seconds if scope == "working" and ttl_seconds is None else ttl_seconds
        expires_at = None
        if ttl:
            expires_at = (datetime.now(timezone.utc) + timedelta(
                seconds=max(1, int(ttl)))).isoformat()
        return self._save({
            "id": hashlib.sha256(fingerprint.encode("utf-8")).hexdigest(),
            "tenant_id": tenant_id or "default", "repository": repository,
            "task_id": task_id, "agent": agent, "scope": scope, "kind": kind,
            "content": normalized, "keywords": sorted(_tokens(normalized) | _tokens(kind)),
            "metadata": metadata, "importance": importance, "created_at": utc_now(),
            "expires_at": expires_at,
        })

    # Working -----------------------------------------------------------------
    def remember_working_state(
        self, tenant_id: str, repository: str, task_id: str, agent: str,
        shard_id: str, state: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Checkpoint compact Agent × Shard state; raw observations stay in LoopContext."""
        if not self.enabled or not task_id or not agent or not shard_id:
            return None
        state = dict(state or {})
        compact = {field: list(dict.fromkeys(_as_list(state.get(field))))[:80]
                   for field in WORKING_FIELDS}
        compact.update({"task_id": task_id, "agent": agent, "shard_id": shard_id})
        identity = "|".join((tenant_id or "default", repository, task_id, agent, shard_id))
        memory_id = hashlib.sha256(("working-state|" + identity).encode("utf-8")).hexdigest()
        expires_at = (datetime.now(timezone.utc) + timedelta(
            seconds=self.working_ttl_seconds)).isoformat()
        content = "Working state for %s/%s" % (agent, shard_id)
        return self._save({
            "id": memory_id, "tenant_id": tenant_id or "default", "repository": repository,
            "task_id": task_id, "agent": agent, "scope": "working", "kind": "working_state",
            "content": content, "keywords": sorted(_tokens(" ".join(compact["files_inspected"] + compact["pending_files"]))),
            "metadata": compact, "importance": .35, "created_at": utc_now(), "expires_at": expires_at,
        })

    def load_working_state(
        self, tenant_id: str, repository: str, task_id: str, agent: str, shard_id: str,
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        self._purge()
        for item in self.store.list_agent_memories(tenant_id or "default", repository, ("working",), 200):
            meta = item.get("metadata") or {}
            if (item.get("task_id") == task_id and item.get("agent") == agent
                    and meta.get("shard_id") == shard_id and item.get("kind") == "working_state"):
                return item
        return None

    # Episodes ----------------------------------------------------------------
    @staticmethod
    def finding_fingerprint(finding: Dict[str, Any], fuzzy: bool = False) -> str:
        evidence = _normal(finding.get("evidence"))
        identity = {
            "rule": finding.get("rule_id", "unknown"), "path": finding.get("path", ""),
            "evidence": evidence or _normal(finding.get("title")),
        }
        if not fuzzy:
            identity["line"] = int(finding.get("line", 0) or 0)
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:24]

    def _memories(self, tenant_id: str, repository: str, scopes: Sequence[str]) -> List[Dict[str, Any]]:
        self._purge()
        return self.store.list_agent_memories(tenant_id or "default", repository, tuple(scopes), 500)

    def _episode_match(self, tenant_id: str, repository: str, finding: Dict[str, Any], pr_number: Any) -> Optional[Dict[str, Any]]:
        strict = self.finding_fingerprint(finding)
        fuzzy = self.finding_fingerprint(finding, fuzzy=True)
        for item in self._memories(tenant_id, repository, ("episodic",)):
            meta = item.get("metadata") or {}
            if meta.get("episode_type") != "finding" or meta.get("pr_number") != pr_number:
                continue
            if meta.get("finding_fingerprint") == strict or meta.get("fuzzy_fingerprint") == fuzzy:
                return item
        return None

    def remember_finding(
        self, tenant_id: str, repository: str, task_id: str,
        finding: Dict[str, Any], approved: bool, reasons: Iterable[str] = (),
        pr_number: Any = None, source_sha: str = "", source_agents: Iterable[str] = (),
        shard_ids: Iterable[str] = (), cross_shard: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Write a verified review episode while retaining legacy finding kinds."""
        if not self.enabled:
            return None
        finding = dict(finding or {})
        decision = "approved" if approved else "rejected"
        existing = self._episode_match(tenant_id, repository, finding, pr_number) if approved else None
        old = dict((existing or {}).get("metadata") or {})
        strict = self.finding_fingerprint(finding)
        fuzzy = self.finding_fingerprint(finding, fuzzy=True)
        prior_support = int(old.get("support_count", 0))
        lifecycle = "open" if existing else ("new" if approved else "rejected")
        meta = {
            **old, "episode_type": "finding", "repository": repository, "pr_number": pr_number,
            "finding_id": old.get("finding_id") or finding.get("id") or strict,
            "finding_fingerprint": old.get("finding_fingerprint") or strict,
            "fuzzy_fingerprint": old.get("fuzzy_fingerprint") or fuzzy,
            "rule_id": finding.get("rule_id", "unknown"), "severity": finding.get("severity", ""),
            "first_seen_sha": old.get("first_seen_sha") or source_sha,
            "last_seen_sha": source_sha, "last_confirmed_sha": source_sha if approved else old.get("last_confirmed_sha", ""),
            "status": lifecycle, "path": finding.get("path", ""),
            "risk_domain": "security" if str(finding.get("rule_id", "")).startswith("SEC") else "reliability" if str(finding.get("rule_id", "")).startswith("REL") else "",
            "related_paths": _as_list(finding.get("related_paths")) or _as_list(finding.get("path")),
            "related_symbols": _as_list(finding.get("related_symbols")),
            "source_agents": sorted(set(_as_list(old.get("source_agents")) + _as_list(source_agents))),
            "shard_ids": sorted(set(_as_list(old.get("shard_ids")) + _as_list(shard_ids))),
            "cross_shard": bool(cross_shard or old.get("cross_shard")),
            "evidence_refs": _as_list(finding.get("evidence_refs")), "decision": decision,
            "decision_reasons": [str(item)[:500] for item in reasons],
            "confidence": float(finding.get("confidence", 0.0) or 0.0),
            "support_count": prior_support + (1 if approved else 0),
            "contradiction_count": int(old.get("contradiction_count", 0)) + (0 if approved else 1),
            "needs_revalidation": False,
        }
        # No evidence excerpt is copied into durable memory; EvidenceLedger owns it.
        content = "%s finding %s at %s:%s (episode %s)" % (
            decision, meta["rule_id"], meta["path"], finding.get("line", 0), meta["finding_id"])
        record = {
            "id": (existing or {}).get("id") or hashlib.sha256(
                ("episode|%s|%s|%s|%s" % (tenant_id, repository, pr_number, strict)).encode("utf-8")).hexdigest(),
            "tenant_id": tenant_id or "default", "repository": repository, "task_id": task_id,
            "agent": "arbiter-agent", "scope": "episodic", "kind": "finding_%s" % decision,
            "content": content, "keywords": sorted(_tokens(content) | _tokens(meta["rule_id"])),
            "metadata": meta, "importance": .8 if approved else .45, "created_at": (existing or {}).get("created_at") or utc_now(), "expires_at": None,
        }
        saved = self._save(record)
        if approved:
            self.promote_episode(tenant_id, repository, saved)
        return saved

    def compute_finding_delta(
        self, tenant_id: str, repository: str, pr_number: Any, source_sha: str,
        current_findings: Iterable[Dict[str, Any]], changed_paths: Iterable[str] = (),
    ) -> Dict[str, List[str]]:
        """Classify one PR revision as new/open/fixed, with fuzzy line-drift matching."""
        current = list(current_findings)
        strict = {self.finding_fingerprint(item) for item in current}
        fuzzy = {self.finding_fingerprint(item, fuzzy=True) for item in current}
        result = {"new": [], "open": [], "fixed": []}
        for item in self._memories(tenant_id, repository, ("episodic",)):
            meta = dict(item.get("metadata") or {})
            if meta.get("episode_type") != "finding" or meta.get("pr_number") != pr_number:
                continue
            present = meta.get("finding_fingerprint") in strict or meta.get("fuzzy_fingerprint") in fuzzy
            if present:
                result["open" if meta.get("first_seen_sha") != source_sha else "new"].append(item["id"])
                continue
            if meta.get("status") in {"new", "open"} and meta.get("last_seen_sha") != source_sha:
                meta["status"] = "fixed"
                meta["fixed_sha"] = source_sha
                meta["needs_revalidation"] = False
                item.update({"metadata": meta, "kind": "finding_approved", "created_at": item.get("created_at") or utc_now()})
                self._save(item)
                result["fixed"].append(item["id"])
        return result

    # Semantic ----------------------------------------------------------------
    def remember_feedback(
        self, tenant_id: str, repository: str, task_id: str, category: str,
        finding: Optional[Dict[str, Any]], note: str, source_sha: str = "",
    ) -> Optional[Dict[str, Any]]:
        finding = dict(finding or {})
        path = str(finding.get("path", ""))
        rule_id = str(finding.get("rule_id", ""))
        module = path.rsplit("/", 1)[0] if "/" in path else ""
        risk_domain = "security" if rule_id.startswith("SEC") else "reliability" if rule_id.startswith("REL") else ""
        metadata = {
            "category": category, "lesson": note.strip()[:2000], "rule_id": rule_id,
            "risk_domain": risk_domain, "module": module,
            "path_patterns": [path] if path else [], "symbols": _as_list(finding.get("related_symbols")),
            "language": finding.get("language", ""), "source_type": "human_feedback",
            "source_task": task_id, "source_pr": finding.get("pr_number"), "source_sha": source_sha,
            "source_finding": finding.get("id") or self.finding_fingerprint(finding) if finding else "",
            "confidence": .98 if category in {"false_positive", "missed_issue", "bad_fix"} else .85,
            "support_count": 1, "contradiction_count": 0, "status": "active",
            "last_confirmed_sha": source_sha,
        }
        content = "Scoped reviewer feedback (%s): %s" % (category, note.strip()[:2000])
        return self.remember(tenant_id, repository, "semantic", "review_feedback", content,
                             metadata, task_id=task_id, importance=metadata["confidence"])

    def promote_episode(self, tenant_id: str, repository: str, episode: Any) -> Optional[Dict[str, Any]]:
        """Promote only verified episodes which satisfy the configurable gate."""
        if not self.enabled:
            return None
        if isinstance(episode, str):
            episode = next((item for item in self._memories(tenant_id, repository, ("episodic",)) if item.get("id") == episode), None)
        if not isinstance(episode, dict):
            return None
        meta = dict(episode.get("metadata") or {})
        if (meta.get("episode_type") != "finding" or meta.get("decision") != "approved"
                or int(meta.get("support_count", 0)) < self.promotion_support_threshold):
            return None
        scope_key = (meta.get("rule_id"), meta.get("path"), tuple(meta.get("related_symbols") or []))
        existing = next((item for item in self._memories(tenant_id, repository, ("semantic",))
                         if (item.get("metadata") or {}).get("source_type") == "verified_episode"
                         and ((item.get("metadata") or {}).get("rule_id"), (item.get("metadata") or {}).get("path_patterns", [""])[0], tuple((item.get("metadata") or {}).get("symbols") or [])) == scope_key), None)
        old = dict((existing or {}).get("metadata") or {})
        path = str(meta.get("path", ""))
        semantic = {
            **old, "category": "verified_finding", "lesson": "Repeated verified %s finding." % meta.get("rule_id"),
            "rule_id": meta.get("rule_id", ""), "risk_domain": "security" if str(meta.get("rule_id", "")).startswith("SEC") else "",
            "module": path.rsplit("/", 1)[0] if "/" in path else "", "path_patterns": [path] if path else [],
            "symbols": list(meta.get("related_symbols") or []), "language": "", "source_type": "verified_episode",
            "source_task": episode.get("task_id", ""), "source_pr": meta.get("pr_number"),
            "source_sha": meta.get("last_confirmed_sha", ""), "source_finding": meta.get("finding_id", ""),
            "confidence": min(.95, .6 + .1 * int(meta.get("support_count", 0))),
            "support_count": int(meta.get("support_count", 0)), "contradiction_count": int(old.get("contradiction_count", 0)),
            "status": "active", "last_confirmed_sha": meta.get("last_confirmed_sha", ""),
        }
        content = "Scoped verified review knowledge: %s at %s" % (semantic["rule_id"], path)
        record = {
            "id": (existing or {}).get("id") or hashlib.sha256(("semantic|%s|%s" % (repository, scope_key)).encode("utf-8")).hexdigest(),
            "tenant_id": tenant_id or "default", "repository": repository, "task_id": semantic["source_task"], "agent": "memory-promotion",
            "scope": "semantic", "kind": "verified_episode", "content": content,
            "keywords": sorted(_tokens(content)), "metadata": semantic, "importance": semantic["confidence"],
            "created_at": (existing or {}).get("created_at") or utc_now(), "expires_at": None,
        }
        return self._save(record)

    def mark_stale_memories(
        self, tenant_id: str, repository: str, paths: Iterable[str] = (), symbols: Iterable[str] = (), source_sha: str = "",
    ) -> int:
        paths, symbols = set(_as_list(paths)), set(_as_list(symbols))
        changed = 0
        for item in self._memories(tenant_id, repository, ("semantic", "episodic")):
            meta = dict(item.get("metadata") or {})
            scope_patterns = _as_list(meta.get("path")) + _as_list(meta.get("path_patterns")) + _as_list(meta.get("related_paths"))
            scope_symbols = set(_as_list(meta.get("symbols")) + _as_list(meta.get("related_symbols")))
            path_touched = any(
                fnmatch(path, pattern) or fnmatch(pattern, path)
                for path in paths for pattern in scope_patterns if pattern
            )
            module = str(meta.get("module", "")).rstrip("/")
            module_touched = bool(module and any(path.startswith(module + "/") for path in paths))
            touched = bool(path_touched or module_touched or symbols.intersection(scope_symbols))
            if not touched or (source_sha and meta.get("last_confirmed_sha") == source_sha):
                continue
            if item.get("scope") == "semantic":
                meta["status"] = "needs_revalidation"
                meta["confidence"] = max(.0, float(meta.get("confidence", item.get("importance", .5))) - .15)
                item["importance"] = min(float(item.get("importance", .5)), float(meta["confidence"]))
            else:
                meta["needs_revalidation"] = True
            item["metadata"] = meta
            item["created_at"] = item.get("created_at") or utc_now()
            self._save(item)
            changed += 1
        return changed

    # Recall ------------------------------------------------------------------
    def _purge(self) -> None:
        purge = getattr(self.store, "purge_expired_agent_memories", None)
        if purge:
            purge()

    @staticmethod
    def _scope_match(meta: Dict[str, Any], scope: Dict[str, Any]) -> bool:
        files, modules = set(_as_list(scope.get("files"))), set(_as_list(scope.get("modules")))
        symbols, rules = set(_as_list(scope.get("symbols"))), set(_as_list(scope.get("rule_ids")))
        domains = set(_as_list(scope.get("risk_domains")))
        patterns = _as_list(meta.get("path_patterns")) + _as_list(meta.get("path")) + _as_list(meta.get("related_paths"))
        if any(fnmatch(path, pattern) or fnmatch(pattern, path) for path in files for pattern in patterns if pattern):
            return True
        if meta.get("module") and any(path.startswith(str(meta["module"]).rstrip("/") + "/") for path in files):
            return True
        if symbols.intersection(_as_list(meta.get("symbols")) + _as_list(meta.get("related_symbols"))):
            return True
        if rules and str(meta.get("rule_id", "")) in rules:
            return True
        # A broad domain may narrow an otherwise unscoped rule, but it cannot
        # override an explicit path/module/symbol scope from human feedback.
        has_concrete_scope = bool(patterns or meta.get("module") or _as_list(meta.get("symbols")) or _as_list(meta.get("related_symbols")))
        return bool(not has_concrete_scope and meta.get("risk_domain")
                    and meta.get("risk_domain") in domains and (files or modules))

    def recall(
        self, tenant_id: str, repository: str, query: str,
        scopes: Sequence[str] = ("semantic", "episodic"), limit: Optional[int] = None,
        scope: Optional[Dict[str, Any]] = None, source_sha: str = "",
    ) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        selected = tuple(item for item in scopes if item in VALID_SCOPES)
        if not selected:
            return []
        query_tokens = _tokens(query)
        ranked = []
        for index, item in enumerate(self._memories(tenant_id, repository, selected)):
            meta = item.get("metadata") or {}
            memory_tokens = set(item.get("keywords") or []) | _tokens(item.get("content", ""))
            overlap = len(query_tokens.intersection(memory_tokens))
            scope_match = self._scope_match(meta, scope or {}) if scope else False
            # Semantic memory cannot enter merely on importance: it must match
            # the assignment scope or carry lexical relevance.
            if item.get("scope") == "semantic" and not scope_match and overlap == 0:
                continue
            if item.get("scope") != "semantic" and query_tokens and overlap == 0 and not scope_match:
                continue
            coverage = overlap / max(1, len(query_tokens))
            freshness = .12 if source_sha and meta.get("last_confirmed_sha") == source_sha else .06 / (index + 1)
            version = -.18 if meta.get("status") == "needs_revalidation" or meta.get("needs_revalidation") else 0
            confidence = float(meta.get("confidence", item.get("importance", .5)))
            score = coverage * .5 + (.25 if scope_match else 0) + confidence * .18 + freshness + version
            value = dict(item)
            value["recall_score"] = round(score, 4)
            ranked.append(value)
        return sorted(ranked, key=lambda item: (-item["recall_score"], item.get("created_at", "")))[:max(1, limit or self.recall_limit)]

    def recall_for_assignment(
        self, tenant_id: str, repository: str, task_id: str, agent: str, shard_id: str,
        files: Iterable[str] = (), symbols: Iterable[str] = (), risk_domains: Iterable[str] = (),
        objective: str = "", source_sha: str = "", limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        files = _as_list(files)
        scope = {"files": files, "modules": [path.rsplit("/", 1)[0] for path in files if "/" in path],
                 "symbols": _as_list(symbols), "risk_domains": _as_list(risk_domains)}
        query = " ".join([objective, " ".join(files), " ".join(scope["symbols"]), " ".join(scope["risk_domains"])])
        values = self.recall(tenant_id, repository, query, ("semantic", "episodic"), limit, scope, source_sha)
        working = self.load_working_state(tenant_id, repository, task_id, agent, shard_id)
        return ([working] if working else []) + values[:max(0, (limit or self.recall_limit) - (1 if working else 0))]

    def forget_working(self, task_id: str) -> int:
        return 0 if not self.enabled else self.store.delete_agent_memories(task_id=task_id, scope="working")

    def consolidate_task(self, tenant_id: str, repository: str, task_id: str, summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.enabled or not task_id:
            return None
        archived = self.remember(tenant_id, repository, "episodic", "task_summary",
                                 "Review task %s completed" % task_id,
                                 metadata={"summary": dict(summary), "episode_type": "task_summary"},
                                 task_id=task_id, agent="agent-runtime", importance=.65)
        self.forget_working(task_id)
        return archived
