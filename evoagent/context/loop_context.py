"""Frozen, active and compressed context state for bounded agent loops."""
from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Dict, List


@dataclass
class EvidenceRecord:
    evidence_id: str
    source_tool: str
    excerpt: str
    path: str = ""
    line: int = 0
    importance: float = 0.5
    pinned: bool = False
    referenced_by_finding: List[str] = field(default_factory=list)
    shard_id: str = ""
    agent: str = ""
    legacy_id: str = ""

    def compact(self) -> Dict[str, Any]:
        return {"id": self.legacy_id or self.evidence_id, "content_id": self.evidence_id,
                "tool": self.source_tool, "path": self.path,
                "line": self.line, "excerpt": self.excerpt[:300], "pinned": self.pinned,
                "finding_ids": self.referenced_by_finding, "shard_id": self.shard_id,
                "agent": self.agent, "importance": self.importance}


@dataclass
class LoopRound:
    step: int
    action: Dict[str, Any]
    observation: Dict[str, Any]


class LoopContext:
    def __init__(self, active_rounds: int = 3, soft_ratio: float = .60, hard_ratio: float = .80):
        self.active_rounds = max(1, active_rounds)
        self.rounds: List[LoopRound] = []
        self.evidence: Dict[str, EvidenceRecord] = {}
        self.evidence_aliases: Dict[str, str] = {}
        self.summary: Dict[str, List[Any]] = {
            "confirmed_evidence": [], "rejected_hypotheses": [], "open_questions": [],
            "files_inspected": [], "pending_files": [], "candidate_findings": [],
        }
        self.compaction_count = 0
        self.compressed_rounds = 0
        self.soft_ratio = soft_ratio
        self.hard_ratio = hard_ratio

    def add_round(self, step: int, action: Dict[str, Any], observation: Dict[str, Any]) -> None:
        self.rounds.append(LoopRound(step, dict(action), dict(observation)))
        result = str(observation.get("result", ""))
        identity = {
            "step": step, "agent": observation.get("agent", ""),
            "shard": observation.get("shard_id", ""),
            "tool": observation.get("tool", ""), "result": result,
        }
        rendered = json.dumps(identity, ensure_ascii=False, sort_keys=True)
        evidence_id = "E:" + hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:24]
        legacy_id = "R%d" % step
        # The next model turn must be able to cite the observation it just
        # produced.  Previously the alias lived only in an internal map, so a
        # strict Challenger could not return any valid evidence_refs.
        observation["evidence_id"] = evidence_id
        observation["evidence_alias"] = legacy_id
        self.rounds[-1].observation.update({
            "evidence_id": evidence_id, "evidence_alias": legacy_id,
        })
        path, line = "", 0
        try:
            structured = json.loads(result)
            if isinstance(structured, dict):
                path = str(structured.get("path", ""))
                line = int(structured.get("line", 0) or 0)
                hits = list(structured.get("hits") or [])
                if not path and len(hits) == 1 and isinstance(hits[0], dict):
                    path = str(hits[0].get("path", ""))
                    line = int(hits[0].get("line", 0) or 0)
            elif isinstance(structured, list) and len(structured) == 1 and isinstance(structured[0], dict):
                path = str(structured[0].get("path", ""))
                line = int(structured[0].get("line", 0) or 0)
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
        self.evidence[evidence_id] = EvidenceRecord(
            evidence_id, str(observation.get("tool", "")), result[:600], path, line,
            shard_id=str(observation.get("shard_id", "")), agent=str(observation.get("agent", "")),
        )
        self.evidence_aliases[legacy_id] = evidence_id
        if path and path not in self.summary["files_inspected"]:
            self.summary["files_inspected"].append(path)
        reason = str(action.get("reason", ""))[:240]
        if reason:
            self.summary["open_questions"] = (self.summary["open_questions"] + [reason])[-12:]
        arguments = action.get("arguments") or {}
        path_hint = str(arguments.get("path", "")) if isinstance(arguments, dict) else ""
        if path_hint and path_hint not in self.summary["pending_files"]:
            self.summary["pending_files"].append(path_hint)

    def pin(self, evidence_ids: List[str], finding_id: str = "") -> None:
        for evidence_id in evidence_ids:
            requested = str(evidence_id)
            record = self.evidence.get(self.evidence_aliases.get(requested, requested))
            if record:
                if requested.startswith("R"):
                    record.legacy_id = requested
                record.pinned = True
                if finding_id and finding_id not in record.referenced_by_finding:
                    record.referenced_by_finding.append(finding_id)

    def pin_finding(self, finding_id: str, path: str = "", line: int = 0) -> List[str]:
        """Pin evidence directly referenced by an exact finding location when possible."""
        matched = [item.evidence_id for item in self.evidence.values()
                   if (not path or item.path == path) and (not line or item.line == line)]
        self.pin(matched, finding_id)
        if finding_id not in self.summary["candidate_findings"]:
            self.summary["candidate_findings"].append(finding_id)
        return matched

    def compact(self, force: bool = False) -> None:
        old = self.rounds[:-self.active_rounds]
        if not old and not force:
            return
        for item in old:
            observation = item.observation
            if observation.get("ok"):
                evidence_id = next((
                    record.evidence_id for record in self.evidence.values()
                    if record.source_tool == str(observation.get("tool", ""))
                    and record.agent == str(observation.get("agent", ""))
                    and record.shard_id == str(observation.get("shard_id", ""))
                ), "")
                evidence_id = next((alias for alias, target in self.evidence_aliases.items()
                                    if target == evidence_id), evidence_id)
                self.summary["confirmed_evidence"].append({
                    "step": item.step, "tool": observation.get("tool"), "evidence_id": evidence_id,
                })
            else:
                self.summary["rejected_hypotheses"].append({"step": item.step, "error": observation.get("error", "")[:160]})
        self.rounds = self.rounds[-self.active_rounds:]
        self.compressed_rounds += len(old)
        self.compaction_count += 1

    def render(self) -> Dict[str, Any]:
        return {
            "active_rounds": [asdict(item) for item in self.rounds],
            "compressed_history": self.summary,
            "pinned_evidence": [item.compact() for item in self.evidence.values() if item.pinned],
            "compaction_count": self.compaction_count,
            "compressed_rounds": self.compressed_rounds,
        }
