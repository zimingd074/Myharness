"""Independent reverse-audit stage contract for the bounded review graph."""
from dataclasses import dataclass
from typing import Any, Dict

from .reviewer import OpenAICompatibleReviewer


class AuditorReviewAgent(OpenAICompatibleReviewer):
    """Shard-local reverse auditor; it may only propose previously unseen findings."""

    agent_role = "auditor"
    domains = ("security", "reliability", "correctness", "regression")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, system_prompt=(
            "You are a shard reverse auditor. Inspect only your assigned shard and search for "
            "defects absent from the supplied confirmed findings. Do not re-verify, refute, or "
            "repeat those findings. Return only new evidence-backed findings, or an empty result "
            "with a receipt of what you re-examined."
        ), **kwargs)
        self.name = "%s:%s:auditor" % (self.provider, self.model)


@dataclass(frozen=True)
class AuditStopPolicy:
    max_rounds: int = 5
    consecutive_dry_rounds: int = 2

    def should_stop(self, rounds_completed: int, dry_rounds: int) -> bool:
        return rounds_completed >= self.max_rounds or dry_rounds >= self.consecutive_dry_rounds


class AuditorAgentNode:
    """Owns reverse-audit stopping policy and invokes shard-audit execution."""

    def __init__(self, stop_policy: AuditStopPolicy = None):
        self.stop_policy = stop_policy or AuditStopPolicy()

    def run(self, coordinator: Any, state: Dict[str, Any]) -> Dict[str, Any]:
        # Coordinator retains task-local tools/evidence ledger; this node keeps
        # audit policy independently importable and directly unit-testable.
        return coordinator._run_shard_audits(state, self.stop_policy)
