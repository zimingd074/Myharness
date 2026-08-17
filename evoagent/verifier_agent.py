"""Independent verifier-stage contract for the bounded review graph."""
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from .reviewer import OpenAICompatibleReviewer


class VerifierReviewAgent(OpenAICompatibleReviewer):
    """Independent false-positive gate; it never proposes new findings."""

    agent_role = "verifier"
    domains = ("security", "reliability", "correctness", "regression")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, system_prompt=(
            "You are a verifier, not a finder. Re-trace each supplied failure scenario using "
            "fresh repository evidence. Return only support, refute, or insufficient for the "
            "provided claims; never emit a new claim. A Critical may be refuted only with code "
            "that directly contradicts its failure scenario."
        ), **kwargs)
        self.name = "%s:%s:verifier" % (self.provider, self.model)


@dataclass(frozen=True)
class VerifierBatch:
    """A normalized, independently re-evidenced verifier assignment."""

    index: int
    claims: List[Any]


class VerifierAgentNode:
    """Owns finding batching and invokes the coordinator's verifier executor."""

    def __init__(self, batch_size: int = 8):
        self.batch_size = max(1, int(batch_size))

    def batches(self, claim_groups: Iterable[List[Any]]) -> List[VerifierBatch]:
        groups = list(claim_groups)
        return [
            VerifierBatch(index + 1, groups[start:start + self.batch_size])
            for index, start in enumerate(range(0, len(groups), self.batch_size))
        ]

    def run(self, coordinator: Any, state: Dict[str, Any]) -> Dict[str, Any]:
        # Coordinator retains snapshot/tool/runtime ownership; this node owns
        # the verification-stage topology and public contract.
        return coordinator._run_batched_verifier(state, self.batches)
