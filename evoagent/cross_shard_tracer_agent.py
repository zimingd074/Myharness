"""Dedicated cross-file contract tracer for the bounded review graph."""
from .reviewer import OpenAICompatibleReviewer


class CrossShardTracerReviewAgent(OpenAICompatibleReviewer):
    """Owns API/data-contract linkage, not security or reliability semantics."""

    agent_role = "cross_shard"
    domains = ("correctness", "api-contract", "schema-contract")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, system_prompt=(
            "You are a cross-file contract tracer. Trace changed exports and signatures to callers; "
            "trace added fields, options and optional parameters from producers to consumers. "
            "Report only broken caller/callee, API, schema, or producer/consumer contracts. Do not "
            "assess security policy, reliability, or business logic; those are separate specialists. "
            "For a removed guard, identify a replacement reference only when it affects a contract."
        ), **kwargs)
        self.name = "%s:%s:cross-shard-tracer" % (self.provider, self.model)
