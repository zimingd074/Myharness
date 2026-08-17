"""Deterministic, coverage-first context building primitives."""

from .budget import TokenCounter, Utf8TokenCounter
from .pr_map import PRContextMap, build_pr_context_map
from .shard_planner import ReviewShard, ShardPlanner
from .retrieval import InMemorySnapshotProvider, RepositorySnapshotProvider
from .evidence import EvidenceLedger, LedgerEvidence
from .budget_policy import BudgetPolicy, ShardBudget
from .coverage import CoverageReceipt
from .risk_priority import DiffRiskScanner, RiskHunk

__all__ = [
    "TokenCounter", "Utf8TokenCounter", "PRContextMap", "build_pr_context_map",
    "ReviewShard", "ShardPlanner", "RepositorySnapshotProvider",
    "InMemorySnapshotProvider",
    "EvidenceLedger", "LedgerEvidence", "BudgetPolicy", "ShardBudget", "CoverageReceipt",
    "DiffRiskScanner", "RiskHunk",
]
