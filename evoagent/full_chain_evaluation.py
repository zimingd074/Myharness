"""Adapters that score findings produced by the production task execution path."""
import os
import time
from dataclasses import replace
from typing import List

from .config import Settings
from .context.retrieval import InMemorySnapshotProvider
from .models import Finding, Severity
from .reviewer import LocalRuleReviewer, Reviewer
from .service import ReviewService


class QueuedServiceReviewer(Reviewer):
    """Run each case through ReviewService.enqueue_review and return final findings."""

    execution_path = "queued-review-service"

    def __init__(self, service: ReviewService, name: str, timeout_seconds: float = 30.0):
        self.service = service
        self.name = name
        self.timeout_seconds = timeout_seconds
        self.task_states: List[str] = []
        self._last_context = {}
        self._base_snapshot_factory = getattr(service.reviewer, "snapshot_factory", None)

    def review(self, diff, parsed):
        raise RuntimeError("QueuedServiceReviewer requires evaluation case metadata")

    def review_case(self, case: dict, parsed) -> List[Finding]:
        # Evaluation fixtures intentionally expose only their post-PR snapshot.
        # This exercises the same bounded repository tools without falling back
        # to the service host filesystem or requiring a GitHub credential.
        coordinator = self.service.reviewer
        if hasattr(coordinator, "snapshot_factory"):
            files = dict(case.get("after_files") or {})
            base_factory = self._base_snapshot_factory
            def fixture_snapshot(repository, pull_request, pr_map, task_id):
                if repository == case["repository"] and files:
                    return InMemorySnapshotProvider(files)
                return base_factory(repository, pull_request, pr_map, task_id) if base_factory else None
            coordinator.snapshot_factory = fixture_snapshot
        submitted = self.service.enqueue_review(
            case["repository"], case["diff"], int(case["pull_request"]),
            source="evaluation-full-chain",
        )
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            task = self.service.store.get(submitted["task_id"]) or {}
            state = str(task.get("state", ""))
            if state == "SUCCESS":
                self.task_states.append(state)
                reader = getattr(self.service.reviewer, "last_collaboration_summary", None)
                self._last_context = reader() if callable(reader) else {}
                return [
                    Finding(
                        rule_id=item["rule_id"],
                        severity=Severity(item["severity"]),
                        title=item["title"],
                        explanation=item["explanation"],
                        path=item["path"],
                        line=int(item["line"]),
                        evidence=item["evidence"],
                        fix=item["fix"],
                        test=item["test"],
                        confidence=float(item.get("confidence", 0.8)),
                    )
                    for item in task.get("report", {}).get("findings", [])
                ]
            if state in {"FAILED", "CANCELLED"}:
                self.task_states.append(state)
                raise RuntimeError("full-chain task %s ended in %s" % (submitted["task_id"], state))
            time.sleep(0.01)
        raise TimeoutError("full-chain task %s did not finish" % submitted["task_id"])

    def close(self) -> None:
        self.service.queue.close(wait=True)

    def last_collaboration_summary(self) -> dict:
        return dict(self._last_context)


def build_service(
    database_path: str, baseline: bool = False, with_llm: bool = False,
    llm_timeout_seconds: int = 300, async_workers: int = 2,
    context_architecture: str = "coverage-first",
    specialist_activation: str = "hybrid",
) -> ReviewService:
    if with_llm:
        configured = Settings.from_env()
        timeout = max(configured.timeout_seconds, llm_timeout_seconds)
        settings = replace(
            configured, db_path=database_path, database_url="", timeout_seconds=timeout,
            agent_runtime_timeout_seconds=max(configured.agent_runtime_timeout_seconds, timeout),
            context_architecture=context_architecture,
            context_specialist_activation=specialist_activation,
            async_workers=async_workers,
            skills_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), "skills"),
            memory_enabled=False, auth_required=False,
        )
    else:
        settings = Settings(
            host="127.0.0.1", port=8080, db_path=database_path, max_diff_bytes=1024 * 1024,
            max_steps=8, timeout_seconds=30, llm_base_url="", llm_api_key="", llm_model="",
            github_webhook_secret="", github_token="", auto_post_review=False,
            skills_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), "skills"),
            memory_enabled=False, async_workers=async_workers,
        )
    service = ReviewService(settings)
    if with_llm and not service.llm_config:
        raise ValueError("LLM evaluation requires a configured provider, model, and API key")
    if baseline:
        service.reviewer = LocalRuleReviewer()
    return service
