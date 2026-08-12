"""EvoAgent's dependency-free durable runtime and bounded agent loop.

The runtime deliberately separates orchestration from agent behaviour:

* ``AgentRuntime`` executes named nodes with budgets, retry policy, cancellation
  checks and application-owned checkpoints.
* ``AgentLoop`` executes model-selected tool actions until the agent returns a
  final result or its step/time budget is exhausted.

Both components are deterministic around side effects.  Persistence remains in
the application store so a worker restart does not depend on a framework-owned
checkpoint format.
"""
from contextlib import nullcontext
from dataclasses import dataclass, field
import json
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from .context.loop_context import LoopContext
from .context.reducers import reduce_tool_result
from .context.budget import ContextBudget


class RuntimeBudgetExceeded(RuntimeError):
    """The configured step or wall-clock budget was exhausted."""


class RuntimeCancelled(RuntimeError):
    """The owning task requested cancellation."""


class RuntimeStaleRun(RuntimeError):
    """A queued worker lost ownership of the task execution token."""


class AgentLoopProtocolError(RuntimeError):
    """An agent returned an invalid loop action."""


@dataclass(frozen=True)
class AgentTool:
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[..., Any]
    reducer: Optional[Callable[[Any, int], str]] = None

    def catalog_entry(self) -> Dict[str, Any]:
        return {
            "name": self.name, "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    """Explicit tool catalog with JSON-schema-like argument validation."""

    def __init__(self, tools: Iterable[AgentTool] = ()):
        self._tools: Dict[str, AgentTool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: AgentTool) -> None:
        if not tool.name or tool.name in self._tools:
            raise ValueError("tool names must be non-empty and unique")
        self._tools[tool.name] = tool

    def names(self) -> List[str]:
        return sorted(self._tools)

    def catalog(self) -> List[Dict[str, Any]]:
        return [self._tools[name].catalog_entry() for name in self.names()]

    def invoke(self, name: str, arguments: Dict[str, Any]) -> Any:
        tool = self._tools.get(name)
        if tool is None:
            raise AgentLoopProtocolError("unknown agent tool: %s" % name)
        self._validate(tool.parameters, arguments)
        return tool.handler(**arguments)

    @staticmethod
    def _validate(schema: Dict[str, Any], arguments: Dict[str, Any]) -> None:
        if not isinstance(arguments, dict):
            raise AgentLoopProtocolError("tool arguments must be an object")
        properties = dict(schema.get("properties") or {})
        required = set(schema.get("required") or [])
        missing = required.difference(arguments)
        if missing:
            raise AgentLoopProtocolError(
                "missing required tool arguments: %s" % ", ".join(sorted(missing))
            )
        if schema.get("additionalProperties", False) is False:
            unknown = set(arguments).difference(properties)
            if unknown:
                raise AgentLoopProtocolError(
                    "unknown tool arguments: %s" % ", ".join(sorted(unknown))
                )
        expected_types = {
            "string": str, "integer": int, "number": (int, float),
            "boolean": bool, "object": dict, "array": list,
        }
        for key, value in arguments.items():
            spec = properties.get(key) or {}
            expected = expected_types.get(spec.get("type"))
            if expected and (not isinstance(value, expected) or (
                spec.get("type") in {"integer", "number"} and isinstance(value, bool)
            )):
                raise AgentLoopProtocolError(
                    "tool argument %s must be %s" % (key, spec.get("type"))
                )
            if isinstance(value, (int, float)):
                if "minimum" in spec and value < spec["minimum"]:
                    raise AgentLoopProtocolError("tool argument %s is below minimum" % key)
                if "maximum" in spec and value > spec["maximum"]:
                    raise AgentLoopProtocolError("tool argument %s exceeds maximum" % key)


@dataclass(frozen=True)
class RuntimeNode:
    name: str
    handler: Callable[[Dict[str, Any]], Dict[str, Any]]
    retries: Optional[int] = None
    checkpoint: bool = True


@dataclass(frozen=True)
class RuntimeEvent:
    kind: str
    node: str
    step: int
    attempt: int = 0
    detail: Dict[str, Any] = field(default_factory=dict)


class AgentRuntime:
    """Execute a bounded node graph without a third-party orchestration engine."""

    def __init__(
        self, max_steps: int = 8, timeout_seconds: int = 120,
        node_retries: int = 0,
    ):
        if max_steps < 1:
            raise ValueError("runtime max_steps must be at least 1")
        if timeout_seconds < 1:
            raise ValueError("runtime timeout_seconds must be at least 1")
        if node_retries < 0:
            raise ValueError("runtime node_retries cannot be negative")
        self.max_steps = max_steps
        self.timeout_seconds = timeout_seconds
        self.node_retries = node_retries

    def execute(
        self, initial_state: Dict[str, Any], nodes: Iterable[RuntimeNode],
        task_id: str = "", checkpoint_store=None,
        checkpoint_fingerprints: Optional[Mapping[str, str]] = None,
        run_token: str = "",
        claim_token: str = "",
        cancel_check: Optional[Callable[[], bool]] = None,
        event_sink: Optional[Callable[[RuntimeEvent], None]] = None,
        span_factory: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
        non_retryable: Tuple[type, ...] = (
            ValueError, RuntimeCancelled, RuntimeBudgetExceeded, RuntimeStaleRun,
        ),
    ) -> Dict[str, Any]:
        state = dict(initial_state)
        started = time.monotonic()
        steps = 0
        checkpoints = (
            checkpoint_store.load_checkpoints(task_id)
            if checkpoint_store is not None and task_id else {}
        )
        checkpoint_fingerprints = dict(checkpoint_fingerprints or {})

        def emit(kind: str, node: str, attempt: int = 0, **detail) -> None:
            if event_sink:
                event_sink(RuntimeEvent(kind, node, steps, attempt, detail))

        def guard(node: str) -> None:
            if cancel_check and cancel_check():
                emit("cancelled", node)
                raise RuntimeCancelled("Task was cancelled")
            if steps >= self.max_steps or time.monotonic() - started > self.timeout_seconds:
                emit("budget_exhausted", node)
                raise RuntimeBudgetExceeded("task execution budget exceeded")

        for node in nodes:
            cached = checkpoints.get(node.name) if node.checkpoint else None
            has_fingerprint = node.name in checkpoint_fingerprints
            fingerprint = checkpoint_fingerprints.get(node.name, "")
            if cached and cached.get("status") == "completed" and (
                not has_fingerprint or (
                    fingerprint and cached.get("fingerprint") == fingerprint
                )
            ):
                output = dict(cached.get("state") or {})
                state.update(output)
                emit("checkpoint_restored", node.name, int(cached.get("attempt", 0)))
                continue
            if (cached and cached.get("status") == "completed" and node.checkpoint
                    and has_fingerprint):
                emit("checkpoint_invalidated", node.name, int(cached.get("attempt", 0)))

            retries = self.node_retries if node.retries is None else node.retries
            previous_attempt = int((cached or {}).get("attempt", 0))
            last_error: Optional[Exception] = None
            for offset in range(1, retries + 2):
                guard(node.name)
                steps += 1
                attempt = previous_attempt + offset
                emit("node_started", node.name, attempt)
                try:
                    attrs = {
                        "task_id": task_id, "node": node.name,
                        "attempt": attempt, "runtime_step": steps,
                    }
                    context = (
                        span_factory("runtime.%s" % node.name, attrs)
                        if span_factory else nullcontext()
                    )
                    with context:
                        output = node.handler(state) or {}
                    if not isinstance(output, dict):
                        raise TypeError("runtime node %s must return a dict" % node.name)
                    state.update(output)
                    if checkpoint_store is not None and task_id and node.checkpoint:
                        saved = checkpoint_store.save_checkpoint(
                            task_id, node.name, output, "completed", attempt,
                            fingerprint=fingerprint, run_token=run_token,
                            claim_token=claim_token,
                        )
                        if saved is False:
                            raise RuntimeStaleRun("task execution was superseded")
                    emit("node_completed", node.name, attempt, output_keys=sorted(output))
                    last_error = None
                    break
                except non_retryable:
                    raise
                except Exception as exc:
                    last_error = exc
                    if checkpoint_store is not None and task_id and node.checkpoint:
                        saved = checkpoint_store.save_checkpoint(
                            task_id, node.name, {}, "failed", attempt, str(exc),
                            fingerprint=fingerprint, run_token=run_token,
                            claim_token=claim_token,
                        )
                        if saved is False:
                            raise RuntimeStaleRun("task execution was superseded")
                    emit(
                        "node_failed", node.name, attempt,
                        error=str(exc)[:1000], will_retry=offset <= retries,
                    )
            if last_error is not None:
                raise last_error
        return state


@dataclass
class AgentLoopResult:
    output: Any
    steps: int
    observations: List[Dict[str, Any]]
    stop_reason: str
    loop_context: Dict[str, Any] = field(default_factory=dict)


class AgentLoop:
    """Run a model/tool loop with strict action, time and observation budgets."""

    def __init__(
        self, max_steps: int = 4, timeout_seconds: int = 45,
        max_observation_chars: int = 4000, active_rounds: int = 3,
        soft_compact_ratio: float = .60, hard_compact_ratio: float = .80,
        context_budget: Optional[ContextBudget] = None,
    ):
        if max_steps < 1:
            raise ValueError("agent loop max_steps must be at least 1")
        if timeout_seconds < 1:
            raise ValueError("agent loop timeout_seconds must be at least 1")
        self.max_steps = max_steps
        self.timeout_seconds = timeout_seconds
        self.max_observation_chars = max(256, max_observation_chars)
        self.active_rounds = max(1, active_rounds)
        self.soft_compact_ratio = soft_compact_ratio
        self.hard_compact_ratio = hard_compact_ratio
        self.context_budget = context_budget

    def run(
        self, stepper: Callable[[Dict[str, Any]], Dict[str, Any]],
        tools: Any, initial_state: Dict[str, Any],
        event_sink: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> AgentLoopResult:
        state = dict(initial_state)
        observations = list(state.get("observations") or [])
        loop_context = LoopContext(self.active_rounds, self.soft_compact_ratio, self.hard_compact_ratio)
        started = time.monotonic()

        def emit(kind: str, **detail) -> None:
            if event_sink:
                event_sink(kind, detail)

        for step in range(1, self.max_steps + 1):
            if time.monotonic() - started > self.timeout_seconds:
                emit("agent_loop_budget_exhausted", step=step, budget="time")
                raise RuntimeBudgetExceeded("agent loop time budget exceeded")
            state["loop_step"] = step
            state["observations"] = list(observations)
            state["loop_context"] = loop_context.render()
            action = stepper(state)
            if not isinstance(action, dict):
                raise AgentLoopProtocolError("agent loop action must be an object")
            kind = str(action.get("action", "")).strip().lower()
            emit("agent_loop_action", step=step, action=kind)
            if kind == "final":
                output = action.get("findings", action.get("output"))
                for finding in list(output or []):
                    finding_id = str(getattr(finding, "rule_id", "finding"))
                    explicit_refs = list(getattr(finding, "evidence_refs", []) or [])
                    if explicit_refs:
                        loop_context.pin(explicit_refs, finding_id)
                    else:
                        # A model that did not return an ID may only fall back
                        # to an exact changed-line locator, never "latest".
                        loop_context.pin_finding(
                            finding_id, str(getattr(finding, "path", "")),
                            int(getattr(finding, "line", 0) or 0),
                        )
                return AgentLoopResult(
                    output, step,
                    observations, "final", loop_context.render(),
                )
            if kind != "tool":
                raise AgentLoopProtocolError("unsupported agent loop action: %s" % kind)
            tool_name = str(action.get("tool", "")).strip()
            arguments = action.get("arguments") or {}
            if not isinstance(arguments, dict):
                raise AgentLoopProtocolError("tool arguments must be an object")
            try:
                if isinstance(tools, ToolRegistry):
                    value = tools.invoke(tool_name, arguments)
                    tool_spec = tools._tools.get(tool_name)
                else:
                    tool = tools.get(tool_name)
                    if tool is None:
                        raise AgentLoopProtocolError("unknown agent tool: %s" % tool_name)
                    value = tool(**arguments)
                    tool_spec = None
                rendered = (
                    tool_spec.reducer(value, self.max_observation_chars)
                    if tool_spec and tool_spec.reducer else
                    reduce_tool_result(tool_name, value, self.max_observation_chars)
                )
                observation = {
                    "step": step, "tool": tool_name, "ok": True,
                    "result": rendered,
                }
            except Exception as exc:
                observation = {
                    "step": step, "tool": tool_name, "ok": False,
                    "error": str(exc)[:1000],
                }
            observations.append(observation)
            observation.update({
                "shard_id": str(state.get("shard_identity", {}).get("id", "")),
                "agent": str(state.get("assignment", {}).get("agent", "")),
            })
            loop_context.add_round(step, action, observation)
            evidence_refs = action.get("evidence_refs") or []
            if isinstance(evidence_refs, list):
                loop_context.pin(evidence_refs)
            context_tokens = int(action.get("_context_tokens", 0) or 0)
            context_max_tokens = max(1, int(action.get("_context_max_tokens", 0) or 1))
            # ContextBudget is the single threshold authority.  A supplied
            # manager budget preserves its configured 60%/80% policy; a loop
            # used stand-alone gets an equivalent local policy.
            budget = self.context_budget or ContextBudget(
                max_tokens=context_max_tokens,
                soft_compact_ratio=self.soft_compact_ratio,
                hard_compact_ratio=self.hard_compact_ratio,
            )
            if budget.max_tokens != context_max_tokens:
                budget = ContextBudget(
                    max_tokens=context_max_tokens,
                    soft_compact_ratio=budget.soft_compact_ratio,
                    hard_compact_ratio=budget.hard_compact_ratio,
                )
            over_soft = budget.should_compact(context_tokens)
            over_hard = budget.should_compact(context_tokens, hard=True)
            if len(observations) > loop_context.active_rounds or over_soft:
                loop_context.compact(force=over_hard)
            emit("agent_loop_observation", **observation)
        emit("agent_loop_budget_exhausted", step=self.max_steps, budget="steps")
        raise RuntimeBudgetExceeded("agent loop step budget exceeded")
