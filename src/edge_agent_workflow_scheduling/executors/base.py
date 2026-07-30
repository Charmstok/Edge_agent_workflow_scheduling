"""Provider-neutral executor contracts and factory selection."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from edge_agent_workflow_scheduling.common import LLMCall, LLMResult, ToolCall, ToolResult
from edge_agent_workflow_scheduling.resources import LLMInstanceProfile, ToolReplicaProfile
from edge_agent_workflow_scheduling.tools import ToolSpec


@runtime_checkable
class LLMExecutor(Protocol):
    """Execute one normalized LLM call on a selected instance."""

    profile: LLMInstanceProfile

    def execute(
        self,
        llm_call: LLMCall,
        *,
        tools: list[ToolSpec] | None = None,
        timeout_sec: float | None = None,
    ) -> LLMResult:
        """Return a provider-neutral LLM result."""


@runtime_checkable
class ToolExecutor(Protocol):
    """Execute one normalized Tool call on a selected replica."""

    profile: ToolReplicaProfile

    def execute(
        self,
        tool_call: ToolCall,
        *,
        timeout_sec: float | None = None,
    ) -> ToolResult:
        """Return a provider-neutral Tool result."""


LLMExecutorFactory = Callable[[LLMInstanceProfile], LLMExecutor]
ToolExecutorFactory = Callable[[ToolReplicaProfile], ToolExecutor]


@dataclass(slots=True)
class ExecutorFactoryRegistry:
    """Create executors from profile executor types without involving the Scheduler."""

    _llm_factories: dict[str, LLMExecutorFactory] = field(default_factory=dict)
    _tool_factories: dict[str, ToolExecutorFactory] = field(default_factory=dict)

    def register_llm(
        self,
        executor_type: str,
        factory: LLMExecutorFactory,
        *,
        replace: bool = False,
    ) -> None:
        _register_factory(self._llm_factories, executor_type, factory, replace=replace)

    def register_tool(
        self,
        executor_type: str,
        factory: ToolExecutorFactory,
        *,
        replace: bool = False,
    ) -> None:
        _register_factory(self._tool_factories, executor_type, factory, replace=replace)

    def create_llm(self, profile: LLMInstanceProfile) -> LLMExecutor:
        factory = _require_factory(self._llm_factories, profile.executor_type, "LLM")
        executor = factory(profile)
        if not isinstance(executor, LLMExecutor):
            raise TypeError("LLM executor factory must return an LLMExecutor")
        if executor.profile.llm_id != profile.llm_id:
            raise ValueError("LLM executor profile ID must match the requested profile")
        return executor

    def create_tool(self, profile: ToolReplicaProfile) -> ToolExecutor:
        factory = _require_factory(self._tool_factories, profile.executor_type, "Tool")
        executor = factory(profile)
        if not isinstance(executor, ToolExecutor):
            raise TypeError("Tool executor factory must return a ToolExecutor")
        if executor.profile.replica_id != profile.replica_id:
            raise ValueError("Tool executor profile ID must match the requested profile")
        return executor


@dataclass(slots=True)
class ExecutorPool:
    """Lazily create and retain executors for selected resource profiles."""

    factories: ExecutorFactoryRegistry
    _llm_executors: dict[str, LLMExecutor] = field(default_factory=dict)
    _tool_executors: dict[str, ToolExecutor] = field(default_factory=dict)

    def llm_executor(self, profile: LLMInstanceProfile) -> LLMExecutor:
        executor = self._llm_executors.get(profile.llm_id)
        if executor is None:
            executor = self.factories.create_llm(profile)
            self._llm_executors[profile.llm_id] = executor
        elif executor.profile.executor_type != profile.executor_type:
            raise ValueError("cached LLM executor_type does not match the resource profile")
        return executor

    def tool_executor(self, profile: ToolReplicaProfile) -> ToolExecutor:
        executor = self._tool_executors.get(profile.replica_id)
        if executor is None:
            executor = self.factories.create_tool(profile)
            self._tool_executors[profile.replica_id] = executor
        elif executor.profile.executor_type != profile.executor_type:
            raise ValueError("cached Tool executor_type does not match the resource profile")
        return executor


def validate_timeout(timeout_sec: float | None) -> None:
    if timeout_sec is not None and timeout_sec <= 0:
        raise ValueError("timeout_sec must be positive when provided")


def llm_call_error(
    profile: LLMInstanceProfile,
    llm_call: LLMCall,
) -> tuple[str, str] | None:
    if llm_call.model_name is not None and llm_call.model_name != profile.model:
        return (
            "model_mismatch",
            f"model_name {llm_call.model_name!r} does not match {profile.model!r}",
        )
    missing = sorted(set(llm_call.required_capabilities) - set(profile.capabilities))
    if missing:
        return "unsupported_capability", f"unsupported capabilities: {missing}"
    if llm_call.context_length > profile.context_window_tokens:
        return "context_length_exceeded", "context_length exceeds the instance limit"
    return None


def tool_call_error(
    profile: ToolReplicaProfile,
    tool_call: ToolCall,
) -> tuple[str, str] | None:
    if tool_call.tool_name != profile.tool_name:
        return (
            "unsupported_tool",
            f"tool_name {tool_call.tool_name!r} does not match {profile.tool_name!r}",
        )
    missing = sorted(set(tool_call.required_capabilities) - set(profile.capabilities))
    if missing:
        return "unsupported_capability", f"unsupported capabilities: {missing}"
    return None


def _register_factory(
    factories: dict[str, Callable],
    executor_type: str,
    factory: Callable,
    *,
    replace: bool,
) -> None:
    if not isinstance(executor_type, str) or not executor_type.strip():
        raise ValueError("executor_type must be a non-empty string")
    if executor_type in factories and not replace:
        raise ValueError(f"executor_type {executor_type!r} is already registered")
    factories[executor_type] = factory


def _require_factory(
    factories: dict[str, Callable],
    executor_type: str,
    resource_kind: str,
) -> Callable:
    try:
        return factories[executor_type]
    except KeyError as exc:
        raise KeyError(
            f"no {resource_kind} executor factory registered for {executor_type!r}"
        ) from exc
