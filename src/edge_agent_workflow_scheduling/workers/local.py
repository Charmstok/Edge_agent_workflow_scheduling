"""Local worker prototype for executing tool calls in-process."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter, sleep

from edge_agent_workflow_scheduling.common import ToolCall, ToolResult
from edge_agent_workflow_scheduling.resources import ToolReplicaProfile, ToolReplicaState
from edge_agent_workflow_scheduling.tools import ToolRegistry


@dataclass(slots=True)
class LocalWorker:
    """In-process Tool Worker used by the first local prototype."""

    profile: ToolReplicaProfile
    tool_registry: ToolRegistry
    artificial_delay_sec: float = 0.0

    def __post_init__(self) -> None:
        if self.profile.tool_name not in self.tool_registry.supported_tools():
            msg = f"tool_name {self.profile.tool_name!r} is not registered locally"
            raise ValueError(msg)
        if self.artificial_delay_sec < 0:
            msg = "artificial_delay_sec must be non-negative"
            raise ValueError(msg)

    @property
    def replica_id(self) -> str:
        return self.profile.replica_id

    @property
    def max_concurrency(self) -> int:
        return self.profile.max_concurrency

    def run_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call and return a structured result."""

        start_time = perf_counter()
        if tool_call.tool_name != self.profile.tool_name:
            return ToolResult(
                tool_call_id=tool_call.tool_call_id,
                replica_id=self.replica_id,
                success=False,
                execution_time_sec=perf_counter() - start_time,
                error_code="unsupported_tool",
                error_message=(
                    f"tool_name {tool_call.tool_name!r} is not supported by worker "
                    f"{self.replica_id!r}"
                ),
            )
        missing_capabilities = sorted(
            set(tool_call.required_capabilities) - set(self.profile.capabilities)
        )
        if missing_capabilities:
            return ToolResult(
                tool_call_id=tool_call.tool_call_id,
                replica_id=self.replica_id,
                success=False,
                execution_time_sec=perf_counter() - start_time,
                error_code="unsupported_capability",
                error_message=f"unsupported capabilities: {missing_capabilities}",
            )

        try:
            if self.artificial_delay_sec > 0:
                sleep(self.artificial_delay_sec)

            tool_execution = self.tool_registry.execute(
                tool_call.tool_name,
                tool_call.arguments,
                invocation_id=tool_call.tool_call_id,
            )
            return ToolResult(
                tool_call_id=tool_call.tool_call_id,
                replica_id=self.replica_id,
                success=tool_execution.success,
                output=tool_execution.output,
                execution_time_sec=perf_counter() - start_time,
                energy_joules=self.profile.energy_profile.get("joules_per_call", 0.0),
                metadata=dict(tool_execution.metadata),
                error_code=tool_execution.error_code,
                error_message=tool_execution.error_message,
            )
        except Exception as exc:
            return ToolResult(
                tool_call_id=tool_call.tool_call_id,
                replica_id=self.replica_id,
                success=False,
                execution_time_sec=perf_counter() - start_time,
                error_code="worker_execution_failed",
                error_message=str(exc) or exc.__class__.__name__,
            )

    def to_profile(self) -> ToolReplicaProfile:
        """Return a copy of this replica's static profile."""

        return deepcopy(self.profile)

    def get_state(
        self,
        *,
        queue_len: int = 0,
        running_tasks: int = 0,
        cpu_util: float = 0.0,
        memory_util: float = 0.0,
        network_latency_ms: float = 0.0,
        avg_execution_time_sec: float = 0.0,
        recent_failure_rate: float = 0.0,
        is_online: bool = True,
    ) -> ToolReplicaState:
        """Return a lightweight runtime state snapshot."""

        return ToolReplicaState(
            replica_id=self.profile.replica_id,
            queue_len=queue_len,
            running_tasks=running_tasks,
            cpu_util=cpu_util,
            memory_util=memory_util,
            network_latency_ms=network_latency_ms,
            avg_execution_time_sec=avg_execution_time_sec,
            recent_failure_rate=recent_failure_rate,
            is_online=is_online,
        )
