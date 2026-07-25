"""Local worker prototype for executing tool calls in-process."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter, sleep
from typing import Any

from edge_agent_workflow_scheduling.common import ToolCall, ToolResult, WorkerInfo, WorkerState
from edge_agent_workflow_scheduling.tools import ToolRegistry


@dataclass(slots=True)
class LocalWorker:
    """In-process Tool Worker used by the first local prototype."""

    worker_id: str
    tool_registry: ToolRegistry
    max_concurrency: int = 1
    artificial_delay_sec: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.worker_id:
            msg = "worker_id must be non-empty"
            raise ValueError(msg)
        if not self.tool_registry.supported_tools():
            msg = "supported_tools must be non-empty"
            raise ValueError(msg)
        if self.max_concurrency < 1:
            msg = "max_concurrency must be at least 1"
            raise ValueError(msg)
        if self.artificial_delay_sec < 0:
            msg = "artificial_delay_sec must be non-negative"
            raise ValueError(msg)

    @property
    def supported_tools(self) -> list[str]:
        """Return Tool functionality deployed on this replica."""

        return self.tool_registry.supported_tools()

    def run_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call and return a structured result."""

        start_time = perf_counter()
        if tool_call.tool_name not in self.supported_tools:
            return ToolResult(
                tool_call_id=tool_call.tool_call_id,
                worker_id=self.worker_id,
                success=False,
                execution_time_sec=perf_counter() - start_time,
                error_code="unsupported_tool",
                error_message=(
                    f"tool_name {tool_call.tool_name!r} is not supported by worker "
                    f"{self.worker_id!r}"
                ),
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
                worker_id=self.worker_id,
                success=tool_execution.success,
                output=tool_execution.output,
                execution_time_sec=perf_counter() - start_time,
                metadata=dict(tool_execution.metadata),
                error_code=tool_execution.error_code,
                error_message=tool_execution.error_message,
            )
        except Exception as exc:
            return ToolResult(
                tool_call_id=tool_call.tool_call_id,
                worker_id=self.worker_id,
                success=False,
                execution_time_sec=perf_counter() - start_time,
                error_code="worker_execution_failed",
                error_message=str(exc) or exc.__class__.__name__,
            )

    def to_info(self) -> WorkerInfo:
        """Return static worker metadata."""

        return WorkerInfo(
            worker_id=self.worker_id,
            supported_tools=list(self.supported_tools),
            max_concurrency=self.max_concurrency,
            metadata=dict(self.metadata),
        )

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
    ) -> WorkerState:
        """Return a lightweight runtime state snapshot."""

        return WorkerState(
            worker_id=self.worker_id,
            supported_tools=list(self.supported_tools),
            queue_len=queue_len,
            running_tasks=running_tasks,
            max_concurrency=self.max_concurrency,
            cpu_util=cpu_util,
            memory_util=memory_util,
            network_latency_ms=network_latency_ms,
            avg_execution_time_sec=avg_execution_time_sec,
            recent_failure_rate=recent_failure_rate,
            is_online=is_online,
        )
