"""Local worker prototype for executing tool calls in-process."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from time import perf_counter, sleep
from typing import Any

from edge_agent_workflow_scheduling.common import ToolCall, ToolResult, WorkerInfo, WorkerState
from edge_agent_workflow_scheduling.tools.base import ToolExecution

ToolExecutor = Callable[[ToolCall], str | ToolExecution | None]


@dataclass(slots=True)
class LocalWorker:
    """In-process Tool Worker used by the first local prototype."""

    worker_id: str
    supported_tools: list[str]
    max_concurrency: int = 1
    artificial_delay_sec: float = 0.0
    output_uri_prefix: str = "local://outputs/tools"
    tool_executors: Mapping[str, ToolExecutor] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.worker_id:
            msg = "worker_id must be non-empty"
            raise ValueError(msg)
        if not self.supported_tools:
            msg = "supported_tools must be non-empty"
            raise ValueError(msg)
        if self.max_concurrency < 1:
            msg = "max_concurrency must be at least 1"
            raise ValueError(msg)
        if self.artificial_delay_sec < 0:
            msg = "artificial_delay_sec must be non-negative"
            raise ValueError(msg)

        self.supported_tools = list(dict.fromkeys(self.supported_tools))

    def run_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call and return a structured result."""

        start_time = perf_counter()
        if tool_call.tool_name not in self.supported_tools:
            return ToolResult(
                tool_call_id=tool_call.tool_call_id,
                worker_id=self.worker_id,
                success=False,
                execution_time_sec=perf_counter() - start_time,
                error_message=(
                    f"tool_name {tool_call.tool_name!r} is not supported by worker "
                    f"{self.worker_id!r}"
                ),
            )

        try:
            if self.artificial_delay_sec > 0:
                sleep(self.artificial_delay_sec)

            tool_execution = self._execute_supported_tool(tool_call)
            return ToolResult(
                tool_call_id=tool_call.tool_call_id,
                worker_id=self.worker_id,
                success=True,
                output_uri=tool_execution.output_uri,
                execution_time_sec=perf_counter() - start_time,
                metadata=dict(tool_execution.metadata),
            )
        except Exception as exc:
            return ToolResult(
                tool_call_id=tool_call.tool_call_id,
                worker_id=self.worker_id,
                success=False,
                execution_time_sec=perf_counter() - start_time,
                error_message=str(exc),
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

    def _execute_supported_tool(self, tool_call: ToolCall) -> ToolExecution:
        executor = self.tool_executors.get(tool_call.tool_name)
        if executor is not None:
            result = executor(tool_call)
            if isinstance(result, ToolExecution):
                return result
            return ToolExecution(output_uri=result)

        return ToolExecution(output_uri=self._default_output_uri(tool_call))

    def _default_output_uri(self, tool_call: ToolCall) -> str:
        prefix = self.output_uri_prefix.rstrip("/")
        return f"{prefix}/{self.worker_id}/{tool_call.tool_name}/{tool_call.tool_call_id}.json"
