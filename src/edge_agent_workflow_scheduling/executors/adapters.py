"""Adapters for existing LLM backends, mock runtimes, and local workers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any

from edge_agent_workflow_scheduling.agents import LLMBackend
from edge_agent_workflow_scheduling.common import LLMCall, LLMResult, ToolCall, ToolResult
from edge_agent_workflow_scheduling.executors.base import (
    llm_call_error,
    tool_call_error,
    validate_timeout,
)
from edge_agent_workflow_scheduling.llm import MockLLMRuntime
from edge_agent_workflow_scheduling.resources import LLMInstanceProfile, ToolReplicaProfile
from edge_agent_workflow_scheduling.tools import ToolSpec
from edge_agent_workflow_scheduling.workers import LocalWorker


@dataclass(slots=True)
class MockLLMExecutor:
    """Expose an existing MockLLMRuntime through the LLMExecutor contract."""

    runtime: MockLLMRuntime

    @property
    def profile(self) -> LLMInstanceProfile:
        return self.runtime.profile

    def execute(
        self,
        llm_call: LLMCall,
        *,
        tools: list[ToolSpec] | None = None,
        timeout_sec: float | None = None,
    ) -> LLMResult:
        del tools
        validate_timeout(timeout_sec)
        result = self.runtime.generate(llm_call)
        total_time = result.queue_wait_time_sec + result.inference_time_sec
        if timeout_sec is not None and total_time > timeout_sec:
            return LLMResult(
                llm_call_id=llm_call.llm_call_id,
                llm_id=self.profile.llm_id,
                success=False,
                queue_wait_time_sec=result.queue_wait_time_sec,
                inference_time_sec=result.inference_time_sec,
                error_code="timeout",
                error_message="mock LLM execution exceeded timeout_sec",
            )
        return result


@dataclass(slots=True)
class BackendLLMExecutor:
    """Adapt a scripted or provider backend to the LLMExecutor contract."""

    profile: LLMInstanceProfile
    backend: LLMBackend
    queue_wait_time_sec: float = 0.0
    input_transfer_time_sec: float = 0.0
    output_transfer_time_sec: float = 0.0

    def execute(
        self,
        llm_call: LLMCall,
        *,
        tools: list[ToolSpec] | None = None,
        timeout_sec: float | None = None,
    ) -> LLMResult:
        validate_timeout(timeout_sec)
        invalid = llm_call_error(self.profile, llm_call)
        if invalid is not None:
            return _failed_llm_result(self.profile, llm_call, *invalid)

        started_at = perf_counter()
        try:
            response = self.backend.create_response(
                llm_call,
                system_instruction=_system_instruction(llm_call.input_items),
                tools=tools or [],
                timeout_sec=timeout_sec or 120.0,
            )
        except TimeoutError as exc:
            return _failed_llm_result(
                self.profile,
                llm_call,
                "timeout",
                str(exc) or "LLM backend timed out",
                inference_time_sec=perf_counter() - started_at,
            )
        except Exception as exc:
            return _failed_llm_result(
                self.profile,
                llm_call,
                "llm_execution_failed",
                str(exc) or exc.__class__.__name__,
                inference_time_sec=perf_counter() - started_at,
            )

        inference_time_sec = perf_counter() - started_at
        if timeout_sec is not None and inference_time_sec > timeout_sec:
            return _failed_llm_result(
                self.profile,
                llm_call,
                "timeout",
                "LLM backend exceeded timeout_sec",
                inference_time_sec=inference_time_sec,
            )
        output_tokens = _output_tokens(response.metadata)
        return LLMResult(
            llm_call_id=llm_call.llm_call_id,
            llm_id=self.profile.llm_id,
            success=True,
            output_items=response.output_items,
            output_text=response.output_text,
            response_id=response.response_id,
            response_model=response.model,
            output_tokens=output_tokens,
            queue_wait_time_sec=self.queue_wait_time_sec,
            input_transfer_time_sec=self.input_transfer_time_sec,
            inference_time_sec=inference_time_sec,
            output_transfer_time_sec=self.output_transfer_time_sec,
            energy_joules=(
                (llm_call.input_tokens + output_tokens)
                * self.profile.energy_profile.get("joules_per_token", 0.0)
            ),
            metadata={**deepcopy(response.metadata), "executor_type": self.profile.executor_type},
        )


@dataclass(slots=True)
class LocalToolExecutor:
    """Expose an existing LocalWorker through the ToolExecutor contract."""

    worker: LocalWorker
    queue_wait_time_sec: float = 0.0
    input_transfer_time_sec: float = 0.0
    output_transfer_time_sec: float = 0.0

    @property
    def profile(self) -> ToolReplicaProfile:
        return self.worker.profile

    def execute(
        self,
        tool_call: ToolCall,
        *,
        timeout_sec: float | None = None,
    ) -> ToolResult:
        validate_timeout(timeout_sec)
        invalid = tool_call_error(self.profile, tool_call)
        if invalid is not None:
            return ToolResult(
                tool_call_id=tool_call.tool_call_id,
                replica_id=self.profile.replica_id,
                success=False,
                error_code=invalid[0],
                error_message=invalid[1],
            )
        result = self.worker.run_tool(tool_call)
        result = replace(
            result,
            queue_wait_time_sec=self.queue_wait_time_sec,
            input_transfer_time_sec=self.input_transfer_time_sec,
            output_transfer_time_sec=self.output_transfer_time_sec,
        )
        total_time = result.queue_wait_time_sec + result.execution_time_sec
        if timeout_sec is not None and total_time > timeout_sec:
            return replace(
                result,
                success=False,
                output=None,
                error_code="timeout",
                error_message="local Tool execution exceeded timeout_sec",
            )
        return result


def _failed_llm_result(
    profile: LLMInstanceProfile,
    llm_call: LLMCall,
    error_code: str,
    error_message: str,
    *,
    inference_time_sec: float = 0.0,
) -> LLMResult:
    return LLMResult(
        llm_call_id=llm_call.llm_call_id,
        llm_id=profile.llm_id,
        success=False,
        inference_time_sec=inference_time_sec,
        error_code=error_code,
        error_message=error_message,
    )


def _system_instruction(input_items: list[dict[str, Any]]) -> str:
    return "\n".join(
        content
        for item in input_items
        if item.get("role") == "system" and isinstance((content := item.get("content")), str)
    )


def _output_tokens(metadata: dict[str, Any]) -> int:
    usage = metadata.get("usage")
    if not isinstance(usage, dict):
        return 0
    output_tokens = usage.get("output_tokens", 0)
    return output_tokens if isinstance(output_tokens, int) and output_tokens >= 0 else 0
