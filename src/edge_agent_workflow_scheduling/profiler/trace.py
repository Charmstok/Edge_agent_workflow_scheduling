"""JSONL trace logging and trace record builders."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from edge_agent_workflow_scheduling.common import (
    LLMCall,
    LLMResult,
    ScheduleDecision,
    ToolCall,
    ToolResult,
    TraceRecord,
)


@dataclass(frozen=True, slots=True)
class JsonlTraceLogger:
    """Append-only JSONL logger for completed workflow-step traces."""

    trace_path: Path

    def write(self, record: TraceRecord) -> None:
        """Append one trace record to the JSONL file."""

        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("a", encoding="utf-8") as trace_file:
            trace_file.write(record.to_json())
            trace_file.write("\n")

    def write_many(self, records: Iterable[TraceRecord]) -> None:
        """Append multiple trace records to the JSONL file."""

        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("a", encoding="utf-8") as trace_file:
            for record in records:
                trace_file.write(record.to_json())
                trace_file.write("\n")

    def read_all(self) -> list[TraceRecord]:
        """Read all trace records from the JSONL file."""

        if not self.trace_path.exists():
            return []

        records: list[TraceRecord] = []
        with self.trace_path.open("r", encoding="utf-8") as trace_file:
            for line in trace_file:
                stripped_line = line.strip()
                if stripped_line:
                    records.append(TraceRecord.from_json(stripped_line))
        return records

    def count(self) -> int:
        """Return the number of non-empty JSONL records."""

        return len(self.read_all())

    def clear(self) -> None:
        """Remove the trace file if it exists."""

        self.trace_path.unlink(missing_ok=True)


def build_llm_trace_record(
    *,
    llm_call: LLMCall,
    decision: ScheduleDecision,
    result: LLMResult,
    timeout: bool = False,
    input_transfer_time_sec: float = 0.0,
    output_transfer_time_sec: float = 0.0,
) -> TraceRecord:
    """Build a profiler trace for a completed LLM inference step."""

    _validate_decision(decision, expected_task_id=llm_call.llm_call_id, expected_task_kind="llm")
    _validate_llm_result(llm_call, result)
    execution_time_sec = result.inference_time_sec
    total_latency_sec = (
        result.queue_wait_time_sec
        + input_transfer_time_sec
        + execution_time_sec
        + output_transfer_time_sec
    )

    return TraceRecord(
        task_id=llm_call.llm_call_id,
        task_kind="llm",
        agent_id=llm_call.agent_id,
        selected_target=decision.selected_target,
        policy_name=decision.policy_name,
        queue_wait_time_sec=result.queue_wait_time_sec,
        execution_time_sec=execution_time_sec,
        total_latency_sec=total_latency_sec,
        success=result.success,
        timeout=timeout,
        reward=calculate_step_reward(
            total_latency_sec=total_latency_sec,
            success=result.success,
            timeout=timeout,
        ),
        model_name=llm_call.model_name,
        input_transfer_time_sec=input_transfer_time_sec,
        output_transfer_time_sec=output_transfer_time_sec,
        input_tokens=llm_call.input_tokens,
        output_tokens=result.output_tokens,
        error_message=result.error_message,
    )


def build_tool_trace_record(
    *,
    tool_call: ToolCall,
    decision: ScheduleDecision,
    result: ToolResult,
    input_transfer_time_sec: float = 0.0,
    output_transfer_time_sec: float = 0.0,
    timeout: bool = False,
) -> TraceRecord:
    """Build a profiler trace for a completed tool execution step."""

    _validate_decision(decision, expected_task_id=tool_call.tool_call_id, expected_task_kind="tool")
    _validate_tool_result(tool_call, result)
    total_latency_sec = (
        result.queue_wait_time_sec
        + input_transfer_time_sec
        + result.execution_time_sec
        + output_transfer_time_sec
    )

    return TraceRecord(
        task_id=tool_call.tool_call_id,
        task_kind="tool",
        agent_id=tool_call.agent_id,
        selected_target=decision.selected_target,
        policy_name=decision.policy_name,
        queue_wait_time_sec=result.queue_wait_time_sec,
        execution_time_sec=result.execution_time_sec,
        total_latency_sec=total_latency_sec,
        success=result.success,
        timeout=timeout,
        reward=calculate_step_reward(
            total_latency_sec=total_latency_sec,
            success=result.success,
            timeout=timeout,
        ),
        tool_type=tool_call.tool_type,
        input_transfer_time_sec=input_transfer_time_sec,
        output_transfer_time_sec=output_transfer_time_sec,
        error_message=result.error_message,
    )


def calculate_step_reward(
    *,
    total_latency_sec: float,
    success: bool,
    timeout: bool,
    timeout_penalty: float = 20.0,
    failure_penalty: float = 50.0,
) -> float:
    """Compute the first-stage per-step reward."""

    reward = -total_latency_sec
    if timeout:
        reward -= timeout_penalty
    if not success:
        reward -= failure_penalty
    return reward


def _validate_decision(
    decision: ScheduleDecision,
    *,
    expected_task_id: str,
    expected_task_kind: str,
) -> None:
    if decision.task_id != expected_task_id:
        msg = f"decision task_id {decision.task_id!r} does not match {expected_task_id!r}"
        raise ValueError(msg)
    if decision.task_kind != expected_task_kind:
        msg = f"decision task_kind {decision.task_kind!r} does not match {expected_task_kind!r}"
        raise ValueError(msg)


def _validate_llm_result(llm_call: LLMCall, result: LLMResult) -> None:
    if result.llm_call_id != llm_call.llm_call_id:
        msg = f"LLMResult id {result.llm_call_id!r} does not match {llm_call.llm_call_id!r}"
        raise ValueError(msg)


def _validate_tool_result(tool_call: ToolCall, result: ToolResult) -> None:
    if result.tool_call_id != tool_call.tool_call_id:
        msg = f"ToolResult id {result.tool_call_id!r} does not match {tool_call.tool_call_id!r}"
        raise ValueError(msg)
