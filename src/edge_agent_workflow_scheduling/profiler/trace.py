"""Trace persistence and builders for calls and complete Agent runs."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from edge_agent_workflow_scheduling.agents.runner import (
    AgentRunner,
    CallExecutionRecord,
    ScheduledAgentExecution,
)
from edge_agent_workflow_scheduling.common import (
    AgentRunStatus,
    LLMCall,
    LLMResult,
    ScheduleDecision,
    ToolCall,
    ToolResult,
    TraceRecord,
)
from edge_agent_workflow_scheduling.profiler.models import (
    AgentRunTrace,
    CallTrace,
    ExperimentManifest,
    TraceBundle,
)
from edge_agent_workflow_scheduling.profiler.privacy import content_digest, sanitize_for_trace
from edge_agent_workflow_scheduling.resources import ResourceRegistry
from edge_agent_workflow_scheduling.tools import build_function_call_output


@dataclass(frozen=True, slots=True)
class JsonlTraceLogger:
    """Append-only JSONL logger for completed call traces."""

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


@dataclass(frozen=True, slots=True)
class TraceBundleStore:
    """Read and write one complete, versioned experiment trace as JSON."""

    trace_path: Path

    def write(self, trace: TraceBundle) -> None:
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.trace_path.write_text(f"{trace.to_json()}\n", encoding="utf-8")

    def read(self) -> TraceBundle:
        return TraceBundle.from_json(self.trace_path.read_text(encoding="utf-8"))


def build_experiment_manifest(
    *,
    experiment_id: str,
    dataset_id: str,
    sample_ids: list[str],
    runner: AgentRunner,
    system_prompt_version: str,
    user_template: str,
    user_template_version: str,
    llm_profile_version: str,
    tool_profile_version: str,
    code_version: str,
    mode: str,
    sampling_parameters: dict[str, Any] | None = None,
    provider_seed: int | None = None,
    scheduler_parameters: dict[str, Any] | None = None,
    scheduler_seed: int | None = None,
    profile_seed: int | None = None,
    run_started_at: str | None = None,
    workload_parameters: dict[str, Any] | None = None,
) -> ExperimentManifest:
    """Capture Agent, Tool, scheduler, and resource inputs for an experiment."""

    if mode not in {"live", "replay"}:
        raise ValueError("mode must be 'live' or 'replay'")
    tool_schemas = runner.tool_registry.tools()
    versions: dict[str, set[str]] = {}
    for snapshot in runner.resources.tool_snapshots():
        versions.setdefault(snapshot.profile.tool_name, set()).add(
            snapshot.profile.implementation_version
        )

    manifest_data = {
        "experiment_id": experiment_id,
        "dataset_id": dataset_id,
        "sample_ids": list(sample_ids),
        "system_prompt": runner.system_instruction,
        "system_prompt_version": system_prompt_version,
        "user_template": user_template,
        "user_template_version": user_template_version,
        "tool_schemas": tool_schemas,
        "tool_order": [schema["name"] for schema in tool_schemas],
        "tool_implementation_versions": {
            name: sorted(implementation_versions)
            for name, implementation_versions in sorted(versions.items())
        },
        "model_endpoints": [
            {
                "base_url": snapshot.profile.base_url,
                "llm_id": snapshot.profile.llm_id,
                "model": snapshot.profile.model,
                "provider": snapshot.profile.provider,
            }
            for snapshot in runner.resources.llm_snapshots()
        ],
        "sampling_parameters": sampling_parameters or {},
        "provider_seed": provider_seed,
        "agent_limits": {
            "max_rounds": runner.max_rounds,
            "max_tool_calls": runner.max_tool_calls,
            "timeout_sec": runner.timeout_sec,
        },
        "scheduler_name": runner.scheduler.policy_name,
        "scheduler_parameters": (
            scheduler_parameters
            if scheduler_parameters is not None
            else runner.scheduler.manifest_parameters()
        ),
        "scheduler_seed": (
            scheduler_seed
            if scheduler_seed is not None
            else runner.scheduler.policy_config.random_seed
        ),
        "llm_profile_version": llm_profile_version,
        "tool_profile_version": tool_profile_version,
        "resource_profiles": _resource_profile_snapshot(runner.resources),
        "profile_seed": profile_seed,
        "code_version": code_version,
        "mode": mode,
        "workload_parameters": workload_parameters or {},
    }
    if run_started_at is not None:
        manifest_data["run_started_at"] = run_started_at
    return ExperimentManifest(**sanitize_for_trace(manifest_data))


def build_trace_bundle(
    execution: ScheduledAgentExecution,
    manifest: ExperimentManifest,
) -> TraceBundle:
    """Build a replayable and measurable trace from one scheduled Agent execution."""

    model_name_by_target = {
        endpoint["llm_id"]: endpoint["model"]
        for endpoint in manifest.model_endpoints
        if isinstance(endpoint.get("llm_id"), str) and isinstance(endpoint.get("model"), str)
    }
    calls = [
        _build_call_trace(
            sequence_id=index,
            record=record,
            model_name_by_target=model_name_by_target,
        )
        for index, record in enumerate(execution.call_records)
    ]
    return TraceBundle(
        manifest=manifest,
        run=_build_agent_run_trace(execution),
        calls=calls,
    )


def build_llm_trace_record(
    *,
    llm_call: LLMCall,
    decision: ScheduleDecision,
    result: LLMResult,
    timeout: bool = False,
) -> TraceRecord:
    """Build a profiler trace for a completed LLM inference call."""

    _validate_decision(decision, expected_call_id=llm_call.llm_call_id, expected_call_kind="llm")
    _validate_llm_result(llm_call, result)
    execution_time_sec = result.inference_time_sec
    total_latency_sec = (
        result.queue_wait_time_sec
        + result.input_transfer_time_sec
        + execution_time_sec
        + result.output_transfer_time_sec
    )

    return TraceRecord(
        run_id=llm_call.run_id,
        call_id=llm_call.llm_call_id,
        call_kind="llm",
        agent_id=llm_call.agent_id,
        turn_index=llm_call.turn_index,
        selected_target=decision.selected_target,
        policy_name=decision.policy_name,
        queue_wait_time_sec=result.queue_wait_time_sec,
        execution_time_sec=execution_time_sec,
        total_latency_sec=total_latency_sec,
        success=result.success,
        timeout=timeout,
        reward=calculate_call_reward(
            total_latency_sec=total_latency_sec,
            success=result.success,
            timeout=timeout,
        ),
        model_name=result.response_model or llm_call.model_name,
        input_transfer_time_sec=result.input_transfer_time_sec,
        output_transfer_time_sec=result.output_transfer_time_sec,
        input_tokens=llm_call.input_tokens,
        output_tokens=result.output_tokens,
        error_message=result.error_message,
    )


def build_tool_trace_record(
    *,
    tool_call: ToolCall,
    decision: ScheduleDecision,
    result: ToolResult,
    timeout: bool = False,
) -> TraceRecord:
    """Build a profiler trace for a completed Tool call."""

    _validate_decision(
        decision,
        expected_call_id=tool_call.tool_call_id,
        expected_call_kind="tool",
    )
    _validate_tool_result(tool_call, result)
    total_latency_sec = (
        result.queue_wait_time_sec
        + result.input_transfer_time_sec
        + result.execution_time_sec
        + result.output_transfer_time_sec
    )

    return TraceRecord(
        run_id=tool_call.run_id,
        call_id=tool_call.tool_call_id,
        call_kind="tool",
        agent_id=tool_call.agent_id,
        turn_index=tool_call.turn_index,
        selected_target=decision.selected_target,
        policy_name=decision.policy_name,
        queue_wait_time_sec=result.queue_wait_time_sec,
        execution_time_sec=result.execution_time_sec,
        total_latency_sec=total_latency_sec,
        success=result.success,
        timeout=timeout,
        reward=calculate_call_reward(
            total_latency_sec=total_latency_sec,
            success=result.success,
            timeout=timeout,
        ),
        function_call_id=tool_call.call_id,
        tool_name=tool_call.tool_name,
        input_transfer_time_sec=result.input_transfer_time_sec,
        output_transfer_time_sec=result.output_transfer_time_sec,
        error_message=result.error_message,
    )


def calculate_call_reward(
    *,
    total_latency_sec: float,
    success: bool,
    timeout: bool,
    timeout_penalty: float = 20.0,
    failure_penalty: float = 50.0,
) -> float:
    """Compute the first-stage per-call reward."""

    reward = -total_latency_sec
    if timeout:
        reward -= timeout_penalty
    if not success:
        reward -= failure_penalty
    return reward


def _validate_decision(
    decision: ScheduleDecision,
    *,
    expected_call_id: str,
    expected_call_kind: str,
) -> None:
    if decision.call_id != expected_call_id:
        msg = f"decision call_id {decision.call_id!r} does not match {expected_call_id!r}"
        raise ValueError(msg)
    if decision.call_kind != expected_call_kind:
        msg = f"decision call_kind {decision.call_kind!r} does not match {expected_call_kind!r}"
        raise ValueError(msg)


def _validate_llm_result(llm_call: LLMCall, result: LLMResult) -> None:
    if result.llm_call_id != llm_call.llm_call_id:
        msg = f"LLMResult id {result.llm_call_id!r} does not match {llm_call.llm_call_id!r}"
        raise ValueError(msg)


def _validate_tool_result(tool_call: ToolCall, result: ToolResult) -> None:
    if result.tool_call_id != tool_call.tool_call_id:
        msg = f"ToolResult id {result.tool_call_id!r} does not match {tool_call.tool_call_id!r}"
        raise ValueError(msg)


def _build_call_trace(
    *,
    sequence_id: int,
    record: CallExecutionRecord,
    model_name_by_target: dict[str, str],
) -> CallTrace:
    call_payload = sanitize_for_trace(record.call.to_dict())
    result_metadata = sanitize_for_trace(record.result.metadata)
    common = {
        "sequence_id": sequence_id,
        "run_id": record.call.run_id,
        "agent_id": record.call.agent_id,
        "turn_index": record.call.turn_index,
        "call_payload": call_payload,
        "call_digest": content_digest(call_payload),
        "selected_target": record.decision.selected_target,
        "policy_name": record.decision.policy_name,
        "status": record.call.status.value,
        "success": record.result.success,
        "timeout": record.result.error_code == "timeout",
        "result_metadata": result_metadata,
        "estimated_objectives": (
            sanitize_for_trace(record.decision.estimated_objectives)
            if record.decision.estimated_objectives is not None
            else None
        ),
        "error_code": record.result.error_code,
        "error_message": record.result.error_message,
        "created_at": record.call.created_at,
        "decided_at": record.decision.decided_at,
        "finished_at": record.result.finished_at,
    }
    if isinstance(record.call, LLMCall) and isinstance(record.result, LLMResult):
        total_latency = (
            record.result.queue_wait_time_sec
            + record.result.input_transfer_time_sec
            + record.result.inference_time_sec
            + record.result.output_transfer_time_sec
        )
        input_items = call_payload["input_items"]
        return CallTrace(
            **common,
            call_id=record.call.llm_call_id,
            call_kind="llm",
            parameter_summary={
                "context_length": record.call.context_length,
                "estimated_output_tokens": record.call.estimated_output_tokens,
                "input_digest": content_digest(input_items),
                "input_item_count": len(input_items),
                "input_tokens": record.call.input_tokens,
            },
            queue_wait_time_sec=record.result.queue_wait_time_sec,
            input_transfer_time_sec=record.result.input_transfer_time_sec,
            execution_time_sec=record.result.inference_time_sec,
            output_transfer_time_sec=record.result.output_transfer_time_sec,
            total_latency_sec=total_latency,
            energy_joules=record.result.energy_joules,
            model_name=(
                record.result.response_model
                or record.call.model_name
                or model_name_by_target.get(record.decision.selected_target)
            ),
            raw_response_items=sanitize_for_trace(record.result.output_items),
        )
    if isinstance(record.call, ToolCall) and isinstance(record.result, ToolResult):
        total_latency = (
            record.result.queue_wait_time_sec
            + record.result.input_transfer_time_sec
            + record.result.execution_time_sec
            + record.result.output_transfer_time_sec
        )
        arguments = call_payload["arguments"]
        function_output = build_function_call_output(record.call.call_id, record.result)
        return CallTrace(
            **common,
            call_id=record.call.tool_call_id,
            call_kind="tool",
            parameter_summary={
                "argument_keys": sorted(arguments),
                "arguments_digest": content_digest(arguments),
            },
            queue_wait_time_sec=record.result.queue_wait_time_sec,
            input_transfer_time_sec=record.result.input_transfer_time_sec,
            execution_time_sec=record.result.execution_time_sec,
            output_transfer_time_sec=record.result.output_transfer_time_sec,
            total_latency_sec=total_latency,
            energy_joules=record.result.energy_joules,
            tool_name=record.call.tool_name,
            function_call_id=record.call.call_id,
            function_call_output=sanitize_for_trace(function_output),
        )
    raise TypeError("call and result types must match")


def _build_agent_run_trace(execution: ScheduledAgentExecution) -> AgentRunTrace:
    run = execution.agent_run
    if run.finished_at is None:
        raise ValueError("AgentRun must be terminal before it can be traced")
    raw_response_items = [
        deepcopy(item)
        for record in execution.llm_records
        if isinstance(record.result, LLMResult)
        for item in record.result.output_items
    ]
    function_call_outputs = [
        deepcopy(item)
        for item in run.conversation_items
        if item.get("type") == "function_call_output"
    ]
    return AgentRunTrace(
        run_id=run.run_id,
        agent_id=run.agent_id,
        task_id=run.task_id,
        status=run.status.value,
        state_transitions=_derive_state_transitions(execution),
        final_output=run.final_output,
        total_rounds=run.turn_index,
        llm_call_count=len(execution.llm_records),
        tool_call_count=len(execution.tool_records),
        end_to_end_latency_sec=_duration_seconds(run.started_at, run.finished_at),
        started_at=run.started_at,
        finished_at=run.finished_at,
        raw_response_items=sanitize_for_trace(raw_response_items),
        function_call_outputs=sanitize_for_trace(function_call_outputs),
        error_code=run.error_code,
        error_message=run.error_message,
    )


def _derive_state_transitions(execution: ScheduledAgentExecution) -> list[str]:
    transitions = [AgentRunStatus.CREATED.value, AgentRunStatus.READY_FOR_LLM.value]
    records = execution.call_records
    for index, record in enumerate(records):
        next_record = records[index + 1] if index + 1 < len(records) else None
        if isinstance(record.call, LLMCall):
            transitions.append(AgentRunStatus.WAITING_FOR_LLM.value)
            if next_record is not None and isinstance(next_record.call, ToolCall):
                transitions.append(AgentRunStatus.WAITING_FOR_TOOLS.value)
        elif isinstance(record.call, ToolCall) and (
            next_record is not None and isinstance(next_record.call, LLMCall)
        ):
            transitions.append(AgentRunStatus.READY_FOR_LLM.value)
    if transitions[-1] != execution.agent_run.status.value:
        transitions.append(execution.agent_run.status.value)
    return transitions


def _duration_seconds(started_at: str, finished_at: str) -> float:
    duration = (
        datetime.fromisoformat(finished_at) - datetime.fromisoformat(started_at)
    ).total_seconds()
    return max(duration, 0.0)


def _resource_profile_snapshot(resources: ResourceRegistry) -> dict[str, Any]:
    return {
        "llm_instances": [
            {"profile": snapshot.profile.to_dict(), "state": snapshot.state.to_dict()}
            for snapshot in resources.llm_snapshots()
        ],
        "tool_replicas": [
            {"profile": snapshot.profile.to_dict(), "state": snapshot.state.to_dict()}
            for snapshot in resources.tool_snapshots()
        ],
    }
