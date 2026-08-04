"""Scheduled Agent loop connecting calls, resources, and executors."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from math import isfinite
from threading import Lock, Semaphore
from time import monotonic
from typing import Literal, TypeAlias
from uuid import uuid4

from edge_agent_workflow_scheduling.agents.function_calling import (
    extract_output_text,
    tool_call_from_response_item,
)
from edge_agent_workflow_scheduling.common import (
    AgentRun,
    AgentRunStatus,
    CallStatus,
    LLMCall,
    LLMResult,
    SchedulableCall,
    ScheduleDecision,
    ToolCall,
    ToolResult,
)
from edge_agent_workflow_scheduling.executors.base import (
    ExecutorPool,
    LLMExecutor,
    ToolExecutor,
)
from edge_agent_workflow_scheduling.queue import InMemoryCallQueue
from edge_agent_workflow_scheduling.resources import (
    LLMInstanceState,
    ResourceRegistry,
    ToolReplicaState,
)
from edge_agent_workflow_scheduling.scheduler import BaselineScheduler
from edge_agent_workflow_scheduling.tools import ToolRegistry, build_function_call_output

CallResult: TypeAlias = LLMResult | ToolResult
ResourceState: TypeAlias = LLMInstanceState | ToolReplicaState
LifecycleStage = Literal["queued", "running", "completed"]


@dataclass(frozen=True, slots=True)
class CallExecutionRecord:
    """One terminal call together with its target decision and result."""

    call: SchedulableCall
    decision: ScheduleDecision
    result: CallResult


@dataclass(frozen=True, slots=True)
class ResourceStateEvent:
    """Resource load snapshot captured at a call lifecycle boundary."""

    call_id: str
    target_id: str
    stage: LifecycleStage
    state: ResourceState


@dataclass(frozen=True, slots=True)
class ScheduledAgentExecution:
    """Terminal AgentRun and the scheduled calls that produced it."""

    agent_run: AgentRun
    call_records: tuple[CallExecutionRecord, ...]
    state_events: tuple[ResourceStateEvent, ...]

    @property
    def llm_records(self) -> tuple[CallExecutionRecord, ...]:
        return tuple(record for record in self.call_records if isinstance(record.call, LLMCall))

    @property
    def tool_records(self) -> tuple[CallExecutionRecord, ...]:
        return tuple(record for record in self.call_records if isinstance(record.call, ToolCall))


@dataclass(slots=True)
class _RunContext:
    deadline: float
    records: list[CallExecutionRecord] = field(default_factory=list)
    state_events: list[ResourceStateEvent] = field(default_factory=list)
    resource_lock: Lock = field(default_factory=Lock)
    tool_semaphores: dict[str, Semaphore] = field(default_factory=dict)


@dataclass(slots=True)
class AgentRunner:
    """Run a bounded Function Calling Agent through Queue, Scheduler, and Executor."""

    agent_id: str
    system_instruction: str
    tool_registry: ToolRegistry
    resources: ResourceRegistry
    scheduler: BaselineScheduler
    executor_pool: ExecutorPool
    call_queue: InMemoryCallQueue = field(default_factory=InMemoryCallQueue)
    max_rounds: int = 8
    max_tool_calls: int = 16
    timeout_sec: float = 120.0
    model_name: str | None = None
    llm_required_capabilities: list[str] = field(default_factory=lambda: ["function_calling"])

    def __post_init__(self) -> None:
        _validate_non_empty(self.agent_id, "agent_id")
        _validate_non_empty(self.system_instruction, "system_instruction")
        _validate_positive_integer(self.max_rounds, "max_rounds")
        _validate_non_negative_integer(self.max_tool_calls, "max_tool_calls")
        if (
            isinstance(self.timeout_sec, bool)
            or not isinstance(self.timeout_sec, int | float)
            or not isfinite(self.timeout_sec)
            or self.timeout_sec <= 0
        ):
            raise ValueError("timeout_sec must be finite and positive")
        if self.model_name is not None:
            _validate_non_empty(self.model_name, "model_name")
        if not isinstance(self.llm_required_capabilities, list) or any(
            not isinstance(item, str) or not item.strip() for item in self.llm_required_capabilities
        ):
            raise ValueError("llm_required_capabilities must contain non-empty strings")
        if len(set(self.llm_required_capabilities)) != len(self.llm_required_capabilities):
            raise ValueError("llm_required_capabilities must not contain duplicates")

    def run(
        self,
        user_task: str,
        *,
        task_id: str,
        run_id: str | None = None,
    ) -> ScheduledAgentExecution:
        """Execute one dynamic Agent run and always return a terminal state."""

        _validate_non_empty(user_task, "user_task")
        _validate_non_empty(task_id, "task_id")
        if run_id is not None:
            _validate_non_empty(run_id, "run_id")
        if not self.call_queue.is_empty():
            raise ValueError("AgentRunner requires an empty call_queue at run start")

        resolved_run_id = run_id or f"run_{uuid4().hex}"
        agent_run = AgentRun(
            run_id=resolved_run_id,
            agent_id=self.agent_id,
            task_id=task_id,
            conversation_items=[
                {"content": self.system_instruction, "role": "system"},
                {"content": user_task, "role": "user"},
            ],
        )
        agent_run.transition_to(AgentRunStatus.READY_FOR_LLM)
        context = _RunContext(deadline=monotonic() + self.timeout_sec)
        seen_function_call_ids: set[str] = set()
        tool_call_count = 0

        def finish() -> ScheduledAgentExecution:
            self.call_queue.clear()
            return ScheduledAgentExecution(
                agent_run=agent_run,
                call_records=tuple(context.records),
                state_events=tuple(context.state_events),
            )

        while True:
            if _remaining(context) <= 0:
                _fail(agent_run, "timeout", "AgentRun exceeded timeout_sec")
                return finish()
            if agent_run.turn_index >= self.max_rounds:
                _fail(agent_run, "max_rounds_exceeded", "AgentRun reached max_rounds")
                return finish()

            llm_call = LLMCall(
                llm_call_id=f"{resolved_run_id}-llm-{agent_run.turn_index:04d}",
                run_id=resolved_run_id,
                agent_id=self.agent_id,
                turn_index=agent_run.turn_index,
                input_items=deepcopy(agent_run.conversation_items),
                required_capabilities=list(self.llm_required_capabilities),
                model_name=self.model_name,
            )
            agent_run.transition_to(AgentRunStatus.WAITING_FOR_LLM)
            llm_record = self._run_llm_call(llm_call, context)
            if llm_record is None:
                _fail(agent_run, "scheduling_error", "LLMCall could not be scheduled")
                return finish()
            context.records.append(llm_record)
            llm_result = llm_record.result
            if not isinstance(llm_result, LLMResult):
                raise TypeError("LLM call record must contain an LLMResult")
            if not llm_result.success:
                _fail(
                    agent_run,
                    llm_result.error_code or "llm_error",
                    llm_result.error_message or "LLM execution failed",
                )
                return finish()

            agent_run.conversation_items.extend(deepcopy(llm_result.output_items))
            agent_run.turn_index += 1
            function_items = [
                item for item in llm_result.output_items if item.get("type") == "function_call"
            ]
            if not function_items:
                final_output = llm_result.output_text.strip() or extract_output_text(
                    llm_result.output_items
                )
                if not final_output:
                    _fail(
                        agent_run,
                        "invalid_llm_output",
                        "LLM result contained neither function_call nor final text",
                    )
                    return finish()
                agent_run.final_output = final_output
                agent_run.transition_to(AgentRunStatus.COMPLETED)
                return finish()

            agent_run.transition_to(AgentRunStatus.WAITING_FOR_TOOLS)
            if tool_call_count + len(function_items) > self.max_tool_calls:
                _fail(
                    agent_run,
                    "max_tool_calls_exceeded",
                    "AgentRun would exceed max_tool_calls",
                )
                return finish()

            round_calls: list[ToolCall] = []
            for item in function_items:
                try:
                    tool_call = tool_call_from_response_item(
                        item,
                        run_id=resolved_run_id,
                        agent_id=self.agent_id,
                        turn_index=agent_run.turn_index - 1,
                        sequence_id=tool_call_count + len(round_calls),
                    )
                except (TypeError, ValueError) as exc:
                    _fail(agent_run, "invalid_llm_output", str(exc))
                    return finish()
                if tool_call.call_id in seen_function_call_ids:
                    _fail(
                        agent_run,
                        "duplicate_call_id",
                        f"duplicate function call_id {tool_call.call_id!r}",
                    )
                    return finish()
                seen_function_call_ids.add(tool_call.call_id)
                round_calls.append(tool_call)
            tool_call_count += len(round_calls)

            round_records = self._run_tool_round(round_calls, context)
            if round_records is None:
                _fail(agent_run, "scheduling_error", "ToolCall could not be scheduled")
                return finish()
            context.records.extend(round_records)
            agent_run.conversation_items.extend(
                build_function_call_output(tool_call.call_id, record.result)
                for tool_call, record in zip(round_calls, round_records, strict=True)
            )
            failed_record = next(
                (record for record in round_records if not record.result.success),
                None,
            )
            if failed_record is not None:
                result = failed_record.result
                _fail(
                    agent_run,
                    result.error_code or "tool_error",
                    result.error_message or "Tool execution failed",
                )
                return finish()
            agent_run.transition_to(AgentRunStatus.READY_FOR_LLM)

    def _run_llm_call(
        self,
        llm_call: LLMCall,
        context: _RunContext,
    ) -> CallExecutionRecord | None:
        self.call_queue.push(llm_call)
        try:
            decision = self.scheduler.schedule(llm_call, resources=self.resources)
            _save_decision(llm_call, decision)
            with context.resource_lock:
                queued_state = self._change_llm_load(decision.selected_target, queue_delta=1)
                _add_state_event(context, llm_call.llm_call_id, decision, "queued", queued_state)
        except Exception:
            popped = self.call_queue.pop()
            if popped is llm_call:
                llm_call.transition_to(CallStatus.FAILED)
            return None

        popped = self.call_queue.pop()
        if popped is not llm_call:
            raise RuntimeError("call_queue returned an unexpected LLMCall")
        with context.resource_lock:
            llm_call.transition_to(CallStatus.RUNNING)
            running_state = self._change_llm_load(
                decision.selected_target,
                queue_delta=-1,
                running_delta=1,
            )
            _add_state_event(context, llm_call.llm_call_id, decision, "running", running_state)

        snapshot = self.resources.llm_snapshot(decision.selected_target)
        try:
            executor = self.executor_pool.llm_executor(snapshot.profile)
            result = _execute_llm(executor, llm_call, self.tool_registry, _remaining(context))
        except Exception as exc:
            result = LLMResult(
                llm_call_id=llm_call.llm_call_id,
                llm_id=decision.selected_target,
                success=False,
                error_code="llm_executor_error",
                error_message=str(exc) or exc.__class__.__name__,
            )
        with context.resource_lock:
            llm_call.transition_to(CallStatus.SUCCEEDED if result.success else CallStatus.FAILED)
            completed_state = self._complete_llm(decision.selected_target, llm_call, result)
            _add_state_event(
                context,
                llm_call.llm_call_id,
                decision,
                "completed",
                completed_state,
            )
        return CallExecutionRecord(llm_call, decision, result)

    def _run_tool_round(
        self,
        tool_calls: list[ToolCall],
        context: _RunContext,
    ) -> list[CallExecutionRecord] | None:
        assignments: dict[str, tuple[ScheduleDecision, ToolExecutor]] = {}
        try:
            for tool_call in tool_calls:
                self.call_queue.push(tool_call)
                decision = self.scheduler.schedule(tool_call, resources=self.resources)
                _save_decision(tool_call, decision)
                snapshot = self.resources.tool_snapshot(decision.selected_target)
                executor = self.executor_pool.tool_executor(snapshot.profile)
                with context.resource_lock:
                    queued_state = self._change_tool_load(
                        decision.selected_target,
                        queue_delta=1,
                    )
                    _add_state_event(
                        context,
                        tool_call.tool_call_id,
                        decision,
                        "queued",
                        queued_state,
                    )
                assignments[tool_call.tool_call_id] = (decision, executor)
                context.tool_semaphores.setdefault(
                    decision.selected_target,
                    Semaphore(snapshot.profile.max_concurrency - snapshot.state.running_tasks),
                )
        except Exception:
            self._cancel_tool_round(tool_calls, assignments, context)
            return None

        queued_calls: list[ToolCall] = []
        for _ in tool_calls:
            popped = self.call_queue.pop()
            if not isinstance(popped, ToolCall):
                raise RuntimeError("call_queue returned an unexpected call in a Tool round")
            queued_calls.append(popped)

        records_by_id: dict[str, CallExecutionRecord] = {}
        with ThreadPoolExecutor(max_workers=len(queued_calls)) as thread_pool:
            futures = {
                thread_pool.submit(
                    self._execute_tool_assignment,
                    tool_call,
                    assignments[tool_call.tool_call_id][0],
                    assignments[tool_call.tool_call_id][1],
                    context,
                ): tool_call.tool_call_id
                for tool_call in queued_calls
            }
            for future, tool_call_id in futures.items():
                records_by_id[tool_call_id] = future.result()
        return [records_by_id[tool_call.tool_call_id] for tool_call in tool_calls]

    def _execute_tool_assignment(
        self,
        tool_call: ToolCall,
        decision: ScheduleDecision,
        executor: ToolExecutor,
        context: _RunContext,
    ) -> CallExecutionRecord:
        semaphore = context.tool_semaphores[decision.selected_target]
        with semaphore:
            with context.resource_lock:
                tool_call.transition_to(CallStatus.RUNNING)
                running_state = self._change_tool_load(
                    decision.selected_target,
                    queue_delta=-1,
                    running_delta=1,
                )
                _add_state_event(
                    context,
                    tool_call.tool_call_id,
                    decision,
                    "running",
                    running_state,
                )
            result = _execute_tool(executor, tool_call, _remaining(context))
            with context.resource_lock:
                tool_call.transition_to(
                    CallStatus.SUCCEEDED if result.success else CallStatus.FAILED
                )
                completed_state = self._complete_tool(decision.selected_target, result)
                _add_state_event(
                    context,
                    tool_call.tool_call_id,
                    decision,
                    "completed",
                    completed_state,
                )
            return CallExecutionRecord(tool_call, decision, result)

    def _cancel_tool_round(
        self,
        tool_calls: list[ToolCall],
        assignments: dict[str, tuple[ScheduleDecision, ToolExecutor]],
        context: _RunContext,
    ) -> None:
        self.call_queue.clear()
        with context.resource_lock:
            for tool_call in tool_calls:
                assignment = assignments.get(tool_call.tool_call_id)
                if assignment is not None:
                    decision, _ = assignment
                    self._change_tool_load(decision.selected_target, queue_delta=-1)
                if tool_call.status is CallStatus.QUEUED:
                    tool_call.transition_to(CallStatus.FAILED)

    def _change_llm_load(
        self,
        llm_id: str,
        *,
        queue_delta: int = 0,
        running_delta: int = 0,
    ) -> LLMInstanceState:
        state = self.resources.llm_snapshot(llm_id).state
        updated = replace(
            state,
            queue_len=state.queue_len + queue_delta,
            running_requests=state.running_requests + running_delta,
            updated_at=_utc_now_iso(),
        )
        self.resources.update_llm_state(updated)
        return updated

    def _change_tool_load(
        self,
        replica_id: str,
        *,
        queue_delta: int = 0,
        running_delta: int = 0,
    ) -> ToolReplicaState:
        state = self.resources.tool_snapshot(replica_id).state
        updated = replace(
            state,
            queue_len=state.queue_len + queue_delta,
            running_tasks=state.running_tasks + running_delta,
            updated_at=_utc_now_iso(),
        )
        self.resources.update_tool_state(updated)
        return updated

    def _complete_llm(
        self,
        llm_id: str,
        llm_call: LLMCall,
        result: LLMResult,
    ) -> LLMInstanceState:
        state = self.resources.llm_snapshot(llm_id).state
        measured_tokens_per_sec = state.tokens_per_sec
        total_tokens = llm_call.input_tokens + result.output_tokens
        if result.inference_time_sec > 0 and total_tokens > 0:
            measured_tokens_per_sec = total_tokens / result.inference_time_sec
        total_latency = (
            result.queue_wait_time_sec
            + result.input_transfer_time_sec
            + result.inference_time_sec
            + result.output_transfer_time_sec
        )
        updated = replace(
            state,
            running_requests=state.running_requests - 1,
            tokens_per_sec=measured_tokens_per_sec,
            avg_latency_sec=total_latency,
            updated_at=_utc_now_iso(),
        )
        self.resources.update_llm_state(updated)
        return updated

    def _complete_tool(
        self,
        replica_id: str,
        result: ToolResult,
    ) -> ToolReplicaState:
        state = self.resources.tool_snapshot(replica_id).state
        updated = replace(
            state,
            running_tasks=state.running_tasks - 1,
            avg_execution_time_sec=result.execution_time_sec,
            recent_failure_rate=0.0 if result.success else 1.0,
            updated_at=_utc_now_iso(),
        )
        self.resources.update_tool_state(updated)
        return updated


def _execute_llm(
    executor: LLMExecutor,
    llm_call: LLMCall,
    tool_registry: ToolRegistry,
    timeout_sec: float,
) -> LLMResult:
    if timeout_sec <= 0:
        return LLMResult(
            llm_call_id=llm_call.llm_call_id,
            llm_id=executor.profile.llm_id,
            success=False,
            error_code="timeout",
            error_message="LLMCall was not started before AgentRun timeout",
        )
    try:
        result = executor.execute(
            llm_call,
            tools=tool_registry.tools(),
            timeout_sec=timeout_sec,
        )
        if not isinstance(result, LLMResult):
            raise TypeError("LLMExecutor.execute() must return LLMResult")
        if result.llm_call_id != llm_call.llm_call_id:
            raise ValueError("LLMResult call ID does not match LLMCall")
        if result.llm_id != executor.profile.llm_id:
            raise ValueError("LLMResult target ID does not match the selected executor")
        return result
    except Exception as exc:
        return LLMResult(
            llm_call_id=llm_call.llm_call_id,
            llm_id=executor.profile.llm_id,
            success=False,
            error_code="llm_executor_error",
            error_message=str(exc) or exc.__class__.__name__,
        )


def _execute_tool(
    executor: ToolExecutor,
    tool_call: ToolCall,
    timeout_sec: float,
) -> ToolResult:
    if timeout_sec <= 0:
        return ToolResult(
            tool_call_id=tool_call.tool_call_id,
            replica_id=executor.profile.replica_id,
            success=False,
            error_code="timeout",
            error_message="ToolCall was not started before AgentRun timeout",
        )
    try:
        result = executor.execute(tool_call, timeout_sec=timeout_sec)
        if not isinstance(result, ToolResult):
            raise TypeError("ToolExecutor.execute() must return ToolResult")
        if result.tool_call_id != tool_call.tool_call_id:
            raise ValueError("ToolResult call ID does not match ToolCall")
        if result.replica_id != executor.profile.replica_id:
            raise ValueError("ToolResult target ID does not match the selected executor")
        return result
    except Exception as exc:
        return ToolResult(
            tool_call_id=tool_call.tool_call_id,
            replica_id=executor.profile.replica_id,
            success=False,
            error_code="tool_executor_error",
            error_message=str(exc) or exc.__class__.__name__,
        )


def _save_decision(call: SchedulableCall, decision: ScheduleDecision) -> None:
    call.metadata["schedule_decision"] = decision.to_dict()


def _add_state_event(
    context: _RunContext,
    call_id: str,
    decision: ScheduleDecision,
    stage: LifecycleStage,
    state: ResourceState,
) -> None:
    context.state_events.append(
        ResourceStateEvent(
            call_id=call_id,
            target_id=decision.selected_target,
            stage=stage,
            state=deepcopy(state),
        )
    )


def _remaining(context: _RunContext) -> float:
    return context.deadline - monotonic()


def _fail(agent_run: AgentRun, error_code: str, error_message: str) -> None:
    agent_run.error_code = error_code
    agent_run.error_message = error_message
    agent_run.transition_to(AgentRunStatus.FAILED)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _validate_non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _validate_positive_integer(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")


def _validate_non_negative_integer(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
