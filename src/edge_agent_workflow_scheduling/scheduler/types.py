"""Shared scheduler types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from edge_agent_workflow_scheduling.common import (
    LLMCall,
    LLMInstanceState,
    ToolCall,
    WorkerState,
)

WorkflowStep = LLMCall | ToolCall
TaskKind = Literal["llm", "tool"]
ExecutionState = LLMInstanceState | WorkerState


@dataclass(frozen=True, slots=True)
class SchedulingCandidate:
    """A filtered execution target that can run a workflow step."""

    target_id: str
    task_kind: TaskKind
    state: ExecutionState

    @property
    def queue_len(self) -> int:
        return self.state.queue_len

    def estimate_finish_time_sec(self, step: WorkflowStep) -> float:
        if isinstance(step, LLMCall) and isinstance(self.state, LLMInstanceState):
            return _estimate_llm_finish_time_sec(step, self.state)
        if isinstance(step, ToolCall) and isinstance(self.state, WorkerState):
            return _estimate_worker_finish_time_sec(self.state)

        msg = "candidate state does not match workflow step type"
        raise TypeError(msg)


@dataclass(frozen=True, slots=True)
class PolicySelection:
    """A scheduler policy's target choice."""

    candidate: SchedulingCandidate
    score: float | None = None
    reason: str | None = None


class SchedulerPolicy(Protocol):
    """Policy interface for baseline scheduler target selection."""

    name: str

    def select(
        self,
        step: WorkflowStep,
        candidates: list[SchedulingCandidate],
    ) -> PolicySelection:
        """Select one candidate from a non-empty candidate list."""


def task_kind_for_step(step: WorkflowStep) -> TaskKind:
    if isinstance(step, LLMCall):
        return "llm"
    if isinstance(step, ToolCall):
        return "tool"

    msg = "step must be an LLMCall or ToolCall"
    raise TypeError(msg)


def task_id_for_step(step: WorkflowStep) -> str:
    if isinstance(step, LLMCall):
        return step.llm_call_id
    if isinstance(step, ToolCall):
        return step.tool_call_id

    msg = "step must be an LLMCall or ToolCall"
    raise TypeError(msg)


def _estimate_llm_finish_time_sec(step: LLMCall, state: LLMInstanceState) -> float:
    queue_unit_sec = state.avg_latency_sec if state.avg_latency_sec > 0 else 1.0
    queue_delay_sec = state.queue_len * queue_unit_sec
    total_tokens = step.input_tokens + step.estimated_output_tokens
    inference_time_sec = total_tokens / state.tokens_per_sec if state.tokens_per_sec > 0 else float(
        "inf",
    )
    return queue_delay_sec + inference_time_sec


def _estimate_worker_finish_time_sec(state: WorkerState) -> float:
    network_latency_sec = state.network_latency_ms / 1000
    return state.queue_len + network_latency_sec + state.cpu_util + state.memory_util
