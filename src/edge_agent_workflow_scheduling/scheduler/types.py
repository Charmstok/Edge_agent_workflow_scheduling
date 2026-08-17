"""Shared scheduler types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from edge_agent_workflow_scheduling.common import LLMCall, SchedulableCall, ToolCall
from edge_agent_workflow_scheduling.resources import (
    LLMInstanceProfile,
    LLMInstanceSnapshot,
    LLMInstanceState,
    ToolReplicaProfile,
    ToolReplicaSnapshot,
    ToolReplicaState,
)

CallKind = Literal["llm", "tool"]
ExecutionState = LLMInstanceState | ToolReplicaState
ExecutionProfile = LLMInstanceProfile | ToolReplicaProfile


@dataclass(frozen=True, slots=True)
class SchedulingCandidate:
    """A filtered execution target that can run a schedulable call."""

    target_id: str
    call_kind: CallKind
    profile: ExecutionProfile
    state: ExecutionState

    @property
    def queue_len(self) -> int:
        return self.state.queue_len

    def estimate_finish_time_sec(self, call: SchedulableCall) -> float:
        if (
            isinstance(call, LLMCall)
            and isinstance(self.profile, LLMInstanceProfile)
            and isinstance(self.state, LLMInstanceState)
        ):
            return _estimate_llm_finish_time_sec(call, self.profile, self.state)
        if (
            isinstance(call, ToolCall)
            and isinstance(self.profile, ToolReplicaProfile)
            and isinstance(self.state, ToolReplicaState)
        ):
            return _estimate_tool_finish_time_sec(self.profile, self.state)

        msg = "candidate state does not match call type"
        raise TypeError(msg)


@dataclass(frozen=True, slots=True)
class PolicySelection:
    """A scheduler policy's target choice."""

    candidate: SchedulingCandidate
    score: float | None = None
    reason: str | None = None
    estimated_objectives: dict[str, float | int] | None = None


class SchedulerPolicy(Protocol):
    """Policy interface for baseline scheduler target selection."""

    name: str

    def select(
        self,
        call: SchedulableCall,
        candidates: list[SchedulingCandidate],
    ) -> PolicySelection:
        """Select one candidate from a non-empty candidate list."""


def call_kind_for(call: SchedulableCall) -> CallKind:
    if isinstance(call, LLMCall):
        return "llm"
    if isinstance(call, ToolCall):
        return "tool"

    msg = "call must be an LLMCall or ToolCall"
    raise TypeError(msg)


def call_id_for(call: SchedulableCall) -> str:
    if isinstance(call, LLMCall):
        return call.llm_call_id
    if isinstance(call, ToolCall):
        return call.tool_call_id

    msg = "call must be an LLMCall or ToolCall"
    raise TypeError(msg)


def candidate_from_snapshot(
    snapshot: LLMInstanceSnapshot | ToolReplicaSnapshot,
) -> SchedulingCandidate:
    if isinstance(snapshot, LLMInstanceSnapshot):
        return SchedulingCandidate(
            target_id=snapshot.profile.llm_id,
            call_kind="llm",
            profile=snapshot.profile,
            state=snapshot.state,
        )
    if isinstance(snapshot, ToolReplicaSnapshot):
        return SchedulingCandidate(
            target_id=snapshot.profile.replica_id,
            call_kind="tool",
            profile=snapshot.profile,
            state=snapshot.state,
        )
    raise TypeError("snapshot must be an LLMInstanceSnapshot or ToolReplicaSnapshot")


def _estimate_llm_finish_time_sec(
    call: LLMCall,
    profile: LLMInstanceProfile,
    state: LLMInstanceState,
) -> float:
    queue_unit_sec = state.avg_latency_sec if state.avg_latency_sec > 0 else 1.0
    queue_delay_sec = state.queue_len * queue_unit_sec
    total_tokens = call.input_tokens + call.estimated_output_tokens
    profiled_tokens_per_sec = profile.token_profile.get("tokens_per_sec", 0.0)
    tokens_per_sec = state.tokens_per_sec or profiled_tokens_per_sec
    inference_time_sec = (
        total_tokens / tokens_per_sec
        if tokens_per_sec > 0
        else float(
            "inf",
        )
    )
    return queue_delay_sec + inference_time_sec


def _estimate_tool_finish_time_sec(
    profile: ToolReplicaProfile,
    state: ToolReplicaState,
) -> float:
    network_latency_sec = state.network_latency_ms / 1000
    profiled_execution_sec = profile.latency_profile.get("execution_time_sec", 0.0)
    execution_time_sec = state.avg_execution_time_sec or profiled_execution_sec
    return state.queue_len * execution_time_sec + network_latency_sec + execution_time_sec
