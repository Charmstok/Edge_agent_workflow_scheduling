"""Baseline scheduler implementation."""

from __future__ import annotations

from dataclasses import dataclass, field

from edge_agent_workflow_scheduling.common import (
    LLMCall,
    LLMInstanceState,
    SchedulableCall,
    ScheduleDecision,
    ToolCall,
    WorkerState,
)
from edge_agent_workflow_scheduling.scheduler.policies import (
    DEFAULT_SCHEDULER_POLICY_REGISTRY,
    SchedulerPolicyRegistry,
)
from edge_agent_workflow_scheduling.scheduler.types import (
    SchedulerPolicy,
    SchedulingCandidate,
    call_id_for,
    call_kind_for,
)


@dataclass(slots=True)
class BaselineScheduler:
    """Scheduler that delegates target selection to a registered baseline policy."""

    policy_name: str
    policy_registry: SchedulerPolicyRegistry | None = None
    _policy: SchedulerPolicy = field(init=False)

    def __post_init__(self) -> None:
        registry = self.policy_registry or DEFAULT_SCHEDULER_POLICY_REGISTRY
        self.policy_registry = registry
        self._policy = registry.create(self.policy_name)

    def schedule(
        self,
        call: SchedulableCall,
        *,
        llm_states: list[LLMInstanceState],
        worker_states: list[WorkerState],
    ) -> ScheduleDecision:
        """Choose an execution target for an LLMCall or ToolCall."""

        candidates = self._build_candidates(
            call,
            llm_states=llm_states,
            worker_states=worker_states,
        )
        if not candidates:
            msg = f"no available execution targets for {call_kind_for(call)} call"
            raise ValueError(msg)

        selection = self._policy.select(call, candidates)
        return ScheduleDecision(
            call_id=call_id_for(call),
            call_kind=call_kind_for(call),
            selected_target=selection.candidate.target_id,
            policy_name=self._policy.name,
            score=selection.score,
            reason=selection.reason,
        )

    def _build_candidates(
        self,
        call: SchedulableCall,
        *,
        llm_states: list[LLMInstanceState],
        worker_states: list[WorkerState],
    ) -> list[SchedulingCandidate]:
        if isinstance(call, LLMCall):
            return [
                SchedulingCandidate(target_id=state.llm_id, call_kind="llm", state=state)
                for state in llm_states
                if _can_run_llm_call(call, state)
            ]
        if isinstance(call, ToolCall):
            return [
                SchedulingCandidate(target_id=state.worker_id, call_kind="tool", state=state)
                for state in worker_states
                if _can_run_tool_call(call, state)
            ]

        msg = "call must be an LLMCall or ToolCall"
        raise TypeError(msg)


def _can_run_llm_call(call: LLMCall, state: LLMInstanceState) -> bool:
    if not state.is_online:
        return False
    return call.model_name is None or call.model_name == state.model_name


def _can_run_tool_call(call: ToolCall, state: WorkerState) -> bool:
    return state.is_online and call.tool_name in state.supported_tools
