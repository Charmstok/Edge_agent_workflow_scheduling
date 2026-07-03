"""Baseline scheduler implementation."""

from __future__ import annotations

from dataclasses import dataclass, field

from edge_agent_workflow_scheduling.common import (
    LLMCall,
    LLMInstanceState,
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
    WorkflowStep,
    task_id_for_step,
    task_kind_for_step,
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
        step: WorkflowStep,
        *,
        llm_states: list[LLMInstanceState],
        worker_states: list[WorkerState],
    ) -> ScheduleDecision:
        """Choose an execution target for an LLMCall or ToolCall."""

        candidates = self._build_candidates(
            step,
            llm_states=llm_states,
            worker_states=worker_states,
        )
        if not candidates:
            msg = f"no available execution targets for {task_kind_for_step(step)} step"
            raise ValueError(msg)

        selection = self._policy.select(step, candidates)
        return ScheduleDecision(
            task_id=task_id_for_step(step),
            task_kind=task_kind_for_step(step),
            selected_target=selection.candidate.target_id,
            policy_name=self._policy.name,
            score=selection.score,
            reason=selection.reason,
        )

    def _build_candidates(
        self,
        step: WorkflowStep,
        *,
        llm_states: list[LLMInstanceState],
        worker_states: list[WorkerState],
    ) -> list[SchedulingCandidate]:
        if isinstance(step, LLMCall):
            return [
                SchedulingCandidate(target_id=state.llm_id, task_kind="llm", state=state)
                for state in llm_states
                if _can_run_llm_call(step, state)
            ]
        if isinstance(step, ToolCall):
            return [
                SchedulingCandidate(target_id=state.worker_id, task_kind="tool", state=state)
                for state in worker_states
                if _can_run_tool_call(step, state)
            ]

        msg = "step must be an LLMCall or ToolCall"
        raise TypeError(msg)


def _can_run_llm_call(step: LLMCall, state: LLMInstanceState) -> bool:
    if not state.is_online:
        return False
    return step.model_name is None or step.model_name == state.model_name


def _can_run_tool_call(step: ToolCall, state: WorkerState) -> bool:
    return state.is_online and step.tool_type in state.supported_tools
