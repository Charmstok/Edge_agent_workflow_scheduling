"""Baseline scheduler implementation."""

from __future__ import annotations

from dataclasses import dataclass, field

from edge_agent_workflow_scheduling.common import SchedulableCall, ScheduleDecision
from edge_agent_workflow_scheduling.resources import ResourceRegistry
from edge_agent_workflow_scheduling.scheduler.policies import (
    DEFAULT_SCHEDULER_POLICY_REGISTRY,
    SchedulerPolicyRegistry,
)
from edge_agent_workflow_scheduling.scheduler.types import (
    SchedulerPolicy,
    call_id_for,
    call_kind_for,
    candidate_from_snapshot,
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
        resources: ResourceRegistry,
    ) -> ScheduleDecision:
        """Choose an execution target for an LLMCall or ToolCall."""

        candidates = [
            candidate_from_snapshot(snapshot) for snapshot in resources.eligible_snapshots(call)
        ]
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
