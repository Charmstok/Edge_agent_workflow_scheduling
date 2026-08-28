"""Baseline scheduler implementation."""

from __future__ import annotations

from dataclasses import dataclass, field

from edge_agent_workflow_scheduling.common import SchedulableCall, ScheduleDecision
from edge_agent_workflow_scheduling.resources import (
    ActionMask,
    ResourceRegistry,
    SchedulingConstraints,
    resolve_scheduling_constraints,
)
from edge_agent_workflow_scheduling.scheduler.config import SchedulerPolicyConfig
from edge_agent_workflow_scheduling.scheduler.objectives import estimate_objectives_dict
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


class NoFeasibleTargetError(ValueError):
    """Raised when hard constraints reject every registered target."""

    def __init__(self, call_kind: str, action_mask: ActionMask) -> None:
        self.call_kind = call_kind
        self.action_mask = action_mask
        reasons = action_mask.reasons_by_target()
        detail = "; ".join(
            f"{target_id}={','.join(target_reasons)}"
            for target_id, target_reasons in reasons.items()
        )
        if not detail:
            detail = "no targets are registered"
        super().__init__(f"no feasible execution targets for {call_kind} call: {detail}")


@dataclass(slots=True)
class BaselineScheduler:
    """Scheduler that delegates target selection to a registered baseline policy."""

    policy_name: str
    policy_registry: SchedulerPolicyRegistry | None = None
    constraints: SchedulingConstraints = field(default_factory=SchedulingConstraints)
    policy_config: SchedulerPolicyConfig = field(default_factory=SchedulerPolicyConfig)
    _policy: SchedulerPolicy = field(init=False)

    def __post_init__(self) -> None:
        registry = self.policy_registry or DEFAULT_SCHEDULER_POLICY_REGISTRY
        self.policy_registry = registry
        self._policy = registry.create(self.policy_name, self.policy_config)

    def schedule(
        self,
        call: SchedulableCall,
        *,
        resources: ResourceRegistry,
    ) -> ScheduleDecision:
        """Choose an execution target for an LLMCall or ToolCall."""

        resolved_constraints = resolve_scheduling_constraints(call, self.constraints)
        if (
            getattr(self._policy, "requires_min_quality", False)
            and resolved_constraints.min_quality is None
        ):
            raise ValueError(
                "quality_constrained_earliest_finish_time requires min_quality"
            )

        action_mask = resources.action_mask_details(call, constraints=resolved_constraints)
        candidates = [
            candidate_from_snapshot(snapshot)
            for snapshot in resources.eligible_snapshots(
                call,
                constraints=resolved_constraints,
            )
        ]
        if not candidates:
            raise NoFeasibleTargetError(call_kind_for(call), action_mask)

        selection = self._policy.select(call, candidates)
        estimated_objectives = selection.estimated_objectives
        if self.policy_config.record_objectives and estimated_objectives is None:
            estimated_objectives = estimate_objectives_dict(
                call,
                selection.candidate,
                candidates,
                allow_missing_optional_profiles=self.policy_name
                not in {
                    "quality_aware",
                    "weighted_objective",
                    "quality_constrained_earliest_finish_time",
                },
            )
        return ScheduleDecision(
            call_id=call_id_for(call),
            call_kind=call_kind_for(call),
            selected_target=selection.candidate.target_id,
            policy_name=self._policy.name,
            score=selection.score,
            reason=selection.reason,
            candidate_target_ids=list(action_mask.target_ids),
            action_mask=list(action_mask.values),
            rejection_reasons={
                target_id: list(reasons)
                for target_id, reasons in action_mask.reasons_by_target().items()
            },
            estimated_objectives=estimated_objectives,
        )

    def manifest_parameters(self) -> dict[str, object]:
        """Return public scheduler parameters needed to reproduce a decision."""

        parameters = self.policy_config.to_dict()
        parameters["constraints"] = {
            "min_quality": self.constraints.min_quality,
            "allowed_node_ids": (
                sorted(self.constraints.allowed_node_ids)
                if self.constraints.allowed_node_ids is not None
                else None
            ),
        }
        return parameters
