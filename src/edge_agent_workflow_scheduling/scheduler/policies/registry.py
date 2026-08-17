"""Registry for scheduler policies."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field

from edge_agent_workflow_scheduling.scheduler.config import SchedulerPolicyConfig
from edge_agent_workflow_scheduling.scheduler.policies.earliest_finish_time import (
    EarliestFinishTimeSchedulerPolicy,
)
from edge_agent_workflow_scheduling.scheduler.policies.energy_aware import (
    EnergyAwareSchedulerPolicy,
)
from edge_agent_workflow_scheduling.scheduler.policies.least_queue import (
    LeastQueueSchedulerPolicy,
)
from edge_agent_workflow_scheduling.scheduler.policies.quality_aware import (
    QualityAwareSchedulerPolicy,
)
from edge_agent_workflow_scheduling.scheduler.policies.quality_constrained_eft import (
    QualityConstrainedEarliestFinishTimeSchedulerPolicy,
)
from edge_agent_workflow_scheduling.scheduler.policies.random import RandomSchedulerPolicy
from edge_agent_workflow_scheduling.scheduler.policies.round_robin import (
    RoundRobinSchedulerPolicy,
)
from edge_agent_workflow_scheduling.scheduler.policies.weighted_objective import (
    WeightedObjectiveSchedulerPolicy,
)
from edge_agent_workflow_scheduling.scheduler.types import SchedulerPolicy

SchedulerPolicyFactory = Callable[[], SchedulerPolicy]
ConfiguredSchedulerPolicyFactory = Callable[[SchedulerPolicyConfig], SchedulerPolicy]


@dataclass(slots=True)
class SchedulerPolicyRegistry:
    """Registry-backed factory for scheduler policies."""

    _factories: dict[str, ConfiguredSchedulerPolicyFactory] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self._factories:
            self.register_configured_factory(
                "random",
                lambda config: RandomSchedulerPolicy(rng=random.Random(config.random_seed)),
            )
            self.register_factory("round_robin", RoundRobinSchedulerPolicy)
            self.register_factory("least_queue", LeastQueueSchedulerPolicy)
            self.register_factory("earliest_finish_time", EarliestFinishTimeSchedulerPolicy)
            self.register_factory("quality_aware", QualityAwareSchedulerPolicy)
            self.register_factory("energy_aware", EnergyAwareSchedulerPolicy)
            self.register_configured_factory("weighted_objective", _create_weighted_objective)
            self.register_factory(
                "quality_constrained_earliest_finish_time",
                QualityConstrainedEarliestFinishTimeSchedulerPolicy,
            )

    def register_factory(
        self,
        name: str,
        factory: SchedulerPolicyFactory,
        *,
        replace: bool = False,
    ) -> None:
        if not name:
            msg = "scheduler policy name must be non-empty"
            raise ValueError(msg)
        if name in self._factories and not replace:
            msg = f"scheduler policy {name!r} is already registered"
            raise ValueError(msg)

        self.register_configured_factory(name, lambda config: factory(), replace=replace)

    def register_configured_factory(
        self,
        name: str,
        factory: ConfiguredSchedulerPolicyFactory,
        *,
        replace: bool = False,
    ) -> None:
        """Register a policy factory that consumes shared scheduler configuration."""

        if not name:
            msg = "scheduler policy name must be non-empty"
            raise ValueError(msg)
        if name in self._factories and not replace:
            msg = f"scheduler policy {name!r} is already registered"
            raise ValueError(msg)
        self._factories[name] = factory

    def create(
        self,
        name: str,
        config: SchedulerPolicyConfig | None = None,
    ) -> SchedulerPolicy:
        try:
            factory = self._factories[name]
        except KeyError as exc:
            msg = f"scheduler policy must be one of {self.available_policies()}"
            raise ValueError(msg) from exc
        return factory(config or SchedulerPolicyConfig())

    def available_policies(self) -> list[str]:
        return sorted(self._factories)


def _create_weighted_objective(config: SchedulerPolicyConfig) -> SchedulerPolicy:
    if config.objective_weights is None or config.objective_normalization is None:
        raise ValueError(
            "weighted_objective requires objective_weights and objective_normalization"
        )
    return WeightedObjectiveSchedulerPolicy(
        weights=config.objective_weights,
        normalization=config.objective_normalization,
    )


DEFAULT_SCHEDULER_POLICY_REGISTRY = SchedulerPolicyRegistry()
