"""Registry for scheduler policies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from edge_agent_workflow_scheduling.scheduler.policies.earliest_finish_time import (
    EarliestFinishTimeSchedulerPolicy,
)
from edge_agent_workflow_scheduling.scheduler.policies.least_queue import (
    LeastQueueSchedulerPolicy,
)
from edge_agent_workflow_scheduling.scheduler.policies.random import RandomSchedulerPolicy
from edge_agent_workflow_scheduling.scheduler.policies.round_robin import (
    RoundRobinSchedulerPolicy,
)
from edge_agent_workflow_scheduling.scheduler.types import SchedulerPolicy

SchedulerPolicyFactory = Callable[[], SchedulerPolicy]


@dataclass(slots=True)
class SchedulerPolicyRegistry:
    """Registry-backed factory for scheduler policies."""

    _factories: dict[str, SchedulerPolicyFactory] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self._factories:
            self.register_factory("random", RandomSchedulerPolicy)
            self.register_factory("round_robin", RoundRobinSchedulerPolicy)
            self.register_factory("least_queue", LeastQueueSchedulerPolicy)
            self.register_factory("earliest_finish_time", EarliestFinishTimeSchedulerPolicy)

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

        self._factories[name] = factory

    def create(self, name: str) -> SchedulerPolicy:
        try:
            return self._factories[name]()
        except KeyError as exc:
            msg = f"scheduler policy must be one of {self.available_policies()}"
            raise ValueError(msg) from exc

    def available_policies(self) -> list[str]:
        return sorted(self._factories)


DEFAULT_SCHEDULER_POLICY_REGISTRY = SchedulerPolicyRegistry()
