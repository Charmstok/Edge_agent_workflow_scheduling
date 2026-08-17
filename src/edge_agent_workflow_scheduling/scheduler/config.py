"""Configuration shared by scheduler policy implementations."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from edge_agent_workflow_scheduling.scheduler.objectives import (
    ObjectiveNormalization,
    ObjectiveWeights,
)


@dataclass(frozen=True, slots=True)
class SchedulerPolicyConfig:
    """Reproducible configuration passed to policies through the registry."""

    random_seed: int | None = None
    record_objectives: bool = False
    objective_weights: ObjectiveWeights | None = None
    objective_normalization: ObjectiveNormalization | None = None

    def __post_init__(self) -> None:
        if self.random_seed is not None and (
            isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int)
        ):
            raise ValueError("random_seed must be an integer or None")
        if not isinstance(self.record_objectives, bool):
            raise ValueError("record_objectives must be a boolean")
        if (self.objective_weights is None) != (self.objective_normalization is None):
            raise ValueError(
                "objective_weights and objective_normalization must be configured together"
            )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
