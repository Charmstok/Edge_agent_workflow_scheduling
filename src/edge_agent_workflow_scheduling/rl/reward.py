"""Configurable multi-objective reward calculation for scheduling episodes."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from edge_agent_workflow_scheduling.scheduler.objectives import (
    ObjectiveNormalization,
    ObjectiveVector,
    ObjectiveWeights,
)


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    """One step's raw objectives, normalized terms, and scalar reward."""

    objectives: dict[str, float | int]
    normalized_terms: dict[str, float]
    cost: float
    reward: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MultiObjectiveReward:
    """Convert the shared objective vector into a negative scalar cost.

    Missing energy or quality values are rejected when their configured weight
    is non-zero. This prevents an unavailable metric from silently becoming a
    favorable zero-cost value.
    """

    weights: ObjectiveWeights = ObjectiveWeights(
        latency=1.0, energy=0.0, deadline_miss=0.0, load_imbalance=0.0, quality=0.0
    )
    normalization: ObjectiveNormalization = ObjectiveNormalization(
        latency_ref_sec=1.0, energy_ref_joules=1.0
    )

    def calculate(self, objectives: ObjectiveVector) -> RewardBreakdown:
        normalized_terms = {
            "latency": objectives.latency_sec / self.normalization.latency_ref_sec,
            "energy": objectives.energy_joules / self.normalization.energy_ref_joules,
            "deadline_miss": float(objectives.deadline_miss),
            "load_imbalance": objectives.load_imbalance,
            "quality": 1.0 - objectives.quality,
        }
        weighted = {
            "latency": self.weights.latency * normalized_terms["latency"],
            "energy": self.weights.energy * normalized_terms["energy"],
            "deadline_miss": self.weights.deadline_miss * normalized_terms["deadline_miss"],
            "load_imbalance": self.weights.load_imbalance * normalized_terms["load_imbalance"],
            "quality": self.weights.quality * normalized_terms["quality"],
        }
        cost = sum(weighted.values())
        return RewardBreakdown(
            objectives=objectives.to_dict(),
            normalized_terms=weighted,
            cost=cost,
            reward=-cost,
        )

    def calculate_from_dict(
        self,
        objectives: dict[str, float | int | None],
    ) -> RewardBreakdown:
        missing = []
        if objectives.get("energy_joules") is None and self.weights.energy > 0:
            missing.append("energy_joules")
        if objectives.get("quality") is None and self.weights.quality > 0:
            missing.append("quality")
        if missing:
            raise ValueError(
                "reward requires unavailable objective profile(s): " + ", ".join(missing)
            )
        return self.calculate(
            ObjectiveVector(
                latency_sec=float(_required(objectives, "latency_sec")),
                energy_joules=float(objectives.get("energy_joules") or 0.0),
                quality=float(
                    objectives.get("quality") if objectives.get("quality") is not None else 1.0
                ),
                deadline_miss=int(_required(objectives, "deadline_miss")),
                load_imbalance=float(_required(objectives, "load_imbalance")),
            )
        )


def _required(values: dict[str, object], key: str) -> object:
    value = values.get(key)
    if value is None:
        raise ValueError(f"reward requires objective {key!r}")
    return value
