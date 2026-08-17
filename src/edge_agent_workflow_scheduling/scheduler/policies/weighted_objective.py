"""Weighted multi-objective scheduler policy."""

from __future__ import annotations

from dataclasses import dataclass

from edge_agent_workflow_scheduling.common import SchedulableCall
from edge_agent_workflow_scheduling.scheduler.objectives import (
    ObjectiveNormalization,
    ObjectiveWeights,
    estimate_objectives,
    normalized_cost,
)
from edge_agent_workflow_scheduling.scheduler.policies.common import require_candidates
from edge_agent_workflow_scheduling.scheduler.types import (
    PolicySelection,
    SchedulingCandidate,
)


@dataclass(slots=True)
class WeightedObjectiveSchedulerPolicy:
    """Minimize a normalized, weighted multi-objective cost."""

    weights: ObjectiveWeights
    normalization: ObjectiveNormalization
    name: str = "weighted_objective"

    def select(
        self,
        call: SchedulableCall,
        candidates: list[SchedulingCandidate],
    ) -> PolicySelection:
        require_candidates(candidates)
        scored_candidates = []
        for candidate in candidates:
            objectives = estimate_objectives(call, candidate, candidates)
            cost = normalized_cost(objectives, self.weights, self.normalization)
            scored_candidates.append((cost, candidate, objectives))
        cost, candidate, objectives = min(
            scored_candidates,
            key=lambda item: (item[0], item[1].target_id),
        )
        return PolicySelection(
            candidate=candidate,
            score=cost,
            reason=f"selected {candidate.target_id} with normalized_cost={cost:.6f}",
            estimated_objectives=objectives.to_dict(),
        )
