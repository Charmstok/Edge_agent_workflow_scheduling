"""Random scheduler policy."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from edge_agent_workflow_scheduling.scheduler.policies.common import require_candidates
from edge_agent_workflow_scheduling.scheduler.types import (
    PolicySelection,
    SchedulingCandidate,
    WorkflowStep,
)


@dataclass(slots=True)
class RandomSchedulerPolicy:
    """Select a random candidate."""

    rng: random.Random = field(default_factory=random.Random)
    name: str = "random"

    def select(
        self,
        step: WorkflowStep,
        candidates: list[SchedulingCandidate],
    ) -> PolicySelection:
        require_candidates(candidates)
        candidate = self.rng.choice(candidates)
        return PolicySelection(
            candidate=candidate,
            reason=f"randomly selected {candidate.target_id}",
        )
