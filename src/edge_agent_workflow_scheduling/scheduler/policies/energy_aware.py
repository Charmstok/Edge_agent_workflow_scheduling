"""Energy-aware scheduler policy."""

from __future__ import annotations

from dataclasses import dataclass

from edge_agent_workflow_scheduling.common import SchedulableCall
from edge_agent_workflow_scheduling.scheduler.objectives import (
    estimate_energy_joules,
    estimate_latency_sec,
)
from edge_agent_workflow_scheduling.scheduler.policies.common import require_candidates
from edge_agent_workflow_scheduling.scheduler.types import (
    PolicySelection,
    SchedulingCandidate,
)


@dataclass(slots=True)
class EnergyAwareSchedulerPolicy:
    """Minimize profiled energy, then prefer earlier completion."""

    name: str = "energy_aware"

    def select(
        self,
        call: SchedulableCall,
        candidates: list[SchedulingCandidate],
    ) -> PolicySelection:
        require_candidates(candidates)
        scored_candidates = [
            (
                estimate_energy_joules(call, candidate),
                estimate_latency_sec(call, candidate),
                candidate,
            )
            for candidate in candidates
        ]
        energy, finish_time, candidate = min(
            scored_candidates,
            key=lambda item: (item[0], item[1], item[2].target_id),
        )
        return PolicySelection(
            candidate=candidate,
            score=energy,
            reason=(
                f"selected {candidate.target_id} with energy_joules={energy:.6f}, "
                f"estimated_finish_time_sec={finish_time:.6f}"
            ),
        )
