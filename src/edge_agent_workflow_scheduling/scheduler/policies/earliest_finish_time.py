"""Earliest-finish-time scheduler policy."""

from __future__ import annotations

from dataclasses import dataclass

from edge_agent_workflow_scheduling.common import SchedulableCall
from edge_agent_workflow_scheduling.scheduler.policies.common import require_candidates
from edge_agent_workflow_scheduling.scheduler.types import (
    PolicySelection,
    SchedulingCandidate,
)


@dataclass(slots=True)
class EarliestFinishTimeSchedulerPolicy:
    """Select the candidate with the smallest rough finish-time estimate."""

    name: str = "earliest_finish_time"

    def select(
        self,
        call: SchedulableCall,
        candidates: list[SchedulingCandidate],
    ) -> PolicySelection:
        require_candidates(candidates)
        scored_candidates = [
            (candidate.estimate_finish_time_sec(call), candidate) for candidate in candidates
        ]
        score, candidate = min(
            scored_candidates,
            key=lambda item: (item[0], item[1].target_id),
        )
        return PolicySelection(
            candidate=candidate,
            score=score,
            reason=f"selected {candidate.target_id} with estimated_finish_time_sec={score:.6f}",
        )
