"""Least-queue scheduler policy."""

from __future__ import annotations

from dataclasses import dataclass

from edge_agent_workflow_scheduling.common import SchedulableCall
from edge_agent_workflow_scheduling.scheduler.policies.common import require_candidates
from edge_agent_workflow_scheduling.scheduler.types import (
    PolicySelection,
    SchedulingCandidate,
)


@dataclass(slots=True)
class LeastQueueSchedulerPolicy:
    """Select the candidate with the smallest reported queue length."""

    name: str = "least_queue"

    def select(
        self,
        call: SchedulableCall,
        candidates: list[SchedulingCandidate],
    ) -> PolicySelection:
        require_candidates(candidates)
        candidate = min(candidates, key=lambda item: (item.queue_len, item.target_id))
        return PolicySelection(
            candidate=candidate,
            score=float(candidate.queue_len),
            reason=f"selected {candidate.target_id} with queue_len={candidate.queue_len}",
        )
