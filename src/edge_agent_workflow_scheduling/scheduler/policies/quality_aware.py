"""Task-quality-aware scheduler policy."""

from __future__ import annotations

from dataclasses import dataclass

from edge_agent_workflow_scheduling.common import SchedulableCall
from edge_agent_workflow_scheduling.resources import profiled_quality
from edge_agent_workflow_scheduling.scheduler.objectives import estimate_latency_sec
from edge_agent_workflow_scheduling.scheduler.policies.common import require_candidates
from edge_agent_workflow_scheduling.scheduler.types import (
    PolicySelection,
    SchedulingCandidate,
)


@dataclass(slots=True)
class QualityAwareSchedulerPolicy:
    """Maximize profiled quality, then prefer earlier completion."""

    name: str = "quality_aware"

    def select(
        self,
        call: SchedulableCall,
        candidates: list[SchedulingCandidate],
    ) -> PolicySelection:
        require_candidates(candidates)
        scored_candidates = [
            (
                profiled_quality(call, candidate.profile),
                estimate_latency_sec(call, candidate),
                candidate,
            )
            for candidate in candidates
        ]
        quality, finish_time, candidate = min(
            scored_candidates,
            key=lambda item: (-item[0], item[1], item[2].target_id),
        )
        return PolicySelection(
            candidate=candidate,
            score=quality,
            reason=(
                f"selected {candidate.target_id} with quality={quality:.6f}, "
                f"estimated_finish_time_sec={finish_time:.6f}"
            ),
        )
