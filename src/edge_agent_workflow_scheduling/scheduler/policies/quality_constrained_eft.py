"""Quality-constrained earliest-finish-time scheduler policy."""

from __future__ import annotations

from dataclasses import dataclass, field

from edge_agent_workflow_scheduling.common import SchedulableCall
from edge_agent_workflow_scheduling.scheduler.objectives import estimate_latency_sec
from edge_agent_workflow_scheduling.scheduler.policies.earliest_finish_time import (
    EarliestFinishTimeSchedulerPolicy,
)
from edge_agent_workflow_scheduling.scheduler.types import (
    PolicySelection,
    SchedulingCandidate,
)


@dataclass(slots=True)
class QualityConstrainedEarliestFinishTimeSchedulerPolicy:
    """Apply EFT after the shared action mask enforces minimum quality."""

    name: str = "quality_constrained_earliest_finish_time"
    requires_min_quality: bool = field(default=True, init=False)
    _earliest_finish_time: EarliestFinishTimeSchedulerPolicy = field(
        default_factory=EarliestFinishTimeSchedulerPolicy,
        init=False,
        repr=False,
    )

    def select(
        self,
        call: SchedulableCall,
        candidates: list[SchedulingCandidate],
    ) -> PolicySelection:
        for candidate in candidates:
            estimate_latency_sec(call, candidate)
        selection = self._earliest_finish_time.select(call, candidates)
        return PolicySelection(
            candidate=selection.candidate,
            score=selection.score,
            reason=f"selected from quality-feasible candidates; {selection.reason}",
        )
