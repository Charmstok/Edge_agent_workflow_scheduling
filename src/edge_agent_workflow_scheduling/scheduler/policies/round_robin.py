"""Round-robin scheduler policy."""

from __future__ import annotations

from dataclasses import dataclass, field

from edge_agent_workflow_scheduling.common import SchedulableCall
from edge_agent_workflow_scheduling.scheduler.policies.common import (
    require_candidates,
    round_robin_key,
)
from edge_agent_workflow_scheduling.scheduler.types import (
    PolicySelection,
    SchedulingCandidate,
)


@dataclass(slots=True)
class RoundRobinSchedulerPolicy:
    """Rotate through available candidates for each call class."""

    name: str = "round_robin"
    _cursors: dict[str, int] = field(default_factory=dict)

    def select(
        self,
        call: SchedulableCall,
        candidates: list[SchedulingCandidate],
    ) -> PolicySelection:
        require_candidates(candidates)
        key = round_robin_key(call)
        ordered_candidates = sorted(candidates, key=lambda candidate: candidate.target_id)
        cursor = self._cursors.get(key, 0)
        candidate = ordered_candidates[cursor % len(ordered_candidates)]
        self._cursors[key] = cursor + 1

        return PolicySelection(
            candidate=candidate,
            score=float(cursor),
            reason=f"round-robin selected {candidate.target_id} for {key}",
        )
