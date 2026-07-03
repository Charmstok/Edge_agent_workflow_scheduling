"""Baseline scheduler policies and registry."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field

from edge_agent_workflow_scheduling.common import LLMCall, ToolCall
from edge_agent_workflow_scheduling.scheduler.types import (
    PolicySelection,
    SchedulerPolicy,
    SchedulingCandidate,
    WorkflowStep,
)

SchedulerPolicyFactory = Callable[[], SchedulerPolicy]


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
        _require_candidates(candidates)
        candidate = self.rng.choice(candidates)
        return PolicySelection(
            candidate=candidate,
            reason=f"randomly selected {candidate.target_id}",
        )


@dataclass(slots=True)
class RoundRobinSchedulerPolicy:
    """Rotate through available candidates for each workflow step class."""

    name: str = "round_robin"
    _cursors: dict[str, int] = field(default_factory=dict)

    def select(
        self,
        step: WorkflowStep,
        candidates: list[SchedulingCandidate],
    ) -> PolicySelection:
        _require_candidates(candidates)
        key = _round_robin_key(step)
        ordered_candidates = sorted(candidates, key=lambda candidate: candidate.target_id)
        cursor = self._cursors.get(key, 0)
        candidate = ordered_candidates[cursor % len(ordered_candidates)]
        self._cursors[key] = cursor + 1

        return PolicySelection(
            candidate=candidate,
            score=float(cursor),
            reason=f"round-robin selected {candidate.target_id} for {key}",
        )


@dataclass(slots=True)
class LeastQueueSchedulerPolicy:
    """Select the candidate with the smallest reported queue length."""

    name: str = "least_queue"

    def select(
        self,
        step: WorkflowStep,
        candidates: list[SchedulingCandidate],
    ) -> PolicySelection:
        _require_candidates(candidates)
        candidate = min(candidates, key=lambda item: (item.queue_len, item.target_id))
        return PolicySelection(
            candidate=candidate,
            score=float(candidate.queue_len),
            reason=f"selected {candidate.target_id} with queue_len={candidate.queue_len}",
        )


@dataclass(slots=True)
class EarliestFinishTimeSchedulerPolicy:
    """Select the candidate with the smallest rough finish-time estimate."""

    name: str = "earliest_finish_time"

    def select(
        self,
        step: WorkflowStep,
        candidates: list[SchedulingCandidate],
    ) -> PolicySelection:
        _require_candidates(candidates)
        scored_candidates = [
            (candidate.estimate_finish_time_sec(step), candidate)
            for candidate in candidates
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


@dataclass(slots=True)
class SchedulerPolicyRegistry:
    """Registry-backed factory for scheduler policies."""

    _factories: dict[str, SchedulerPolicyFactory] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self._factories:
            self.register_factory("random", RandomSchedulerPolicy)
            self.register_factory("round_robin", RoundRobinSchedulerPolicy)
            self.register_factory("least_queue", LeastQueueSchedulerPolicy)
            self.register_factory("earliest_finish_time", EarliestFinishTimeSchedulerPolicy)

    def register_factory(
        self,
        name: str,
        factory: SchedulerPolicyFactory,
        *,
        replace: bool = False,
    ) -> None:
        if not name:
            msg = "scheduler policy name must be non-empty"
            raise ValueError(msg)
        if name in self._factories and not replace:
            msg = f"scheduler policy {name!r} is already registered"
            raise ValueError(msg)

        self._factories[name] = factory

    def create(self, name: str) -> SchedulerPolicy:
        try:
            return self._factories[name]()
        except KeyError as exc:
            msg = f"scheduler policy must be one of {self.available_policies()}"
            raise ValueError(msg) from exc

    def available_policies(self) -> list[str]:
        return sorted(self._factories)


DEFAULT_SCHEDULER_POLICY_REGISTRY = SchedulerPolicyRegistry()


def _require_candidates(candidates: list[SchedulingCandidate]) -> None:
    if not candidates:
        msg = "scheduler policy requires at least one candidate"
        raise ValueError(msg)


def _round_robin_key(step: WorkflowStep) -> str:
    if isinstance(step, LLMCall):
        return f"llm:{step.model_name or '*'}"
    if isinstance(step, ToolCall):
        return f"tool:{step.tool_type}"

    msg = "step must be an LLMCall or ToolCall"
    raise TypeError(msg)
