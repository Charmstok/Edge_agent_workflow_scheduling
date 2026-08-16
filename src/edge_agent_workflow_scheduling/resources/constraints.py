"""Scheduling constraints and stable action-mask representations."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from edge_agent_workflow_scheduling.common import LLMCall, SchedulableCall, ToolCall
from edge_agent_workflow_scheduling.resources.models import (
    LLMInstanceProfile,
    ToolReplicaProfile,
)

ResourceProfile = LLMInstanceProfile | ToolReplicaProfile


class MissingQualityProfileError(ValueError):
    """Raised when a quality-dependent decision has no configured quality value."""


@dataclass(frozen=True, slots=True)
class SchedulingConstraints:
    """Additional hard constraints applied before a scheduler policy runs."""

    min_quality: float | None = None
    allowed_node_ids: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if self.min_quality is not None:
            _validate_fraction(self.min_quality, "min_quality")
        if self.allowed_node_ids is not None:
            node_ids = _normalize_node_ids(self.allowed_node_ids, "allowed_node_ids")
            object.__setattr__(self, "allowed_node_ids", node_ids)


@dataclass(frozen=True, slots=True)
class ActionMask:
    """A target-ID-aligned mask plus reasons for rejected targets."""

    target_ids: tuple[str, ...]
    values: tuple[bool, ...]
    rejection_reasons: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        if len(self.target_ids) != len(self.values) or len(self.values) != len(
            self.rejection_reasons
        ):
            raise ValueError("action mask fields must have the same length")
        if len(set(self.target_ids)) != len(self.target_ids):
            raise ValueError("action mask target_ids must be unique")
        if tuple(sorted(self.target_ids)) != self.target_ids:
            raise ValueError("action mask target_ids must use stable sorted order")
        for target_id, is_feasible, reasons in zip(
            self.target_ids,
            self.values,
            self.rejection_reasons,
            strict=True,
        ):
            if not target_id:
                raise ValueError("action mask target IDs must be non-empty")
            if not isinstance(is_feasible, bool):
                raise ValueError("action mask values must be booleans")
            if is_feasible and reasons:
                raise ValueError("feasible action-mask targets cannot have rejection reasons")
            if not is_feasible and not reasons:
                raise ValueError("infeasible action-mask targets require a rejection reason")

    @property
    def eligible_target_ids(self) -> tuple[str, ...]:
        return tuple(
            target_id
            for target_id, is_feasible in zip(self.target_ids, self.values, strict=True)
            if is_feasible
        )

    def as_dict(self) -> dict[str, bool]:
        return dict(zip(self.target_ids, self.values, strict=True))

    def reasons_by_target(self) -> dict[str, tuple[str, ...]]:
        return {
            target_id: reasons
            for target_id, reasons in zip(
                self.target_ids,
                self.rejection_reasons,
                strict=True,
            )
            if reasons
        }


def resolve_scheduling_constraints(
    call: SchedulableCall,
    base: SchedulingConstraints | None = None,
) -> SchedulingConstraints:
    """Merge scheduler constraints with stricter per-call metadata constraints."""

    resolved = base or SchedulingConstraints()
    metadata_min_quality = call.metadata.get("min_quality")
    metadata_node_ids = call.metadata.get("allowed_node_ids")

    min_quality = resolved.min_quality
    if metadata_min_quality is not None:
        _validate_fraction(metadata_min_quality, "call.metadata.min_quality")
        min_quality = (
            metadata_min_quality
            if min_quality is None
            else max(min_quality, metadata_min_quality)
        )

    allowed_node_ids = resolved.allowed_node_ids
    if metadata_node_ids is not None:
        call_node_ids = _normalize_node_ids(
            metadata_node_ids,
            "call.metadata.allowed_node_ids",
        )
        allowed_node_ids = (
            call_node_ids
            if allowed_node_ids is None
            else allowed_node_ids.intersection(call_node_ids)
        )

    return SchedulingConstraints(
        min_quality=min_quality,
        allowed_node_ids=allowed_node_ids,
    )


def task_type_for_call(call: SchedulableCall) -> str:
    """Return the task-type key used to look up a resource quality profile."""

    task_type = call.metadata.get("task_type", "default")
    if not isinstance(task_type, str) or not task_type.strip():
        raise ValueError("call.metadata.task_type must be a non-empty string")
    return task_type


def profiled_quality(call: SchedulableCall, profile: ResourceProfile) -> float:
    """Resolve task-specific quality, treating unprofiled equivalent Tools as 1.0."""

    if isinstance(call, LLMCall) and not isinstance(profile, LLMInstanceProfile):
        raise TypeError("LLMCall quality requires an LLMInstanceProfile")
    if isinstance(call, ToolCall) and not isinstance(profile, ToolReplicaProfile):
        raise TypeError("ToolCall quality requires a ToolReplicaProfile")

    if isinstance(profile, ToolReplicaProfile) and not profile.quality_profile:
        return 1.0

    task_type = task_type_for_call(call)
    try:
        return profile.quality_profile[task_type]
    except KeyError as exc:
        target_id = (
            profile.llm_id if isinstance(profile, LLMInstanceProfile) else profile.replica_id
        )
        raise MissingQualityProfileError(
            f"target {target_id!r} has no quality profile for task_type {task_type!r}"
        ) from exc


def _normalize_node_ids(value: object, field_name: str) -> frozenset[str]:
    if isinstance(value, str) or not isinstance(value, (list, tuple, set, frozenset)):
        raise ValueError(f"{field_name} must be a collection of node IDs")
    node_ids = frozenset(value)
    if any(not isinstance(node_id, str) or not node_id.strip() for node_id in node_ids):
        raise ValueError(f"{field_name} must contain non-empty strings")
    return node_ids


def _validate_fraction(value: object, field_name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{field_name} must be between 0.0 and 1.0")
