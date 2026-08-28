"""Profile-based multi-objective estimates for baseline scheduling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isclose, isfinite
from statistics import fmean, pstdev

from edge_agent_workflow_scheduling.common import LLMCall, SchedulableCall, ToolCall
from edge_agent_workflow_scheduling.resources import (
    LLMInstanceProfile,
    LLMInstanceState,
    MissingQualityProfileError,
    ToolReplicaProfile,
    ToolReplicaState,
    profiled_quality,
)
from edge_agent_workflow_scheduling.scheduler.types import SchedulingCandidate


class MissingObjectiveProfileError(ValueError):
    """Raised when a requested objective cannot be estimated from a profile."""


@dataclass(frozen=True, slots=True)
class ObjectiveVector:
    """Estimated objectives for assigning one call to one candidate."""

    latency_sec: float
    energy_joules: float
    quality: float
    deadline_miss: int
    load_imbalance: float

    def __post_init__(self) -> None:
        _validate_non_negative(self.latency_sec, "latency_sec")
        _validate_non_negative(self.energy_joules, "energy_joules")
        _validate_fraction(self.quality, "quality")
        if self.deadline_miss not in {0, 1} or isinstance(self.deadline_miss, bool):
            raise ValueError("deadline_miss must be 0 or 1")
        _validate_non_negative(self.load_imbalance, "load_imbalance")

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ObjectiveWeights:
    """Non-negative objective weights normalized to a sum of one."""

    latency: float
    energy: float
    deadline_miss: float
    load_imbalance: float
    quality: float

    def __post_init__(self) -> None:
        values = (
            self.latency,
            self.energy,
            self.deadline_miss,
            self.load_imbalance,
            self.quality,
        )
        for value, name in zip(
            values,
            ("latency", "energy", "deadline_miss", "load_imbalance", "quality"),
            strict=True,
        ):
            _validate_non_negative(value, name)
        if not isclose(sum(values), 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("objective weights must sum to 1.0")


@dataclass(frozen=True, slots=True)
class ObjectiveNormalization:
    """Fixed experiment-wide scales for dimensional latency and energy."""

    latency_ref_sec: float
    energy_ref_joules: float

    def __post_init__(self) -> None:
        _validate_positive(self.latency_ref_sec, "latency_ref_sec")
        _validate_positive(self.energy_ref_joules, "energy_ref_joules")


def estimate_objectives(
    call: SchedulableCall,
    candidate: SchedulingCandidate,
    candidates: list[SchedulingCandidate],
) -> ObjectiveVector:
    """Estimate all objectives for one feasible candidate assignment."""

    _validate_candidate_set(call, candidate, candidates)
    latency_sec = estimate_latency_sec(call, candidate)
    energy_joules = estimate_energy_joules(call, candidate)
    quality = profiled_quality(call, candidate.profile)
    deadline_miss = int(call.deadline_sec is not None and latency_sec > call.deadline_sec)
    load_imbalance = _projected_load_imbalance(candidate, candidates)
    return ObjectiveVector(
        latency_sec=latency_sec,
        energy_joules=energy_joules,
        quality=quality,
        deadline_miss=deadline_miss,
        load_imbalance=load_imbalance,
    )


def estimate_objectives_dict(
    call: SchedulableCall,
    candidate: SchedulingCandidate,
    candidates: list[SchedulingCandidate],
    *,
    allow_missing_optional_profiles: bool = False,
) -> dict[str, float | int | None]:
    """Return a trace-friendly estimate, optionally preserving unknown fields."""

    _validate_candidate_set(call, candidate, candidates)
    latency_sec = estimate_latency_sec(call, candidate)
    try:
        energy_joules: float | None = estimate_energy_joules(call, candidate)
    except MissingObjectiveProfileError:
        if not allow_missing_optional_profiles:
            raise
        energy_joules = None
    try:
        quality: float | None = profiled_quality(call, candidate.profile)
    except MissingQualityProfileError:
        if not allow_missing_optional_profiles:
            raise
        quality = None
    return {
        "latency_sec": latency_sec,
        "energy_joules": energy_joules,
        "quality": quality,
        "deadline_miss": int(call.deadline_sec is not None and latency_sec > call.deadline_sec),
        "load_imbalance": _projected_load_imbalance(candidate, candidates),
    }


def normalized_cost(
    objectives: ObjectiveVector,
    weights: ObjectiveWeights,
    normalization: ObjectiveNormalization,
) -> float:
    """Scalarize a raw objective vector using fixed experiment-wide scales."""

    return (
        weights.latency * objectives.latency_sec / normalization.latency_ref_sec
        + weights.energy * objectives.energy_joules / normalization.energy_ref_joules
        + weights.deadline_miss * objectives.deadline_miss
        + weights.load_imbalance * objectives.load_imbalance
        + weights.quality * (1 - objectives.quality)
    )


def estimate_energy_joules(
    call: SchedulableCall,
    candidate: SchedulingCandidate,
) -> float:
    """Estimate energy for policies that do not require the full objective vector."""

    profile = candidate.profile
    if isinstance(call, LLMCall) and isinstance(profile, LLMInstanceProfile):
        if "joules_per_token" not in profile.energy_profile:
            raise MissingObjectiveProfileError(
                f"target {candidate.target_id!r} has no joules_per_token profile"
            )
        total_tokens = call.input_tokens + call.estimated_output_tokens
        return total_tokens * profile.energy_profile["joules_per_token"]
    if isinstance(call, ToolCall) and isinstance(profile, ToolReplicaProfile):
        if "joules_per_call" not in profile.energy_profile:
            raise MissingObjectiveProfileError(
                f"target {candidate.target_id!r} has no joules_per_call profile"
            )
        return profile.energy_profile["joules_per_call"]
    raise TypeError("candidate profile does not match call type")


def estimate_latency_sec(
    call: SchedulableCall,
    candidate: SchedulingCandidate,
) -> float:
    """Estimate finish time and reject missing latency/throughput profiles."""

    _require_latency_profile(call, candidate)
    latency_sec = candidate.estimate_finish_time_sec(call)
    if not isfinite(latency_sec):
        raise MissingObjectiveProfileError(
            f"target {candidate.target_id!r} has no usable latency/throughput profile"
        )
    return latency_sec


def _require_latency_profile(
    call: SchedulableCall,
    candidate: SchedulingCandidate,
) -> None:
    profile = candidate.profile
    state = candidate.state
    if isinstance(call, LLMCall) and isinstance(profile, LLMInstanceProfile):
        measured_throughput = (
            state.tokens_per_sec if isinstance(state, LLMInstanceState) else 0.0
        )
        if measured_throughput <= 0 and profile.token_profile.get("tokens_per_sec", 0.0) <= 0:
            raise MissingObjectiveProfileError(
                f"target {candidate.target_id!r} has no positive tokens_per_sec profile"
            )
        return
    if isinstance(call, ToolCall) and isinstance(profile, ToolReplicaProfile):
        measured_execution = (
            state.avg_execution_time_sec if isinstance(state, ToolReplicaState) else 0.0
        )
        if measured_execution <= 0 and "execution_time_sec" not in profile.latency_profile:
            raise MissingObjectiveProfileError(
                f"target {candidate.target_id!r} has no execution_time_sec profile"
            )
        return
    raise TypeError("candidate profile does not match call type")


def _projected_load_imbalance(
    selected: SchedulingCandidate,
    candidates: list[SchedulingCandidate],
) -> float:
    projected_loads = [
        _normalized_load(candidate, add_queued_call=candidate.target_id == selected.target_id)
        for candidate in candidates
    ]
    mean_load = fmean(projected_loads)
    if len(projected_loads) < 2 or mean_load == 0:
        return 0.0
    return pstdev(projected_loads) / mean_load


def _normalized_load(candidate: SchedulingCandidate, *, add_queued_call: bool) -> float:
    profile = candidate.profile
    state = candidate.state
    if isinstance(profile, LLMInstanceProfile) and isinstance(state, LLMInstanceState):
        active_work = state.queue_len + state.running_requests
    elif isinstance(profile, ToolReplicaProfile) and isinstance(state, ToolReplicaState):
        active_work = state.queue_len + state.running_tasks
    else:
        raise TypeError("candidate profile and state types do not match")
    return (active_work + int(add_queued_call)) / profile.max_concurrency


def _validate_candidate_set(
    call: SchedulableCall,
    selected: SchedulingCandidate,
    candidates: list[SchedulingCandidate],
) -> None:
    if not candidates:
        raise ValueError("candidates must not be empty")
    matching = [candidate for candidate in candidates if candidate.target_id == selected.target_id]
    if len(matching) != 1 or matching[0] != selected:
        raise ValueError("selected candidate must appear exactly once in candidates")
    expected_kind = "llm" if isinstance(call, LLMCall) else "tool"
    if any(candidate.call_kind != expected_kind for candidate in candidates):
        raise TypeError("all candidates must match the call type")


def _validate_non_negative(value: object, field_name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{field_name} must be finite and non-negative")


def _validate_positive(value: object, field_name: str) -> None:
    _validate_non_negative(value, field_name)
    if value == 0:
        raise ValueError(f"{field_name} must be finite and positive")


def _validate_fraction(value: object, field_name: str) -> None:
    _validate_non_negative(value, field_name)
    if value > 1:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0")
