"""Registration and eligibility filtering for heterogeneous resources."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TypeAlias

from edge_agent_workflow_scheduling.common import LLMCall, SchedulableCall, ToolCall
from edge_agent_workflow_scheduling.resources.constraints import (
    ActionMask,
    SchedulingConstraints,
    profiled_quality,
    resolve_scheduling_constraints,
)
from edge_agent_workflow_scheduling.resources.models import (
    LLMInstanceProfile,
    LLMInstanceState,
    ToolConsistencySample,
    ToolReplicaProfile,
    ToolReplicaState,
)


@dataclass(frozen=True, slots=True)
class LLMInstanceSnapshot:
    profile: LLMInstanceProfile
    state: LLMInstanceState


@dataclass(frozen=True, slots=True)
class ToolReplicaSnapshot:
    profile: ToolReplicaProfile
    state: ToolReplicaState


ResourceSnapshot: TypeAlias = LLMInstanceSnapshot | ToolReplicaSnapshot


@dataclass(slots=True)
class ResourceRegistry:
    """Store resource profiles separately from their current measured state."""

    _llm_profiles: dict[str, LLMInstanceProfile] = field(default_factory=dict)
    _llm_states: dict[str, LLMInstanceState] = field(default_factory=dict)
    _tool_profiles: dict[str, ToolReplicaProfile] = field(default_factory=dict)
    _tool_states: dict[str, ToolReplicaState] = field(default_factory=dict)
    _consistency_samples: dict[tuple[str, str], ToolConsistencySample] = field(default_factory=dict)

    def register_llm(
        self,
        profile: LLMInstanceProfile,
        state: LLMInstanceState | None = None,
        *,
        replace: bool = False,
    ) -> None:
        if profile.llm_id in self._llm_profiles and not replace:
            raise ValueError(f"llm_id {profile.llm_id!r} is already registered")
        resolved_state = state or LLMInstanceState(llm_id=profile.llm_id)
        if resolved_state.llm_id != profile.llm_id:
            raise ValueError("LLM profile and state IDs must match")
        if resolved_state.running_requests > profile.max_concurrency:
            raise ValueError("running_requests must not exceed profile max_concurrency")
        self._llm_profiles[profile.llm_id] = deepcopy(profile)
        self._llm_states[profile.llm_id] = deepcopy(resolved_state)

    def register_tool_replica(
        self,
        profile: ToolReplicaProfile,
        state: ToolReplicaState | None = None,
        *,
        replace: bool = False,
    ) -> None:
        if profile.replica_id in self._tool_profiles and not replace:
            raise ValueError(f"replica_id {profile.replica_id!r} is already registered")
        resolved_state = state or ToolReplicaState(replica_id=profile.replica_id)
        if resolved_state.replica_id != profile.replica_id:
            raise ValueError("Tool replica profile and state IDs must match")
        if resolved_state.running_tasks > profile.max_concurrency:
            raise ValueError("running_tasks must not exceed profile max_concurrency")
        self._tool_profiles[profile.replica_id] = deepcopy(profile)
        self._tool_states[profile.replica_id] = deepcopy(resolved_state)

    def update_llm_state(self, state: LLMInstanceState) -> None:
        profile = self._require_llm_profile(state.llm_id)
        if state.running_requests > profile.max_concurrency:
            raise ValueError("running_requests must not exceed profile max_concurrency")
        self._llm_states[state.llm_id] = deepcopy(state)

    def update_tool_state(self, state: ToolReplicaState) -> None:
        profile = self._require_tool_profile(state.replica_id)
        if state.running_tasks > profile.max_concurrency:
            raise ValueError("running_tasks must not exceed profile max_concurrency")
        self._tool_states[state.replica_id] = deepcopy(state)

    def register_consistency_sample(
        self,
        sample: ToolConsistencySample,
        *,
        replace: bool = False,
    ) -> None:
        key = (sample.tool_name, sample.sample_id)
        if key in self._consistency_samples and not replace:
            raise ValueError(f"consistency sample {key!r} is already registered")
        self._consistency_samples[key] = deepcopy(sample)

    def llm_snapshot(self, llm_id: str) -> LLMInstanceSnapshot:
        return LLMInstanceSnapshot(
            profile=deepcopy(self._require_llm_profile(llm_id)),
            state=deepcopy(self._llm_states[llm_id]),
        )

    def tool_snapshot(self, replica_id: str) -> ToolReplicaSnapshot:
        return ToolReplicaSnapshot(
            profile=deepcopy(self._require_tool_profile(replica_id)),
            state=deepcopy(self._tool_states[replica_id]),
        )

    def llm_snapshots(self) -> list[LLMInstanceSnapshot]:
        return [self.llm_snapshot(llm_id) for llm_id in sorted(self._llm_profiles)]

    def tool_snapshots(self, *, tool_name: str | None = None) -> list[ToolReplicaSnapshot]:
        replica_ids = sorted(self._tool_profiles)
        if tool_name is not None:
            replica_ids = [
                replica_id
                for replica_id in replica_ids
                if self._tool_profiles[replica_id].tool_name == tool_name
            ]
        return [self.tool_snapshot(replica_id) for replica_id in replica_ids]

    def consistency_samples(self, tool_name: str) -> list[ToolConsistencySample]:
        return [
            deepcopy(sample)
            for (registered_tool, _), sample in sorted(self._consistency_samples.items())
            if registered_tool == tool_name
        ]

    def action_mask_details(
        self,
        call: SchedulableCall,
        *,
        constraints: SchedulingConstraints | None = None,
    ) -> ActionMask:
        """Return stable target-aligned feasibility values and rejection reasons."""

        resolved_constraints = resolve_scheduling_constraints(call, constraints)
        snapshots = self.snapshots_for(call)
        target_ids: list[str] = []
        values: list[bool] = []
        rejection_reasons: list[tuple[str, ...]] = []
        for snapshot in snapshots:
            if isinstance(snapshot, LLMInstanceSnapshot):
                target_id = snapshot.profile.llm_id
                reasons = _llm_rejection_reasons(call, snapshot, resolved_constraints)
            else:
                target_id = snapshot.profile.replica_id
                reasons = _tool_rejection_reasons(call, snapshot, resolved_constraints)
            target_ids.append(target_id)
            values.append(not reasons)
            rejection_reasons.append(tuple(reasons))

        return ActionMask(
            target_ids=tuple(target_ids),
            values=tuple(values),
            rejection_reasons=tuple(rejection_reasons),
        )

    def action_mask(
        self,
        call: SchedulableCall,
        *,
        constraints: SchedulingConstraints | None = None,
    ) -> dict[str, bool]:
        """Return the backward-compatible target-to-feasibility mapping."""

        return self.action_mask_details(call, constraints=constraints).as_dict()

    def eligible_snapshots(
        self,
        call: SchedulableCall,
        *,
        constraints: SchedulingConstraints | None = None,
    ) -> list[ResourceSnapshot]:
        details = self.action_mask_details(call, constraints=constraints)
        return [
            snapshot
            for snapshot, is_feasible in zip(
                self.snapshots_for(call),
                details.values,
                strict=True,
            )
            if is_feasible
        ]

    def snapshots_for(self, call: SchedulableCall) -> list[ResourceSnapshot]:
        if isinstance(call, LLMCall):
            return list(self.llm_snapshots())
        if isinstance(call, ToolCall):
            return list(self.tool_snapshots())
        raise TypeError("call must be an LLMCall or ToolCall")

    def _require_llm_profile(self, llm_id: str) -> LLMInstanceProfile:
        try:
            return self._llm_profiles[llm_id]
        except KeyError as exc:
            raise KeyError(f"llm_id {llm_id!r} is not registered") from exc

    def _require_tool_profile(self, replica_id: str) -> ToolReplicaProfile:
        try:
            return self._tool_profiles[replica_id]
        except KeyError as exc:
            raise KeyError(f"replica_id {replica_id!r} is not registered") from exc


def _llm_rejection_reasons(
    call: LLMCall,
    snapshot: LLMInstanceSnapshot,
    constraints: SchedulingConstraints,
) -> list[str]:
    profile = snapshot.profile
    state = snapshot.state
    reasons: list[str] = []
    if not state.is_online:
        reasons.append("offline")
    if state.running_requests >= profile.max_concurrency:
        reasons.append("at_capacity")
    if call.model_name is not None and call.model_name != profile.model:
        reasons.append("model_mismatch")
    if call.context_length > profile.context_window_tokens:
        reasons.append("context_window_exceeded")
    missing_capabilities = sorted(set(call.required_capabilities) - set(profile.capabilities))
    if missing_capabilities:
        reasons.append(f"missing_capabilities:{','.join(missing_capabilities)}")
    _append_experiment_constraint_reasons(call, profile, constraints, reasons)
    return reasons


def _tool_rejection_reasons(
    call: ToolCall,
    snapshot: ToolReplicaSnapshot,
    constraints: SchedulingConstraints,
) -> list[str]:
    profile = snapshot.profile
    state = snapshot.state
    reasons: list[str] = []
    if not state.is_online:
        reasons.append("offline")
    if state.running_tasks >= profile.max_concurrency:
        reasons.append("at_capacity")
    if call.tool_name != profile.tool_name:
        reasons.append("tool_name_mismatch")
    missing_capabilities = sorted(set(call.required_capabilities) - set(profile.capabilities))
    if missing_capabilities:
        reasons.append(f"missing_capabilities:{','.join(missing_capabilities)}")
    _append_experiment_constraint_reasons(call, profile, constraints, reasons)
    return reasons


def _append_experiment_constraint_reasons(
    call: SchedulableCall,
    profile: LLMInstanceProfile | ToolReplicaProfile,
    constraints: SchedulingConstraints,
    reasons: list[str],
) -> None:
    if (
        constraints.allowed_node_ids is not None
        and profile.node_id not in constraints.allowed_node_ids
    ):
        reasons.append("node_not_allowed")
    if constraints.min_quality is None or reasons:
        return
    quality = profiled_quality(call, profile)
    if quality < constraints.min_quality:
        reasons.append(
            f"quality_below_minimum:{quality:.6f}<{constraints.min_quality:.6f}"
        )
