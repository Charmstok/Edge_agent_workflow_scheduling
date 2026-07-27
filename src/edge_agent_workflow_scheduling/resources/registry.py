"""Registration and eligibility filtering for heterogeneous resources."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TypeAlias

from edge_agent_workflow_scheduling.common import LLMCall, SchedulableCall, ToolCall
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

    def action_mask(self, call: SchedulableCall) -> dict[str, bool]:
        if isinstance(call, LLMCall):
            return {
                snapshot.profile.llm_id: _can_run_llm(call, snapshot)
                for snapshot in self.llm_snapshots()
            }
        if isinstance(call, ToolCall):
            return {
                snapshot.profile.replica_id: _can_run_tool(call, snapshot)
                for snapshot in self.tool_snapshots()
            }
        raise TypeError("call must be an LLMCall or ToolCall")

    def eligible_snapshots(self, call: SchedulableCall) -> list[ResourceSnapshot]:
        mask = self.action_mask(call)
        if isinstance(call, LLMCall):
            return [snapshot for snapshot in self.llm_snapshots() if mask[snapshot.profile.llm_id]]
        if isinstance(call, ToolCall):
            return [
                snapshot for snapshot in self.tool_snapshots() if mask[snapshot.profile.replica_id]
            ]
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


def _can_run_llm(call: LLMCall, snapshot: LLMInstanceSnapshot) -> bool:
    profile = snapshot.profile
    state = snapshot.state
    if not state.is_online or state.running_requests >= profile.max_concurrency:
        return False
    if call.model_name is not None and call.model_name != profile.model:
        return False
    if call.context_length > profile.context_window_tokens:
        return False
    return set(call.required_capabilities).issubset(profile.capabilities)


def _can_run_tool(call: ToolCall, snapshot: ToolReplicaSnapshot) -> bool:
    profile = snapshot.profile
    state = snapshot.state
    return (
        state.is_online
        and state.running_tasks < profile.max_concurrency
        and call.tool_name == profile.tool_name
        and set(call.required_capabilities).issubset(profile.capabilities)
    )
