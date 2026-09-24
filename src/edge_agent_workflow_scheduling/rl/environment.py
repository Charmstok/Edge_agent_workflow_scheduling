"""Gymnasium-compatible environment for replay-based resource scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from edge_agent_workflow_scheduling.common import LLMCall, SchedulableCall, ToolCall
from edge_agent_workflow_scheduling.executors import (
    ExecutorFactoryRegistry,
    ExecutorPool,
    ProfileLLMExecutor,
    ProfileToolExecutor,
)
from edge_agent_workflow_scheduling.resources import (
    ResourceRegistry,
    SchedulingConstraints,
    resolve_llm_joules_per_token,
    resolve_tool_joules_per_call,
)
from edge_agent_workflow_scheduling.scheduler.objectives import estimate_objectives_dict
from edge_agent_workflow_scheduling.scheduler.types import candidate_from_snapshot, call_id_for


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    """Fixed bounds used to make observations reproducible across episodes."""

    max_calls: int = 128
    max_input_tokens: int = 131072
    max_output_tokens: int = 16384
    max_latency_sec: float = 3600.0
    max_energy_joules: float = 1_000_000.0


class SchedulingEnv(gym.Env[np.ndarray, int]):
    """Schedule a fixed replay sequence one call at a time.

    The environment never invents future ToolCalls. It consumes the supplied
    replay sequence and uses the same ResourceRegistry action mask and profile
    estimates as the baseline Scheduler. ``info["action_mask"]`` is aligned
    with ``action_space`` and contains only feasible targets.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        calls: list[SchedulableCall],
        resources: ResourceRegistry,
        *,
        constraints: SchedulingConstraints | None = None,
        config: EnvironmentConfig | None = None,
        profile_seed: int = 0,
        executor_pool: ExecutorPool | None = None,
    ) -> None:
        super().__init__()
        if not calls:
            raise ValueError("calls must not be empty")
        if len(calls) > (config or EnvironmentConfig()).max_calls:
            raise ValueError("calls exceeds EnvironmentConfig.max_calls")
        self.calls = list(calls)
        self.resources = resources
        self.constraints = constraints or SchedulingConstraints()
        self.config = config or EnvironmentConfig()
        self.profile_seed = profile_seed
        self.executor_pool = executor_pool or _profile_executor_pool(resources, profile_seed)
        self._target_ids = tuple(sorted(_all_target_ids(resources)))
        if not self._target_ids:
            raise ValueError("resources must contain at least one target")
        self._target_index = {target_id: i for i, target_id in enumerate(self._target_ids)}
        self.action_space = spaces.Discrete(len(self._target_ids))
        # Per-call features followed by one row per stable target. Each target
        # row is [eligible, queue, running, capacity, online, latency, energy, quality].
        feature_count = 8 + 8 * len(self._target_ids)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(feature_count,), dtype=np.float32)
        self._call_index = 0
        self._decisions: list[dict[str, Any]] = []
        self._initial_states = resources.snapshot_states()

    @property
    def target_ids(self) -> tuple[str, ...]:
        return self._target_ids

    @property
    def decisions(self) -> list[dict[str, Any]]:
        return list(self._decisions)

    def action_masks(self) -> np.ndarray:
        """Return a stable action mask aligned with ``action_space``."""

        if self._call_index >= len(self.calls):
            return np.zeros(len(self._target_ids), dtype=np.int8)
        mask = self.resources.action_mask_details(
            self.calls[self._call_index], constraints=self.constraints
        )
        values = dict(zip(mask.target_ids, mask.values, strict=True))
        return np.asarray([int(values.get(target_id, False)) for target_id in self._target_ids], dtype=np.int8)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        del options
        self._call_index = 0
        self._decisions = []
        self.resources.restore_states(self._initial_states)
        return self._observation(), {"action_mask": self.action_masks(), "target_ids": self._target_ids}

    def step(self, action: int):
        if self._call_index >= len(self.calls):
            raise RuntimeError("episode is terminated; call reset()")
        if not self.action_space.contains(action):
            raise ValueError(f"action {action!r} is outside action_space")
        mask = self.action_masks()
        if not mask[action]:
            raise ValueError(f"illegal action {action}: target is masked")
        call = self.calls[self._call_index]
        target_id = self._target_ids[int(action)]
        snapshot = self.resources.llm_snapshot(target_id) if isinstance(call, LLMCall) else self.resources.tool_snapshot(target_id)
        candidate = candidate_from_snapshot(snapshot)
        candidates = [candidate_from_snapshot(item) for item in self.resources.eligible_snapshots(call, constraints=self.constraints)]
        objectives = estimate_objectives_dict(call, candidate, candidates, allow_missing_optional_profiles=True)
        result = self._execute(call, candidate.profile)
        self._decisions.append({"call_id": call_id_for(call), "target_id": target_id, "objectives": objectives, "success": _result_success(result)})
        self._call_index += 1
        terminated = self._call_index >= len(self.calls)
        info = {"action_mask": self.action_masks(), "target_ids": self._target_ids, "decision": self._decisions[-1], "result": result.to_dict()}
        return self._observation() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, terminated, False, info

    def _observation(self) -> np.ndarray:
        if self._call_index >= len(self.calls):
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        call = self.calls[self._call_index]
        call_features = np.asarray([
            float(isinstance(call, ToolCall)), float(isinstance(call, LLMCall)),
            min(call.turn_index / self.config.max_calls, 1.0),
            min((call.input_tokens if isinstance(call, LLMCall) else 0) / self.config.max_input_tokens, 1.0),
            min((call.estimated_output_tokens if isinstance(call, LLMCall) else 0) / self.config.max_output_tokens, 1.0),
            min((call.context_length if isinstance(call, LLMCall) else 0) / self.config.max_input_tokens, 1.0),
            min((call.deadline_sec or self.config.max_latency_sec) / self.config.max_latency_sec, 1.0),
            min(self._call_index / self.config.max_calls, 1.0),
        ], dtype=np.float32)
        mask = self.action_masks()
        rows: list[float] = []
        for index, target_id in enumerate(self._target_ids):
            row = [float(mask[index]), 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            try:
                snapshot = self.resources.llm_snapshot(target_id) if isinstance(call, LLMCall) else self.resources.tool_snapshot(target_id)
                state, profile = snapshot.state, snapshot.profile
                running = getattr(state, "running_requests", getattr(state, "running_tasks", 0))
                row[1] = min(state.queue_len / max(profile.max_concurrency, 1), 1.0)
                row[2] = min(running / max(profile.max_concurrency, 1), 1.0)
                row[3] = min((state.queue_len + running) / max(profile.max_concurrency, 1), 1.0)
                row[4] = float(state.is_online)
                candidate = candidate_from_snapshot(snapshot)
                candidates = [candidate_from_snapshot(item) for item in self.resources.eligible_snapshots(call, constraints=self.constraints)]
                estimated = estimate_objectives_dict(call, candidate, candidates, allow_missing_optional_profiles=True)
                row[5] = min(float(estimated["latency_sec"] or 0.0) / self.config.max_latency_sec, 1.0)
                row[6] = min(float(estimated["energy_joules"] or 0.0) / self.config.max_energy_joules, 1.0)
                row[7] = float(estimated["quality"] if estimated["quality"] is not None else 0.0)
            except (KeyError, ValueError, TypeError):
                pass
            rows.extend(row)
        return np.concatenate((call_features, np.asarray(rows, dtype=np.float32))).astype(np.float32)

    def _execute(self, call: SchedulableCall, profile: Any) -> Any:
        if isinstance(call, LLMCall):
            return self.executor_pool.llm_executor(profile).execute(call)
        return self.executor_pool.tool_executor(profile).execute(call)


def _all_target_ids(resources: ResourceRegistry) -> set[str]:
    return {snapshot.profile.llm_id for snapshot in resources.llm_snapshots()} | {snapshot.profile.replica_id for snapshot in resources.tool_snapshots()}


def _profile_executor_pool(resources: ResourceRegistry, seed: int) -> ExecutorPool:
    factories = ExecutorFactoryRegistry()
    factories.register_llm("profile", lambda profile: ProfileLLMExecutor(profile, seed=seed))
    factories.register_tool("profile", lambda profile: ProfileToolExecutor(profile, seed=seed))
    factories.register_llm("openai_chat", lambda profile: ProfileLLMExecutor(profile, seed=seed))
    factories.register_tool("local_tool", lambda profile: ProfileToolExecutor(profile, seed=seed))
    return ExecutorPool(factories)


def _result_success(result: Any) -> bool:
    return bool(getattr(result, "success", False))
