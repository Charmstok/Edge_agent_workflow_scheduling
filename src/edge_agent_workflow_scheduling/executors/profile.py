"""Seeded profile executors for hardware-independent experiments."""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from edge_agent_workflow_scheduling.common import LLMCall, LLMResult, ToolCall, ToolResult
from edge_agent_workflow_scheduling.executors.base import (
    llm_call_error,
    tool_call_error,
    validate_timeout,
)
from edge_agent_workflow_scheduling.resources import (
    LLMInstanceProfile,
    ProfileLookupError,
    ProfileMetricUnavailableError,
    ToolReplicaProfile,
    resolve_llm_joules_per_token,
    resolve_llm_tokens_per_sec,
    resolve_tool_execution_time_sec,
    resolve_tool_joules_per_call,
)
from edge_agent_workflow_scheduling.tools import ToolSpec


@dataclass(slots=True)
class ProfileLLMExecutor:
    """Generate normalized LLM results from a fixed or seeded performance profile."""

    profile: LLMInstanceProfile
    seed: int = 0
    jitter_ratio: float = 0.0
    failure_rate: float = 0.0
    output_text: str = "Profiled response."
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_simulation_parameters(self.jitter_ratio, self.failure_rate)
        self._rng = random.Random(self.seed)

    def execute(
        self,
        llm_call: LLMCall,
        *,
        tools: list[ToolSpec] | None = None,
        timeout_sec: float | None = None,
    ) -> LLMResult:
        del tools
        validate_timeout(timeout_sec)
        invalid = llm_call_error(self.profile, llm_call)
        if invalid is not None:
            return _llm_failure(self.profile, llm_call, *invalid)
        if self._rng.random() < self.failure_rate:
            return _llm_failure(
                self.profile,
                llm_call,
                "profile_failure",
                "seeded profile LLM failure",
            )

        try:
            resolved_rate = resolve_llm_tokens_per_sec(self.profile, llm_call)
        except ProfileLookupError as exc:
            return _llm_failure(
                self.profile,
                llm_call,
                "invalid_profile",
                str(exc),
            )
        tokens_per_sec = resolved_rate.value
        output_tokens = llm_call.estimated_output_tokens
        total_tokens = llm_call.input_tokens + output_tokens
        inference_time_sec = _sample(total_tokens / tokens_per_sec, self.jitter_ratio, self._rng)
        if timeout_sec is not None and inference_time_sec > timeout_sec:
            return _llm_failure(
                self.profile,
                llm_call,
                "timeout",
                "profile LLM execution exceeded timeout_sec",
                inference_time_sec=inference_time_sec,
            )
        output_items = [_message_item(self.output_text)] if self.output_text else []
        energy_joules, energy_metadata = _llm_energy(self.profile, llm_call)
        return LLMResult(
            llm_call_id=llm_call.llm_call_id,
            llm_id=self.profile.llm_id,
            success=True,
            output_items=output_items,
            output_text=self.output_text,
            response_id=f"profile-{llm_call.llm_call_id}",
            response_model=self.profile.model,
            output_tokens=output_tokens,
            inference_time_sec=inference_time_sec,
            energy_joules=energy_joules,
            metadata={
                "executor_type": "profile",
                "seed": self.seed,
                "profile_metric": {
                    "tokens_per_sec_source": resolved_rate.source,
                    "tokens_per_sec_bucket_id": resolved_rate.bucket_id,
                    **energy_metadata,
                },
            },
        )


@dataclass(slots=True)
class ProfileToolExecutor:
    """Generate Tool results from a fixed or seeded replica profile."""

    profile: ToolReplicaProfile
    output: Any = field(default_factory=dict)
    seed: int = 0
    jitter_ratio: float = 0.0
    failure_rate: float = 0.0
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_simulation_parameters(self.jitter_ratio, self.failure_rate)
        self._rng = random.Random(self.seed)

    def execute(
        self,
        tool_call: ToolCall,
        *,
        timeout_sec: float | None = None,
    ) -> ToolResult:
        validate_timeout(timeout_sec)
        invalid = tool_call_error(self.profile, tool_call)
        if invalid is not None:
            return _tool_failure(self.profile, tool_call, *invalid)
        try:
            resolved_latency = resolve_tool_execution_time_sec(self.profile, tool_call)
        except ProfileLookupError as exc:
            return _tool_failure(self.profile, tool_call, "invalid_profile", str(exc))
        execution_time_sec = _sample(resolved_latency.value, self.jitter_ratio, self._rng)
        if timeout_sec is not None and execution_time_sec > timeout_sec:
            return _tool_failure(
                self.profile,
                tool_call,
                "timeout",
                "profile Tool execution exceeded timeout_sec",
                execution_time_sec=execution_time_sec,
            )
        if self._rng.random() < self.failure_rate:
            return _tool_failure(
                self.profile,
                tool_call,
                "profile_failure",
                "seeded profile Tool failure",
                execution_time_sec=execution_time_sec,
            )
        energy_joules, energy_metadata = _tool_energy(self.profile, tool_call)
        return ToolResult(
            tool_call_id=tool_call.tool_call_id,
            replica_id=self.profile.replica_id,
            success=True,
            output=deepcopy(self.output),
            execution_time_sec=execution_time_sec,
            energy_joules=energy_joules,
            metadata={
                "executor_type": "profile",
                "seed": self.seed,
                "profile_metric": {
                    "execution_time_sec_source": resolved_latency.source,
                    "execution_time_sec_bucket_id": resolved_latency.bucket_id,
                    **energy_metadata,
                },
            },
        )


def _sample(value: float, jitter_ratio: float, rng: random.Random) -> float:
    if jitter_ratio == 0:
        return value
    return value * rng.uniform(1.0 - jitter_ratio, 1.0 + jitter_ratio)


def _llm_energy(
    profile: LLMInstanceProfile,
    call: LLMCall,
) -> tuple[float, dict[str, Any]]:
    try:
        resolved = resolve_llm_joules_per_token(profile, call)
    except ProfileMetricUnavailableError:
        return 0.0, {
            "energy_status": "unavailable",
            "energy_value_semantics": "legacy_schema_placeholder_not_measurement",
        }
    total_tokens = call.input_tokens + call.estimated_output_tokens
    return total_tokens * resolved.value, {
        "energy_status": "profiled",
        "energy_source": resolved.source,
        "energy_bucket_id": resolved.bucket_id,
    }


def _tool_energy(
    profile: ToolReplicaProfile,
    call: ToolCall,
) -> tuple[float, dict[str, Any]]:
    try:
        resolved = resolve_tool_joules_per_call(profile, call)
    except ProfileMetricUnavailableError:
        return 0.0, {
            "energy_status": "unavailable",
            "energy_value_semantics": "legacy_schema_placeholder_not_measurement",
        }
    return resolved.value, {
        "energy_status": "profiled",
        "energy_source": resolved.source,
        "energy_bucket_id": resolved.bucket_id,
    }


def _validate_simulation_parameters(jitter_ratio: float, failure_rate: float) -> None:
    if not 0.0 <= jitter_ratio < 1.0:
        raise ValueError("jitter_ratio must be between 0.0 and 1.0")
    if not 0.0 <= failure_rate <= 1.0:
        raise ValueError("failure_rate must be between 0.0 and 1.0")


def _llm_failure(
    profile: LLMInstanceProfile,
    llm_call: LLMCall,
    error_code: str,
    error_message: str,
    *,
    inference_time_sec: float = 0.0,
) -> LLMResult:
    return LLMResult(
        llm_call_id=llm_call.llm_call_id,
        llm_id=profile.llm_id,
        success=False,
        inference_time_sec=inference_time_sec,
        error_code=error_code,
        error_message=error_message,
        metadata={"executor_type": "profile"},
    )


def _tool_failure(
    profile: ToolReplicaProfile,
    tool_call: ToolCall,
    error_code: str,
    error_message: str,
    *,
    execution_time_sec: float = 0.0,
) -> ToolResult:
    return ToolResult(
        tool_call_id=tool_call.tool_call_id,
        replica_id=profile.replica_id,
        success=False,
        execution_time_sec=execution_time_sec,
        error_code=error_code,
        error_message=error_message,
        metadata={"executor_type": "profile"},
    )


def _message_item(text: str) -> dict[str, Any]:
    return {
        "content": [{"annotations": [], "text": text, "type": "output_text"}],
        "role": "assistant",
        "status": "completed",
        "type": "message",
    }
