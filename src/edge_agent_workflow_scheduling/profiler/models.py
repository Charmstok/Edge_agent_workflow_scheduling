"""Serializable experiment trace and reproducibility manifest models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from math import isfinite
from typing import Any, Literal, Self


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class ExperimentManifest:
    """Inputs needed to identify and reproduce one live or replay experiment."""

    experiment_id: str
    dataset_id: str
    sample_ids: list[str]
    system_prompt: str
    system_prompt_version: str
    user_template: str
    user_template_version: str
    tool_schemas: list[dict[str, Any]]
    tool_order: list[str]
    tool_implementation_versions: dict[str, list[str]]
    model_endpoints: list[dict[str, Any]]
    sampling_parameters: dict[str, Any]
    provider_seed: int | None
    agent_limits: dict[str, Any]
    scheduler_name: str
    scheduler_parameters: dict[str, Any]
    scheduler_seed: int | None
    llm_profile_version: str
    tool_profile_version: str
    resource_profiles: dict[str, Any]
    profile_seed: int | None
    code_version: str
    mode: Literal["live", "replay"]
    run_started_at: str = field(default_factory=_utc_now_iso)
    manifest_version: int = 1

    def __post_init__(self) -> None:
        for value, name in (
            (self.experiment_id, "experiment_id"),
            (self.dataset_id, "dataset_id"),
            (self.system_prompt, "system_prompt"),
            (self.system_prompt_version, "system_prompt_version"),
            (self.user_template, "user_template"),
            (self.user_template_version, "user_template_version"),
            (self.scheduler_name, "scheduler_name"),
            (self.llm_profile_version, "llm_profile_version"),
            (self.tool_profile_version, "tool_profile_version"),
            (self.code_version, "code_version"),
            (self.run_started_at, "run_started_at"),
        ):
            _non_empty(value, name)
        _string_list(self.sample_ids, "sample_ids", allow_empty=False)
        _string_list(self.tool_order, "tool_order")
        _json_value(self.tool_schemas, "tool_schemas")
        _json_value(self.tool_implementation_versions, "tool_implementation_versions")
        _json_value(self.model_endpoints, "model_endpoints")
        _json_object(self.sampling_parameters, "sampling_parameters")
        _json_object(self.agent_limits, "agent_limits")
        _json_object(self.scheduler_parameters, "scheduler_parameters")
        _json_object(self.resource_profiles, "resource_profiles")
        _optional_integer(self.provider_seed, "provider_seed")
        _optional_integer(self.scheduler_seed, "scheduler_seed")
        _optional_integer(self.profile_seed, "profile_seed")
        if self.mode not in {"live", "replay"}:
            raise ValueError("mode must be 'live' or 'replay'")
        if self.manifest_version != 1:
            raise ValueError("manifest_version must be 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        _json_object(data, "ExperimentManifest data")
        return cls(**data)


@dataclass(slots=True)
class CallTrace:
    """A reconstructable schedulable call plus its observed scheduling metrics."""

    sequence_id: int
    run_id: str
    agent_id: str
    turn_index: int
    call_id: str
    call_kind: Literal["llm", "tool"]
    call_payload: dict[str, Any]
    call_digest: str
    parameter_summary: dict[str, Any]
    selected_target: str
    policy_name: str
    status: str
    success: bool
    timeout: bool
    queue_wait_time_sec: float
    input_transfer_time_sec: float
    execution_time_sec: float
    output_transfer_time_sec: float
    total_latency_sec: float
    energy_joules: float
    model_name: str | None = None
    tool_name: str | None = None
    function_call_id: str | None = None
    raw_response_items: list[dict[str, Any]] = field(default_factory=list)
    function_call_output: dict[str, Any] | None = None
    estimated_objectives: dict[str, Any] | None = None
    result_metadata: dict[str, Any] = field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None
    created_at: str = field(default_factory=_utc_now_iso)
    decided_at: str = field(default_factory=_utc_now_iso)
    finished_at: str = field(default_factory=_utc_now_iso)

    def __post_init__(self) -> None:
        _non_negative_integer(self.sequence_id, "sequence_id")
        _non_negative_integer(self.turn_index, "turn_index")
        for value, name in (
            (self.run_id, "run_id"),
            (self.agent_id, "agent_id"),
            (self.call_id, "call_id"),
            (self.call_digest, "call_digest"),
            (self.selected_target, "selected_target"),
            (self.policy_name, "policy_name"),
            (self.status, "status"),
            (self.created_at, "created_at"),
            (self.decided_at, "decided_at"),
            (self.finished_at, "finished_at"),
        ):
            _non_empty(value, name)
        if self.call_kind not in {"llm", "tool"}:
            raise ValueError("call_kind must be 'llm' or 'tool'")
        _json_object(self.call_payload, "call_payload")
        _json_object(self.parameter_summary, "parameter_summary")
        _json_value(self.raw_response_items, "raw_response_items")
        if self.function_call_output is not None:
            _json_object(self.function_call_output, "function_call_output")
        if self.estimated_objectives is not None:
            _json_object(self.estimated_objectives, "estimated_objectives")
        _json_object(self.result_metadata, "result_metadata")
        for value, name in (
            (self.queue_wait_time_sec, "queue_wait_time_sec"),
            (self.input_transfer_time_sec, "input_transfer_time_sec"),
            (self.execution_time_sec, "execution_time_sec"),
            (self.output_transfer_time_sec, "output_transfer_time_sec"),
            (self.total_latency_sec, "total_latency_sec"),
            (self.energy_joules, "energy_joules"),
        ):
            _non_negative_number(value, name)
        if self.call_kind == "tool" and (not self.tool_name or not self.function_call_id):
            raise ValueError("tool traces require tool_name and function_call_id")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        _json_object(data, "CallTrace data")
        return cls(**data)


@dataclass(slots=True)
class AgentRunTrace:
    """End-to-end summary and preserved model conversation artifacts for one run."""

    run_id: str
    agent_id: str
    task_id: str
    status: str
    state_transitions: list[str]
    final_output: str | None
    total_rounds: int
    llm_call_count: int
    tool_call_count: int
    end_to_end_latency_sec: float
    started_at: str
    finished_at: str
    raw_response_items: list[dict[str, Any]] = field(default_factory=list)
    function_call_outputs: list[dict[str, Any]] = field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.run_id, "run_id"),
            (self.agent_id, "agent_id"),
            (self.task_id, "task_id"),
            (self.status, "status"),
            (self.started_at, "started_at"),
            (self.finished_at, "finished_at"),
        ):
            _non_empty(value, name)
        _string_list(self.state_transitions, "state_transitions", allow_empty=False)
        _non_negative_integer(self.total_rounds, "total_rounds")
        _non_negative_integer(self.llm_call_count, "llm_call_count")
        _non_negative_integer(self.tool_call_count, "tool_call_count")
        _non_negative_number(self.end_to_end_latency_sec, "end_to_end_latency_sec")
        _json_value(self.raw_response_items, "raw_response_items")
        _json_value(self.function_call_outputs, "function_call_outputs")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        _json_object(data, "AgentRunTrace data")
        return cls(**data)


@dataclass(slots=True)
class TraceBundle:
    """One manifest and the complete trace for one AgentRun."""

    manifest: ExperimentManifest
    run: AgentRunTrace
    calls: list[CallTrace]
    trace_version: int = 1

    def __post_init__(self) -> None:
        if self.trace_version != 1:
            raise ValueError("trace_version must be 1")
        if self.calls and self.run.run_id != self.calls[0].run_id:
            raise ValueError("run and call trace run_id values must match")
        if any(call.run_id != self.run.run_id for call in self.calls):
            raise ValueError("all calls must belong to the traced run")
        if [call.sequence_id for call in self.calls] != list(range(len(self.calls))):
            raise ValueError("call sequence_id values must be contiguous and ordered")

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest": self.manifest.to_dict(),
            "run": self.run.to_dict(),
            "calls": [call.to_dict() for call in self.calls],
            "trace_version": self.trace_version,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        _json_object(data, "TraceBundle data")
        return cls(
            manifest=ExperimentManifest.from_dict(data["manifest"]),
            run=AgentRunTrace.from_dict(data["run"]),
            calls=[CallTrace.from_dict(item) for item in data["calls"]],
            trace_version=data.get("trace_version", 1),
        )

    @classmethod
    def from_json(cls, data: str) -> Self:
        parsed = json.loads(data)
        if not isinstance(parsed, dict):
            raise ValueError("TraceBundle JSON must contain an object")
        return cls.from_dict(parsed)


def _non_empty(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _string_list(value: Any, field_name: str, *, allow_empty: bool = True) -> None:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise ValueError(f"{field_name} must be a list of non-empty strings")
    for item in value:
        _non_empty(item, f"{field_name} item")


def _json_object(value: Any, field_name: str) -> None:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field_name} must be a JSON object")
    _json_value(value, field_name)


def _json_value(value: Any, field_name: str) -> None:
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be JSON serializable") from exc


def _optional_integer(value: Any, field_name: str) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
        raise ValueError(f"{field_name} must be an integer or None")


def _non_negative_integer(value: Any, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _non_negative_number(value: Any, field_name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{field_name} must be finite and non-negative")
