"""Shared data schemas used across agents, schedulers, workers, and profilers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from math import isfinite
from typing import Any, Self, TypeAlias


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _validate_non_empty(value: str, field_name: str) -> None:
    if not value.strip():
        msg = f"{field_name} must be non-empty"
        raise ValueError(msg)


def _validate_non_negative(value: int | float, field_name: str) -> None:
    if not isfinite(value) or value < 0:
        msg = f"{field_name} must be finite and non-negative"
        raise ValueError(msg)


def _validate_non_negative_integer(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        msg = f"{field_name} must be a non-negative integer"
        raise ValueError(msg)


def _validate_positive_integer(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        msg = f"{field_name} must be a positive integer"
        raise ValueError(msg)


def _validate_fraction(value: float, field_name: str) -> None:
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        msg = f"{field_name} must be between 0.0 and 1.0"
        raise ValueError(msg)


def _validate_json_object(value: object, field_name: str) -> None:
    if not isinstance(value, dict):
        msg = f"{field_name} must be a JSON object"
        raise ValueError(msg)
    if any(not isinstance(key, str) for key in value):
        msg = f"{field_name} keys must be strings"
        raise ValueError(msg)
    try:
        json.dumps(value)
    except (TypeError, ValueError) as exc:
        msg = f"{field_name} must be JSON serializable"
        raise ValueError(msg) from exc


def _validate_json_object_list(value: object, field_name: str) -> None:
    if not isinstance(value, list):
        msg = f"{field_name} must be a list"
        raise ValueError(msg)
    for index, item in enumerate(value):
        _validate_json_object(item, f"{field_name}[{index}]")


def _validate_string_list(value: object, field_name: str) -> None:
    if not isinstance(value, list):
        msg = f"{field_name} must be a list"
        raise ValueError(msg)
    for item in value:
        if not isinstance(item, str):
            msg = f"{field_name} items must be strings"
            raise ValueError(msg)
        _validate_non_empty(item, f"{field_name} item")


class AgentRunStatus(StrEnum):
    """Lifecycle states for one end-to-end agent run."""

    CREATED = "created"
    READY_FOR_LLM = "ready_for_llm"
    WAITING_FOR_LLM = "waiting_for_llm"
    WAITING_FOR_TOOLS = "waiting_for_tools"
    COMPLETED = "completed"
    FAILED = "failed"


class CallStatus(StrEnum):
    """Lifecycle states for one schedulable LLM or Tool call."""

    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


_AGENT_RUN_TRANSITIONS: dict[AgentRunStatus, frozenset[AgentRunStatus]] = {
    AgentRunStatus.CREATED: frozenset({AgentRunStatus.READY_FOR_LLM, AgentRunStatus.FAILED}),
    AgentRunStatus.READY_FOR_LLM: frozenset(
        {AgentRunStatus.WAITING_FOR_LLM, AgentRunStatus.FAILED}
    ),
    AgentRunStatus.WAITING_FOR_LLM: frozenset(
        {
            AgentRunStatus.WAITING_FOR_TOOLS,
            AgentRunStatus.COMPLETED,
            AgentRunStatus.FAILED,
        }
    ),
    AgentRunStatus.WAITING_FOR_TOOLS: frozenset(
        {AgentRunStatus.READY_FOR_LLM, AgentRunStatus.FAILED}
    ),
    AgentRunStatus.COMPLETED: frozenset(),
    AgentRunStatus.FAILED: frozenset(),
}

_CALL_TRANSITIONS: dict[CallStatus, frozenset[CallStatus]] = {
    CallStatus.CREATED: frozenset({CallStatus.QUEUED, CallStatus.FAILED}),
    CallStatus.QUEUED: frozenset({CallStatus.RUNNING, CallStatus.FAILED}),
    CallStatus.RUNNING: frozenset({CallStatus.SUCCEEDED, CallStatus.FAILED}),
    CallStatus.SUCCEEDED: frozenset(),
    CallStatus.FAILED: frozenset(),
}


@dataclass(slots=True)
class SerializableSchema:
    """Small JSON helper for dataclass-based schemas."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        if not isinstance(data, dict):
            msg = f"{cls.__name__} data must be a JSON object"
            raise ValueError(msg)
        return cls(**data)

    @classmethod
    def from_json(cls, data: str) -> Self:
        parsed = json.loads(data)
        if not isinstance(parsed, dict):
            msg = f"{cls.__name__} JSON must contain an object"
            raise ValueError(msg)
        return cls.from_dict(parsed)


@dataclass(slots=True)
class AgentRun(SerializableSchema):
    """State and correlation data for one end-to-end agent request."""

    run_id: str
    agent_id: str
    task_id: str
    status: AgentRunStatus = AgentRunStatus.CREATED
    turn_index: int = 0
    conversation_items: list[dict[str, Any]] = field(default_factory=list)
    final_output: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    started_at: str = field(default_factory=_utc_now_iso)
    finished_at: str | None = None

    def __post_init__(self) -> None:
        _validate_non_empty(self.run_id, "run_id")
        _validate_non_empty(self.agent_id, "agent_id")
        _validate_non_empty(self.task_id, "task_id")
        _validate_non_negative_integer(self.turn_index, "turn_index")
        _validate_json_object_list(self.conversation_items, "conversation_items")
        if self.error_code is not None:
            _validate_non_empty(self.error_code, "error_code")
        self.status = AgentRunStatus(self.status)
        if self.status in {AgentRunStatus.COMPLETED, AgentRunStatus.FAILED}:
            self.finished_at = self.finished_at or _utc_now_iso()
        elif self.finished_at is not None:
            msg = "finished_at is only valid for a completed or failed AgentRun"
            raise ValueError(msg)

    def transition_to(self, status: AgentRunStatus) -> None:
        """Move the run to a valid next lifecycle state."""

        next_status = AgentRunStatus(status)
        if next_status not in _AGENT_RUN_TRANSITIONS[self.status]:
            msg = f"invalid AgentRun transition: {self.status.value} -> {next_status.value}"
            raise ValueError(msg)
        self.status = next_status
        if next_status in {AgentRunStatus.COMPLETED, AgentRunStatus.FAILED}:
            self.finished_at = _utc_now_iso()


@dataclass(slots=True)
class ToolCall(SerializableSchema):
    """A provider-neutral function call generated by an agent."""

    tool_call_id: str
    run_id: str
    agent_id: str
    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    turn_index: int = 0
    required_capabilities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    deadline_sec: float | None = None
    priority: int = 0
    status: CallStatus = CallStatus.CREATED
    created_at: str = field(default_factory=_utc_now_iso)

    def __post_init__(self) -> None:
        _validate_non_empty(self.tool_call_id, "tool_call_id")
        _validate_non_empty(self.run_id, "run_id")
        _validate_non_empty(self.agent_id, "agent_id")
        _validate_non_empty(self.call_id, "call_id")
        _validate_non_empty(self.tool_name, "tool_name")
        _validate_non_negative_integer(self.turn_index, "turn_index")
        _validate_json_object(self.arguments, "arguments")
        _validate_string_list(self.required_capabilities, "required_capabilities")
        if len(set(self.required_capabilities)) != len(self.required_capabilities):
            msg = "required_capabilities must not contain duplicates"
            raise ValueError(msg)
        _validate_json_object(self.metadata, "metadata")
        if self.deadline_sec is not None:
            _validate_non_negative(self.deadline_sec, "deadline_sec")
        self.status = CallStatus(self.status)

    def transition_to(self, status: CallStatus) -> None:
        """Move the call to a valid next lifecycle state."""

        next_status = CallStatus(status)
        if next_status not in _CALL_TRANSITIONS[self.status]:
            msg = f"invalid ToolCall transition: {self.status.value} -> {next_status.value}"
            raise ValueError(msg)
        self.status = next_status


@dataclass(slots=True)
class LLMCall(SerializableSchema):
    """A provider-neutral LLM request generated during an agent run."""

    llm_call_id: str
    run_id: str
    agent_id: str
    turn_index: int = 0
    input_items: list[dict[str, Any]] = field(default_factory=list)
    required_capabilities: list[str] = field(default_factory=list)
    input_tokens: int = 0
    estimated_output_tokens: int = 0
    context_length: int = 0
    model_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    deadline_sec: float | None = None
    priority: int = 0
    status: CallStatus = CallStatus.CREATED
    created_at: str = field(default_factory=_utc_now_iso)

    def __post_init__(self) -> None:
        _validate_non_empty(self.llm_call_id, "llm_call_id")
        _validate_non_empty(self.run_id, "run_id")
        _validate_non_empty(self.agent_id, "agent_id")
        _validate_non_negative_integer(self.turn_index, "turn_index")
        _validate_json_object_list(self.input_items, "input_items")
        _validate_string_list(self.required_capabilities, "required_capabilities")
        if len(set(self.required_capabilities)) != len(self.required_capabilities):
            msg = "required_capabilities must not contain duplicates"
            raise ValueError(msg)
        _validate_non_negative_integer(self.input_tokens, "input_tokens")
        _validate_non_negative_integer(
            self.estimated_output_tokens,
            "estimated_output_tokens",
        )
        _validate_non_negative_integer(self.context_length, "context_length")
        if self.model_name is not None:
            _validate_non_empty(self.model_name, "model_name")
        _validate_json_object(self.metadata, "metadata")
        if self.deadline_sec is not None:
            _validate_non_negative(self.deadline_sec, "deadline_sec")
        self.status = CallStatus(self.status)

    def transition_to(self, status: CallStatus) -> None:
        """Move the call to a valid next lifecycle state."""

        next_status = CallStatus(status)
        if next_status not in _CALL_TRANSITIONS[self.status]:
            msg = f"invalid LLMCall transition: {self.status.value} -> {next_status.value}"
            raise ValueError(msg)
        self.status = next_status


SchedulableCall: TypeAlias = LLMCall | ToolCall


@dataclass(slots=True)
class ScheduleDecision(SerializableSchema):
    """The scheduler's target choice for one schedulable call."""

    call_id: str
    call_kind: str
    selected_target: str
    policy_name: str
    score: float | None = None
    reason: str | None = None
    candidate_target_ids: list[str] = field(default_factory=list)
    action_mask: list[bool] = field(default_factory=list)
    rejection_reasons: dict[str, list[str]] = field(default_factory=dict)
    decided_at: str = field(default_factory=_utc_now_iso)

    def __post_init__(self) -> None:
        for value, name in (
            (self.call_id, "call_id"),
            (self.call_kind, "call_kind"),
            (self.selected_target, "selected_target"),
            (self.policy_name, "policy_name"),
        ):
            _validate_non_empty(value, name)
        if self.call_kind not in {"llm", "tool"}:
            raise ValueError("call_kind must be 'llm' or 'tool'")
        if len(self.candidate_target_ids) != len(self.action_mask):
            raise ValueError("candidate_target_ids and action_mask must have equal lengths")
        _validate_string_list(self.candidate_target_ids, "candidate_target_ids")
        if self.candidate_target_ids != sorted(self.candidate_target_ids):
            raise ValueError("candidate_target_ids must use stable sorted order")
        if any(not isinstance(value, bool) for value in self.action_mask):
            raise ValueError("action_mask must contain booleans")
        _validate_json_object(self.rejection_reasons, "rejection_reasons")
        for target_id, reasons in self.rejection_reasons.items():
            _validate_non_empty(target_id, "rejection_reasons target ID")
            _validate_string_list(reasons, f"rejection_reasons.{target_id}")
        unknown_rejections = set(self.rejection_reasons) - set(self.candidate_target_ids)
        if unknown_rejections:
            raise ValueError("rejection_reasons contains an unknown target")
        if self.candidate_target_ids:
            try:
                selected_index = self.candidate_target_ids.index(self.selected_target)
            except ValueError as exc:
                raise ValueError("selected_target must appear in candidate_target_ids") from exc
            if not self.action_mask[selected_index]:
                raise ValueError("selected_target must be feasible in action_mask")


@dataclass(slots=True)
class LLMResult(SerializableSchema):
    """Inference result returned by an LLM runtime instance."""

    llm_call_id: str
    llm_id: str
    success: bool
    output_items: list[dict[str, Any]] = field(default_factory=list)
    output_text: str = ""
    response_id: str | None = None
    response_model: str | None = None
    output_uri: str | None = None
    output_tokens: int = 0
    queue_wait_time_sec: float = 0.0
    input_transfer_time_sec: float = 0.0
    inference_time_sec: float = 0.0
    output_transfer_time_sec: float = 0.0
    energy_joules: float = 0.0
    error_code: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    finished_at: str = field(default_factory=_utc_now_iso)

    def __post_init__(self) -> None:
        _validate_non_empty(self.llm_call_id, "llm_call_id")
        _validate_non_empty(self.llm_id, "llm_id")
        _validate_json_object_list(self.output_items, "output_items")
        if not isinstance(self.output_text, str):
            raise ValueError("output_text must be a string")
        for value, field_name in (
            (self.response_id, "response_id"),
            (self.response_model, "response_model"),
            (self.output_uri, "output_uri"),
        ):
            if value is not None:
                _validate_non_empty(value, field_name)
        _validate_non_negative_integer(self.output_tokens, "output_tokens")
        _validate_non_negative(self.queue_wait_time_sec, "queue_wait_time_sec")
        _validate_non_negative(self.input_transfer_time_sec, "input_transfer_time_sec")
        _validate_non_negative(self.inference_time_sec, "inference_time_sec")
        _validate_non_negative(self.output_transfer_time_sec, "output_transfer_time_sec")
        _validate_non_negative(self.energy_joules, "energy_joules")
        _validate_json_object(self.metadata, "metadata")
        if self.success and (self.error_code is not None or self.error_message is not None):
            raise ValueError("successful LLMResult cannot contain an error")
        if not self.success and (not self.error_code or not self.error_message):
            raise ValueError("failed LLMResult requires error_code and error_message")

    @property
    def execution_time_sec(self) -> float:
        """Return the provider-neutral execution duration."""

        return self.inference_time_sec


@dataclass(slots=True)
class ToolResult(SerializableSchema):
    """Execution result returned by a Tool replica."""

    tool_call_id: str
    replica_id: str
    success: bool
    output: Any = None
    queue_wait_time_sec: float = 0.0
    input_transfer_time_sec: float = 0.0
    execution_time_sec: float = 0.0
    output_transfer_time_sec: float = 0.0
    energy_joules: float = 0.0
    error_code: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    finished_at: str = field(default_factory=_utc_now_iso)

    def __post_init__(self) -> None:
        _validate_non_empty(self.tool_call_id, "tool_call_id")
        _validate_non_empty(self.replica_id, "replica_id")
        _validate_non_negative(self.queue_wait_time_sec, "queue_wait_time_sec")
        _validate_non_negative(self.input_transfer_time_sec, "input_transfer_time_sec")
        _validate_non_negative(self.execution_time_sec, "execution_time_sec")
        _validate_non_negative(self.output_transfer_time_sec, "output_transfer_time_sec")
        _validate_non_negative(self.energy_joules, "energy_joules")
        try:
            json.dumps(self.output, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("output must be JSON serializable") from exc
        _validate_json_object(self.metadata, "metadata")
        if self.success and (self.error_code is not None or self.error_message is not None):
            raise ValueError("successful ToolResult cannot contain an error")
        if not self.success and (not self.error_code or not self.error_message):
            raise ValueError("failed ToolResult requires error_code and error_message")


@dataclass(slots=True)
class TraceRecord(SerializableSchema):
    """Profiler record for a completed schedulable call."""

    run_id: str
    call_id: str
    call_kind: str
    agent_id: str
    turn_index: int
    selected_target: str
    policy_name: str
    queue_wait_time_sec: float
    execution_time_sec: float
    total_latency_sec: float
    success: bool
    timeout: bool
    reward: float
    function_call_id: str | None = None
    tool_name: str | None = None
    model_name: str | None = None
    input_transfer_time_sec: float = 0.0
    output_transfer_time_sec: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    error_message: str | None = None
    recorded_at: str = field(default_factory=_utc_now_iso)
