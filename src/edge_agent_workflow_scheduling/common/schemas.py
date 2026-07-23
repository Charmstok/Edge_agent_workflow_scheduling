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
    error_message: str | None = None
    started_at: str = field(default_factory=_utc_now_iso)
    finished_at: str | None = None

    def __post_init__(self) -> None:
        _validate_non_empty(self.run_id, "run_id")
        _validate_non_empty(self.agent_id, "agent_id")
        _validate_non_empty(self.task_id, "task_id")
        _validate_non_negative_integer(self.turn_index, "turn_index")
        _validate_json_object_list(self.conversation_items, "conversation_items")
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
class WorkerInfo(SerializableSchema):
    """Static information about a worker."""

    worker_id: str
    supported_tools: list[str]
    max_concurrency: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LLMInstanceInfo(SerializableSchema):
    """Static information about an LLM runtime instance."""

    llm_id: str
    model_name: str
    endpoint_url: str
    device_id: str
    model_size_b: float | None = None
    accelerator: str | None = None
    max_concurrency: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WorkerState(SerializableSchema):
    """Dynamic runtime state reported by a worker."""

    worker_id: str
    supported_tools: list[str]
    queue_len: int = 0
    running_tasks: int = 0
    max_concurrency: int = 1
    cpu_util: float = 0.0
    memory_util: float = 0.0
    network_latency_ms: float = 0.0
    avg_execution_time_sec: float = 0.0
    recent_failure_rate: float = 0.0
    is_online: bool = True
    updated_at: str = field(default_factory=_utc_now_iso)

    def __post_init__(self) -> None:
        _validate_non_empty(self.worker_id, "worker_id")
        _validate_non_negative_integer(self.queue_len, "queue_len")
        _validate_non_negative_integer(self.running_tasks, "running_tasks")
        _validate_positive_integer(self.max_concurrency, "max_concurrency")
        if self.running_tasks > self.max_concurrency:
            msg = "running_tasks must not exceed max_concurrency"
            raise ValueError(msg)
        _validate_fraction(self.cpu_util, "cpu_util")
        _validate_fraction(self.memory_util, "memory_util")
        _validate_non_negative(self.network_latency_ms, "network_latency_ms")
        _validate_non_negative(self.avg_execution_time_sec, "avg_execution_time_sec")
        _validate_fraction(self.recent_failure_rate, "recent_failure_rate")


@dataclass(slots=True)
class LLMInstanceState(SerializableSchema):
    """Dynamic runtime state reported by an LLM instance."""

    llm_id: str
    model_name: str
    device_id: str
    queue_len: int = 0
    running_requests: int = 0
    max_concurrency: int = 1
    gpu_util: float = 0.0
    memory_util: float = 0.0
    tokens_per_sec: float = 0.0
    avg_latency_sec: float = 0.0
    is_online: bool = True
    updated_at: str = field(default_factory=_utc_now_iso)

    def __post_init__(self) -> None:
        _validate_non_empty(self.llm_id, "llm_id")
        _validate_non_empty(self.model_name, "model_name")
        _validate_non_empty(self.device_id, "device_id")
        _validate_non_negative_integer(self.queue_len, "queue_len")
        _validate_non_negative_integer(self.running_requests, "running_requests")
        _validate_positive_integer(self.max_concurrency, "max_concurrency")
        if self.running_requests > self.max_concurrency:
            msg = "running_requests must not exceed max_concurrency"
            raise ValueError(msg)
        _validate_fraction(self.gpu_util, "gpu_util")
        _validate_fraction(self.memory_util, "memory_util")
        _validate_non_negative(self.tokens_per_sec, "tokens_per_sec")
        _validate_non_negative(self.avg_latency_sec, "avg_latency_sec")


@dataclass(slots=True)
class ScheduleDecision(SerializableSchema):
    """The scheduler's target choice for one schedulable call."""

    call_id: str
    call_kind: str
    selected_target: str
    policy_name: str
    score: float | None = None
    reason: str | None = None
    decided_at: str = field(default_factory=_utc_now_iso)


@dataclass(slots=True)
class LLMResult(SerializableSchema):
    """Inference result returned by an LLM runtime instance."""

    llm_call_id: str
    llm_id: str
    success: bool
    output_uri: str | None = None
    output_tokens: int = 0
    queue_wait_time_sec: float = 0.0
    inference_time_sec: float = 0.0
    error_message: str | None = None
    finished_at: str = field(default_factory=_utc_now_iso)

    def __post_init__(self) -> None:
        _validate_non_empty(self.llm_call_id, "llm_call_id")
        _validate_non_empty(self.llm_id, "llm_id")
        _validate_non_negative_integer(self.output_tokens, "output_tokens")
        _validate_non_negative(self.queue_wait_time_sec, "queue_wait_time_sec")
        _validate_non_negative(self.inference_time_sec, "inference_time_sec")


@dataclass(slots=True)
class ToolResult(SerializableSchema):
    """Execution result returned by a worker."""

    tool_call_id: str
    worker_id: str
    success: bool
    output_uri: str | None = None
    queue_wait_time_sec: float = 0.0
    execution_time_sec: float = 0.0
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    finished_at: str = field(default_factory=_utc_now_iso)

    def __post_init__(self) -> None:
        _validate_non_empty(self.tool_call_id, "tool_call_id")
        _validate_non_empty(self.worker_id, "worker_id")
        _validate_non_negative(self.queue_wait_time_sec, "queue_wait_time_sec")
        _validate_non_negative(self.execution_time_sec, "execution_time_sec")


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
