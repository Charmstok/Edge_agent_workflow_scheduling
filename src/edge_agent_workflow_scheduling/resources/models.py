"""Static resource profiles and dynamic runtime states."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from math import isfinite
from typing import Any

from edge_agent_workflow_scheduling.common.schemas import SerializableSchema


@dataclass(slots=True)
class LLMInstanceProfile(SerializableSchema):
    """Static description of one schedulable LLM deployment."""

    llm_id: str
    provider: str
    model: str
    node_id: str
    platform: str
    executor_type: str
    base_url: str | None = None
    model_size_b: float | None = None
    capabilities: list[str] = field(default_factory=list)
    context_window_tokens: int = 1
    quality_profile: dict[str, float] = field(default_factory=dict)
    token_profile: dict[str, float] = field(default_factory=dict)
    energy_profile: dict[str, float] = field(default_factory=dict)
    max_concurrency: int = 1
    deployment_config: dict[str, Any] = field(default_factory=dict)
    secret_env_vars: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for value, name in (
            (self.llm_id, "llm_id"),
            (self.provider, "provider"),
            (self.model, "model"),
            (self.node_id, "node_id"),
            (self.platform, "platform"),
            (self.executor_type, "executor_type"),
        ):
            _validate_non_empty(value, name)
        if self.base_url is not None:
            _validate_non_empty(self.base_url, "base_url")
        if self.model_size_b is not None:
            _validate_positive_number(self.model_size_b, "model_size_b")
        _validate_string_list(self.capabilities, "capabilities")
        _validate_positive_integer(self.context_window_tokens, "context_window_tokens")
        _validate_fraction_mapping(self.quality_profile, "quality_profile")
        _validate_non_negative_mapping(self.token_profile, "token_profile")
        _validate_non_negative_mapping(self.energy_profile, "energy_profile")
        _validate_positive_integer(self.max_concurrency, "max_concurrency")
        _validate_deployment_config(self.deployment_config)
        _validate_string_list(self.secret_env_vars, "secret_env_vars")
        _validate_json_object(self.metadata, "metadata")


@dataclass(slots=True)
class ToolReplicaProfile(SerializableSchema):
    """Static description of one Tool deployment replica."""

    replica_id: str
    tool_name: str
    node_id: str
    platform: str
    implementation_version: str
    executor_type: str
    capabilities: list[str] = field(default_factory=list)
    latency_profile: dict[str, float] = field(default_factory=dict)
    energy_profile: dict[str, float] = field(default_factory=dict)
    quality_profile: dict[str, float] = field(default_factory=dict)
    max_concurrency: int = 1
    deployment_config: dict[str, Any] = field(default_factory=dict)
    secret_env_vars: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for value, name in (
            (self.replica_id, "replica_id"),
            (self.tool_name, "tool_name"),
            (self.node_id, "node_id"),
            (self.platform, "platform"),
            (self.implementation_version, "implementation_version"),
            (self.executor_type, "executor_type"),
        ):
            _validate_non_empty(value, name)
        _validate_string_list(self.capabilities, "capabilities")
        _validate_non_negative_mapping(self.latency_profile, "latency_profile")
        _validate_non_negative_mapping(self.energy_profile, "energy_profile")
        _validate_fraction_mapping(self.quality_profile, "quality_profile")
        _validate_positive_integer(self.max_concurrency, "max_concurrency")
        _validate_deployment_config(self.deployment_config)
        _validate_string_list(self.secret_env_vars, "secret_env_vars")
        _validate_json_object(self.metadata, "metadata")


@dataclass(slots=True)
class LLMInstanceState(SerializableSchema):
    """Dynamic measurements for one LLM instance."""

    llm_id: str
    queue_len: int = 0
    running_requests: int = 0
    compute_util: float = 0.0
    memory_util: float = 0.0
    tokens_per_sec: float = 0.0
    avg_latency_sec: float = 0.0
    is_online: bool = True
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def __post_init__(self) -> None:
        _validate_non_empty(self.llm_id, "llm_id")
        _validate_non_negative_integer(self.queue_len, "queue_len")
        _validate_non_negative_integer(self.running_requests, "running_requests")
        _validate_fraction(self.compute_util, "compute_util")
        _validate_fraction(self.memory_util, "memory_util")
        _validate_non_negative_number(self.tokens_per_sec, "tokens_per_sec")
        _validate_non_negative_number(self.avg_latency_sec, "avg_latency_sec")
        if not isinstance(self.is_online, bool):
            raise ValueError("is_online must be a boolean")


@dataclass(slots=True)
class ToolReplicaState(SerializableSchema):
    """Dynamic measurements for one Tool replica."""

    replica_id: str
    queue_len: int = 0
    running_tasks: int = 0
    cpu_util: float = 0.0
    memory_util: float = 0.0
    network_latency_ms: float = 0.0
    avg_execution_time_sec: float = 0.0
    recent_failure_rate: float = 0.0
    is_online: bool = True
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def __post_init__(self) -> None:
        _validate_non_empty(self.replica_id, "replica_id")
        _validate_non_negative_integer(self.queue_len, "queue_len")
        _validate_non_negative_integer(self.running_tasks, "running_tasks")
        _validate_fraction(self.cpu_util, "cpu_util")
        _validate_fraction(self.memory_util, "memory_util")
        _validate_non_negative_number(self.network_latency_ms, "network_latency_ms")
        _validate_non_negative_number(
            self.avg_execution_time_sec,
            "avg_execution_time_sec",
        )
        _validate_fraction(self.recent_failure_rate, "recent_failure_rate")
        if not isinstance(self.is_online, bool):
            raise ValueError("is_online must be a boolean")


@dataclass(slots=True)
class ToolConsistencySample(SerializableSchema):
    """Shared input and expected measurements for comparing Tool replicas."""

    sample_id: str
    tool_name: str
    arguments: dict[str, Any]
    expected: dict[str, Any] = field(default_factory=dict)
    numeric_tolerances: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_empty(self.sample_id, "sample_id")
        _validate_non_empty(self.tool_name, "tool_name")
        _validate_json_object(self.arguments, "arguments")
        _validate_json_object(self.expected, "expected")
        _validate_non_negative_mapping(self.numeric_tolerances, "numeric_tolerances")


def _validate_non_empty(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _validate_positive_integer(value: Any, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")


def _validate_non_negative_integer(value: Any, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _validate_positive_number(value: Any, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not isfinite(value):
        raise ValueError(f"{field_name} must be finite and positive")
    if value <= 0:
        raise ValueError(f"{field_name} must be finite and positive")


def _validate_non_negative_number(value: Any, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not isfinite(value):
        raise ValueError(f"{field_name} must be finite and non-negative")
    if value < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")


def _validate_fraction(value: Any, field_name: str) -> None:
    _validate_non_negative_number(value, field_name)
    if value > 1:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0")


def _validate_string_list(value: Any, field_name: str) -> None:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    for item in value:
        _validate_non_empty(item, f"{field_name} item")
    if len(value) != len(set(value)):
        raise ValueError(f"{field_name} must not contain duplicates")


def _validate_non_negative_mapping(value: Any, field_name: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    for key, item in value.items():
        _validate_non_empty(key, f"{field_name} key")
        _validate_non_negative_number(item, f"{field_name}.{key}")


def _validate_fraction_mapping(value: Any, field_name: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    for key, item in value.items():
        _validate_non_empty(key, f"{field_name} key")
        _validate_fraction(item, f"{field_name}.{key}")


def _validate_json_object(value: Any, field_name: str) -> None:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field_name} must be a JSON object")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be JSON serializable") from exc


def _validate_deployment_config(value: Any) -> None:
    _validate_json_object(value, "deployment_config")
    secret_keys = {"api_key", "authorization", "password", "secret", "token"}
    if _contains_secret_key(value, secret_keys):
        raise ValueError(
            "deployment_config must reference secrets through secret_env_vars, not store values"
        )


def _contains_secret_key(value: Any, secret_keys: set[str]) -> bool:
    if isinstance(value, dict):
        return any(
            key.casefold() in secret_keys or _contains_secret_key(item, secret_keys)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_secret_key(item, secret_keys) for item in value)
    return False
