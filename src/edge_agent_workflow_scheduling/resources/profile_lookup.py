"""Shared lookup for scalar and bucketed execution profiles."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal

from edge_agent_workflow_scheduling.common import LLMCall, ToolCall
from edge_agent_workflow_scheduling.resources.models import (
    LLMInstanceProfile,
    ToolReplicaProfile,
)

FallbackPolicy = Literal["error", "aggregate"]


class ProfileLookupError(ValueError):
    """Base error for unusable calibrated profile data."""


class ProfileScopeError(ProfileLookupError):
    """Raised when a call falls outside a calibrated profile's declared scope."""


class ProfileMetricUnavailableError(ProfileLookupError):
    """Raised when a required metric is absent from a profile."""


@dataclass(frozen=True, slots=True)
class ResolvedProfileMetric:
    """One resolved metric plus the provenance needed for trace metadata."""

    value: float
    source: Literal["bucket", "aggregate", "legacy"]
    bucket_id: str | None = None


def resolve_llm_tokens_per_sec(
    profile: LLMInstanceProfile,
    call: LLMCall,
) -> ResolvedProfileMetric:
    return _resolve_metric(
        profile=profile,
        call=call,
        bucket_key="token_rate_buckets",
        scalar_mapping=profile.token_profile,
        scalar_key="tokens_per_sec",
    )


def resolve_tool_execution_time_sec(
    profile: ToolReplicaProfile,
    call: ToolCall,
) -> ResolvedProfileMetric:
    return _resolve_metric(
        profile=profile,
        call=call,
        bucket_key="latency_buckets",
        scalar_mapping=profile.latency_profile,
        scalar_key="execution_time_sec",
    )


def resolve_llm_joules_per_token(
    profile: LLMInstanceProfile,
    call: LLMCall,
) -> ResolvedProfileMetric:
    return _resolve_metric(
        profile=profile,
        call=call,
        bucket_key="energy_buckets",
        scalar_mapping=profile.energy_profile,
        scalar_key="joules_per_token",
    )


def resolve_tool_joules_per_call(
    profile: ToolReplicaProfile,
    call: ToolCall,
) -> ResolvedProfileMetric:
    return _resolve_metric(
        profile=profile,
        call=call,
        bucket_key="energy_buckets",
        scalar_mapping=profile.energy_profile,
        scalar_key="joules_per_call",
    )


def _resolve_metric(
    *,
    profile: LLMInstanceProfile | ToolReplicaProfile,
    call: LLMCall | ToolCall,
    bucket_key: str,
    scalar_mapping: dict[str, float],
    scalar_key: str,
) -> ResolvedProfileMetric:
    lookup = profile.metadata.get("profile_lookup")
    if lookup is None:
        return _legacy_metric(profile, scalar_mapping, scalar_key)
    if not isinstance(lookup, dict):
        raise ProfileLookupError("metadata.profile_lookup must be an object")

    fallback_policy = lookup.get("fallback_policy", "error")
    if fallback_policy not in {"error", "aggregate"}:
        raise ProfileLookupError(
            "metadata.profile_lookup.fallback_policy must be 'error' or 'aggregate'"
        )

    scope_error = _scope_error(profile, call)
    if scope_error is not None:
        if fallback_policy == "aggregate" and scalar_key in scalar_mapping:
            return ResolvedProfileMetric(
                value=_positive_metric(scalar_mapping[scalar_key], scalar_key),
                source="aggregate",
            )
        raise ProfileScopeError(scope_error)

    buckets = lookup.get(bucket_key, [])
    if not isinstance(buckets, list):
        raise ProfileLookupError(f"metadata.profile_lookup.{bucket_key} must be a list")
    matches = [bucket for bucket in buckets if _bucket_matches(bucket, profile, call)]
    if len(matches) > 1:
        target = _target_id(profile)
        raise ProfileLookupError(f"target {target!r} has multiple matching {bucket_key} entries")
    if matches:
        bucket = matches[0]
        bucket_scope_error = _bucket_scope_error(bucket, call, profile)
        if bucket_scope_error is not None:
            if fallback_policy == "aggregate" and scalar_key in scalar_mapping:
                return ResolvedProfileMetric(
                    value=_positive_metric(scalar_mapping[scalar_key], scalar_key),
                    source="aggregate",
                )
            raise ProfileScopeError(bucket_scope_error)
        value = _positive_metric(bucket.get("value"), f"{bucket_key} bucket value")
        bucket_id = bucket.get("bucket_id")
        if not isinstance(bucket_id, str) or not bucket_id.strip():
            raise ProfileLookupError(f"each {bucket_key} entry requires a bucket_id")
        return ResolvedProfileMetric(value=value, source="bucket", bucket_id=bucket_id)

    if not buckets and scalar_key in scalar_mapping:
        return ResolvedProfileMetric(
            value=_positive_metric(scalar_mapping[scalar_key], scalar_key),
            source="aggregate",
        )
    if not buckets:
        raise ProfileMetricUnavailableError(
            f"target {_target_id(profile)!r} has no {scalar_key} profile"
        )
    if fallback_policy == "aggregate" and scalar_key in scalar_mapping:
        return ResolvedProfileMetric(
            value=_positive_metric(scalar_mapping[scalar_key], scalar_key),
            source="aggregate",
        )

    target = _target_id(profile)
    selectors = _call_selectors(profile, call)
    raise ProfileScopeError(
        f"target {target!r} has no {bucket_key} entry for selectors {selectors!r}; "
        f"fallback_policy={fallback_policy!r}"
    )


def _legacy_metric(
    profile: LLMInstanceProfile | ToolReplicaProfile,
    mapping: dict[str, float],
    key: str,
) -> ResolvedProfileMetric:
    if key not in mapping:
        raise ProfileMetricUnavailableError(f"target {_target_id(profile)!r} has no {key} profile")
    return ResolvedProfileMetric(
        value=_positive_metric(mapping[key], key),
        source="legacy",
    )


def _bucket_matches(
    bucket: Any,
    profile: LLMInstanceProfile | ToolReplicaProfile,
    call: LLMCall | ToolCall,
) -> bool:
    if not isinstance(bucket, dict):
        raise ProfileLookupError("profile lookup buckets must be objects")
    selectors = bucket.get("selectors")
    if not isinstance(selectors, dict):
        raise ProfileLookupError("profile lookup bucket selectors must be an object")
    actual = _call_selectors(profile, call)
    unknown = set(selectors) - set(actual)
    if unknown:
        raise ProfileLookupError(
            f"profile lookup bucket has unsupported selectors: {sorted(unknown)!r}"
        )
    return all(actual[key] == value for key, value in selectors.items())


def _call_selectors(
    profile: LLMInstanceProfile | ToolReplicaProfile,
    call: LLMCall | ToolCall,
) -> dict[str, Any]:
    task_type = call.metadata.get("task_type", "default")
    input_size = call.metadata.get("input_size")
    if not isinstance(task_type, str) or not task_type.strip():
        raise ProfileLookupError("call.metadata.task_type must be a non-empty string")
    if input_size is not None and (not isinstance(input_size, str) or not input_size.strip()):
        raise ProfileLookupError("call.metadata.input_size must be a non-empty string")
    return {
        "task_type": task_type,
        "input_size": input_size,
        "concurrency": profile.max_concurrency,
    }


def _scope_error(
    profile: LLMInstanceProfile | ToolReplicaProfile,
    call: LLMCall | ToolCall,
) -> str | None:
    if not isinstance(call, LLMCall):
        return None
    scope = profile.metadata.get("scope")
    if not isinstance(scope, dict):
        return None
    for field, value in (
        ("input_token_range", call.input_tokens),
        ("output_token_range", call.estimated_output_tokens),
    ):
        limits = scope.get(field)
        if limits is None:
            continue
        if (
            not isinstance(limits, list)
            or len(limits) != 2
            or any(isinstance(item, bool) or not isinstance(item, int | float) for item in limits)
            or limits[0] > limits[1]
        ):
            raise ProfileLookupError(f"metadata.scope.{field} must be an ordered pair")
        if not limits[0] <= value <= limits[1]:
            return (
                f"target {_target_id(profile)!r} call {field.removesuffix('_range')}={value} "
                f"is outside calibrated range {limits!r}"
            )
    return None


def _bucket_scope_error(
    bucket: dict[str, Any],
    call: LLMCall | ToolCall,
    profile: LLMInstanceProfile | ToolReplicaProfile,
) -> str | None:
    if not isinstance(call, LLMCall):
        return None
    scope = bucket.get("scope")
    if scope is None:
        return None
    if not isinstance(scope, dict):
        raise ProfileLookupError("profile lookup bucket scope must be an object")
    for field, value in (
        ("input_token_range", call.input_tokens),
        ("output_token_range", call.estimated_output_tokens),
    ):
        limits = scope.get(field)
        if limits is None:
            continue
        if (
            not isinstance(limits, list)
            or len(limits) != 2
            or any(isinstance(item, bool) or not isinstance(item, int | float) for item in limits)
            or limits[0] > limits[1]
        ):
            raise ProfileLookupError(f"profile lookup bucket scope {field} is invalid")
        if not limits[0] <= value <= limits[1]:
            return (
                f"target {_target_id(profile)!r} call {field.removesuffix('_range')}={value} "
                f"is outside bucket range {limits!r}"
            )
    return None


def _target_id(profile: LLMInstanceProfile | ToolReplicaProfile) -> str:
    return profile.llm_id if isinstance(profile, LLMInstanceProfile) else profile.replica_id


def _positive_metric(value: Any, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not isfinite(value)
        or value <= 0
    ):
        raise ProfileMetricUnavailableError(f"{field_name} must be finite and positive")
    return float(value)
