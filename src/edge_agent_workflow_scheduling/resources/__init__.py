"""Heterogeneous LLM instance and Tool replica resources."""

from edge_agent_workflow_scheduling.resources.constraints import (
    ActionMask,
    MissingQualityProfileError,
    SchedulingConstraints,
    profiled_quality,
    resolve_scheduling_constraints,
    task_type_for_call,
)
from edge_agent_workflow_scheduling.resources.models import (
    LLMInstanceProfile,
    LLMInstanceState,
    ToolConsistencySample,
    ToolReplicaProfile,
    ToolReplicaState,
)
from edge_agent_workflow_scheduling.resources.profile_lookup import (
    ProfileLookupError,
    ProfileMetricUnavailableError,
    ProfileScopeError,
    ResolvedProfileMetric,
    resolve_llm_joules_per_token,
    resolve_llm_tokens_per_sec,
    resolve_tool_execution_time_sec,
    resolve_tool_joules_per_call,
)
from edge_agent_workflow_scheduling.resources.registry import (
    LLMInstanceSnapshot,
    ResourceRegistry,
    ResourceSnapshot,
    ToolReplicaSnapshot,
)

__all__ = [
    "ActionMask",
    "LLMInstanceProfile",
    "LLMInstanceSnapshot",
    "LLMInstanceState",
    "MissingQualityProfileError",
    "ProfileLookupError",
    "ProfileMetricUnavailableError",
    "ProfileScopeError",
    "ResolvedProfileMetric",
    "ResourceRegistry",
    "ResourceSnapshot",
    "SchedulingConstraints",
    "ToolConsistencySample",
    "ToolReplicaProfile",
    "ToolReplicaSnapshot",
    "ToolReplicaState",
    "profiled_quality",
    "resolve_llm_joules_per_token",
    "resolve_llm_tokens_per_sec",
    "resolve_scheduling_constraints",
    "resolve_tool_execution_time_sec",
    "resolve_tool_joules_per_call",
    "task_type_for_call",
]
