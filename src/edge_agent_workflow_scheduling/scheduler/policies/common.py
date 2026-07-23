"""Shared helpers for scheduler policies."""

from __future__ import annotations

from edge_agent_workflow_scheduling.common import LLMCall, SchedulableCall, ToolCall
from edge_agent_workflow_scheduling.scheduler.types import SchedulingCandidate


def require_candidates(candidates: list[SchedulingCandidate]) -> None:
    if not candidates:
        msg = "scheduler policy requires at least one candidate"
        raise ValueError(msg)


def round_robin_key(call: SchedulableCall) -> str:
    if isinstance(call, LLMCall):
        return f"llm:{call.model_name or '*'}"
    if isinstance(call, ToolCall):
        return f"tool:{call.tool_name}"

    msg = "call must be an LLMCall or ToolCall"
    raise TypeError(msg)
