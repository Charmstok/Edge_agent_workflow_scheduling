"""Shared helpers for scheduler policies."""

from __future__ import annotations

from edge_agent_workflow_scheduling.common import LLMCall, ToolCall
from edge_agent_workflow_scheduling.scheduler.types import SchedulingCandidate, WorkflowStep


def require_candidates(candidates: list[SchedulingCandidate]) -> None:
    if not candidates:
        msg = "scheduler policy requires at least one candidate"
        raise ValueError(msg)


def round_robin_key(step: WorkflowStep) -> str:
    if isinstance(step, LLMCall):
        return f"llm:{step.model_name or '*'}"
    if isinstance(step, ToolCall):
        return f"tool:{step.tool_type}"

    msg = "step must be an LLMCall or ToolCall"
    raise TypeError(msg)
