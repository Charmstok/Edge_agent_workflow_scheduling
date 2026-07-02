"""Common interfaces for executable tools."""

from __future__ import annotations

from typing import Protocol

from edge_agent_workflow_scheduling.common import ToolCall


class Tool(Protocol):
    """Interface implemented by concrete tool executors."""

    tool_type: str

    def __call__(self, tool_call: ToolCall) -> str | None:
        """Execute a tool call and return an output URI."""
