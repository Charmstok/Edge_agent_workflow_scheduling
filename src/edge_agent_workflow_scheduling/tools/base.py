"""Common interfaces for executable tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, TypedDict

from edge_agent_workflow_scheduling.common import ToolCall


class ToolSpec(TypedDict):
    """OpenAI function tool definition consumed by agents."""

    type: Literal["function"]
    name: str
    description: str
    parameters: dict[str, object]
    strict: bool


@dataclass(frozen=True, slots=True)
class ToolExecution:
    """Structured output returned by tools that expose execution metadata."""

    output_uri: str | None
    metadata: dict[str, object] = field(default_factory=dict)


class Tool(Protocol):
    """Interface implemented by concrete tool executors."""

    tool_name: str
    spec: ToolSpec

    def __call__(self, tool_call: ToolCall) -> ToolExecution:
        """Execute a tool call and return structured output."""
