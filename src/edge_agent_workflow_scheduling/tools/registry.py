"""Registry for decoupling workers from concrete tool implementations."""

from __future__ import annotations

from dataclasses import dataclass, field

from edge_agent_workflow_scheduling.tools.base import Tool


@dataclass(slots=True)
class ToolRegistry:
    """Registry-backed container for tool executors."""

    _tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool, *, replace: bool = False) -> None:
        if not tool.tool_type:
            msg = "tool_type must be non-empty"
            raise ValueError(msg)
        if tool.tool_type in self._tools and not replace:
            msg = f"tool_type {tool.tool_type!r} is already registered"
            raise ValueError(msg)

        self._tools[tool.tool_type] = tool

    def get(self, tool_type: str) -> Tool:
        try:
            return self._tools[tool_type]
        except KeyError as exc:
            msg = f"tool_type {tool_type!r} is not registered"
            raise KeyError(msg) from exc

    def supported_tools(self) -> list[str]:
        return sorted(self._tools)

    def as_executor_mapping(self) -> dict[str, Tool]:
        return dict(self._tools)
