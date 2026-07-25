"""Common interfaces for executable tools."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypedDict

from edge_agent_workflow_scheduling.common import ToolResult


class ToolSpec(TypedDict):
    """OpenAI function tool definition consumed by agents."""

    type: Literal["function"]
    name: str
    description: str
    parameters: dict[str, Any]
    strict: bool


class FunctionCallOutput(TypedDict):
    """Responses API input item returned after one function call."""

    type: Literal["function_call_output"]
    call_id: str
    output: str


@dataclass(frozen=True, slots=True)
class ToolExecution:
    """Provider-neutral result returned by a local Tool implementation."""

    success: bool
    output: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        _ensure_json_serializable(self.output, "output")
        _ensure_json_serializable(self.metadata, "metadata")
        if self.success and (self.error_code is not None or self.error_message is not None):
            raise ValueError("successful ToolExecution cannot contain an error")
        if not self.success and (not self.error_code or not self.error_message):
            raise ValueError("failed ToolExecution requires error_code and error_message")


class Tool(Protocol):
    """Interface implemented by concrete tool executors."""

    tool_name: str
    spec: ToolSpec

    def execute(
        self,
        arguments: dict[str, Any],
        *,
        invocation_id: str,
    ) -> ToolExecution:
        """Execute parsed function arguments without selecting a replica."""


def build_function_call_output(
    call_id: str,
    result: ToolExecution | ToolResult,
) -> FunctionCallOutput:
    """Serialize one Tool result using the project's canonical wire format."""

    if not call_id.strip():
        raise ValueError("call_id must be non-empty")
    if result.success:
        payload: dict[str, Any] = {"result": result.output, "success": True}
    else:
        payload = {
            "error": {
                "code": result.error_code or "tool_execution_failed",
                "message": result.error_message or "Tool execution failed",
            },
            "success": False,
        }
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
    }


def _ensure_json_serializable(value: Any, field_name: str) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be JSON serializable") from exc
