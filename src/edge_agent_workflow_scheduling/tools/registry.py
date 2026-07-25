"""Registry for decoupling workers from concrete tool implementations."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from jsonschema import Draft202012Validator

from edge_agent_workflow_scheduling.tools.base import (
    FunctionCallOutput,
    Tool,
    ToolExecution,
    ToolSpec,
    build_function_call_output,
)


@dataclass(slots=True)
class ToolRegistry:
    """Map function names to Tool definitions and local implementations.

    A registry describes available Tool functionality. A Worker that owns a
    registry is still a separately schedulable deployment replica.
    """

    _tools: dict[str, Tool] = field(default_factory=dict)
    _specs: dict[str, ToolSpec] = field(default_factory=dict)
    _validators: dict[str, Draft202012Validator] = field(default_factory=dict)

    def register(self, tool: Tool, *, replace: bool = False) -> None:
        if not tool.tool_name:
            msg = "tool_name must be non-empty"
            raise ValueError(msg)
        if tool.tool_name in self._tools and not replace:
            msg = f"tool_name {tool.tool_name!r} is already registered"
            raise ValueError(msg)
        spec = deepcopy(tool.spec)
        if spec["type"] != "function":
            raise ValueError("ToolSpec type must be 'function'")
        if spec["name"] != tool.tool_name:
            raise ValueError("ToolSpec name must match tool_name")
        if not spec["description"].strip():
            raise ValueError("ToolSpec description must be non-empty")
        if not isinstance(spec["strict"], bool):
            raise ValueError("ToolSpec strict must be a boolean")

        Draft202012Validator.check_schema(spec["parameters"])
        self._tools[tool.tool_name] = tool
        self._specs[tool.tool_name] = spec
        self._validators[tool.tool_name] = Draft202012Validator(spec["parameters"])

    def get(self, tool_name: str) -> Tool:
        try:
            return self._tools[tool_name]
        except KeyError as exc:
            msg = f"tool_name {tool_name!r} is not registered"
            raise KeyError(msg) from exc

    def supported_tools(self) -> list[str]:
        return sorted(self._tools)

    def tools(self) -> list[ToolSpec]:
        return [deepcopy(self._specs[tool_name]) for tool_name in self.supported_tools()]

    def execute(
        self,
        tool_name: str,
        arguments: str | Mapping[str, Any],
        *,
        invocation_id: str,
    ) -> ToolExecution:
        """Validate and execute one function call without choosing a Worker."""

        tool = self._tools.get(tool_name)
        if tool is None:
            return _failure("unknown_tool", f"tool_name {tool_name!r} is not registered")

        parsed_arguments = _parse_arguments(arguments)
        if isinstance(parsed_arguments, ToolExecution):
            return parsed_arguments

        validation_errors = sorted(
            self._validators[tool_name].iter_errors(parsed_arguments),
            key=lambda error: list(error.absolute_path),
        )
        if validation_errors:
            error = validation_errors[0]
            path = ".".join(str(part) for part in error.absolute_path)
            location = f"arguments.{path}" if path else "arguments"
            return _failure("invalid_arguments", f"{location}: {error.message}")

        try:
            result = tool.execute(parsed_arguments, invocation_id=invocation_id)
            if not isinstance(result, ToolExecution):
                raise TypeError("Tool.execute() must return ToolExecution")
            return result
        except Exception as exc:
            return _failure(
                "tool_execution_failed",
                str(exc) or exc.__class__.__name__,
            )

    def execute_call(self, function_call: Mapping[str, Any]) -> FunctionCallOutput:
        """Execute a Responses API function_call item and build its output item."""

        call_id = function_call.get("call_id")
        name = function_call.get("name")
        arguments = function_call.get("arguments")
        if not isinstance(call_id, str) or not call_id.strip():
            raise ValueError("function_call.call_id must be a non-empty string")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("function_call.name must be a non-empty string")
        if not isinstance(arguments, str | Mapping):
            result = _failure(
                "invalid_arguments",
                "function_call.arguments must be a JSON string or object",
            )
        else:
            result = self.execute(name, arguments, invocation_id=call_id)
        return build_function_call_output(call_id, result)


def _parse_arguments(arguments: str | Mapping[str, Any]) -> dict[str, Any] | ToolExecution:
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as exc:
            return _failure(
                "invalid_arguments_json",
                f"arguments is not valid JSON: {exc.msg}",
            )
    elif isinstance(arguments, Mapping):
        parsed = dict(arguments)
    else:
        return _failure("invalid_arguments", "arguments must be a JSON string or object")

    if not isinstance(parsed, dict):
        return _failure("invalid_arguments", "arguments must contain a JSON object")
    return parsed


def _failure(code: str, message: str) -> ToolExecution:
    return ToolExecution(
        success=False,
        error_code=code,
        error_message=message or code,
    )
