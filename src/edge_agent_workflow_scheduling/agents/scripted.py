"""Deterministic Responses-style backend for offline Agent verification."""

from __future__ import annotations

import json
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from edge_agent_workflow_scheduling.agents.function_calling import LLMResponse
from edge_agent_workflow_scheduling.common import LLMCall
from edge_agent_workflow_scheduling.tools import ToolSpec


@dataclass(frozen=True, slots=True)
class ScriptedFunctionCall:
    """One deterministic function_call emitted by a scripted turn."""

    call_id: str
    name: str
    arguments: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.call_id, str) or not self.call_id.strip():
            raise ValueError("call_id must be a non-empty string")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        if not isinstance(self.arguments, dict):
            raise ValueError("arguments must be a JSON object")
        try:
            json.dumps(self.arguments, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("arguments must be JSON serializable") from exc


@dataclass(slots=True)
class ScriptedLLMBackend:
    """Return a finite sequence of predetermined LLM responses."""

    responses: list[LLMResponse]
    _received_calls: list[LLMCall] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.responses:
            raise ValueError("responses must be non-empty")
        self.responses = deepcopy(self.responses)

    @property
    def received_calls(self) -> tuple[LLMCall, ...]:
        return tuple(self._received_calls)

    def create_response(
        self,
        llm_call: LLMCall,
        *,
        system_instruction: str,
        tools: list[ToolSpec],
        timeout_sec: float,
    ) -> LLMResponse:
        del system_instruction, tools, timeout_sec
        self._received_calls.append(deepcopy(llm_call))
        response_index = len(self._received_calls) - 1
        if response_index >= len(self.responses):
            raise RuntimeError("scripted LLM response sequence exhausted")
        return deepcopy(self.responses[response_index])

    @classmethod
    def single_tool(
        cls,
        function_call: ScriptedFunctionCall,
        *,
        final_text: str = "Done.",
    ) -> ScriptedLLMBackend:
        return cls.from_function_calls([function_call], final_text=final_text)

    @classmethod
    def multiple_tools(
        cls,
        function_calls: Iterable[ScriptedFunctionCall],
        *,
        final_text: str = "Done.",
    ) -> ScriptedLLMBackend:
        return cls.from_function_calls(function_calls, final_text=final_text)

    @classmethod
    def from_function_calls(
        cls,
        function_calls: Iterable[ScriptedFunctionCall],
        *,
        final_text: str,
    ) -> ScriptedLLMBackend:
        calls = list(function_calls)
        if not calls:
            raise ValueError("function_calls must be non-empty")
        if not final_text.strip():
            raise ValueError("final_text must be non-empty")
        return cls(
            responses=[
                LLMResponse(
                    output_items=[_function_call_item(call) for call in calls],
                    response_id="resp_scripted_tools",
                ),
                LLMResponse(
                    output_items=[_message_item(final_text)],
                    output_text=final_text,
                    response_id="resp_scripted_final",
                ),
            ]
        )


def _function_call_item(function_call: ScriptedFunctionCall) -> dict[str, Any]:
    return {
        "arguments": json.dumps(
            function_call.arguments,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        "call_id": function_call.call_id,
        "name": function_call.name,
        "status": "completed",
        "type": "function_call",
    }


def _message_item(text: str) -> dict[str, Any]:
    return {
        "content": [
            {
                "annotations": [],
                "text": text,
                "type": "output_text",
            }
        ],
        "role": "assistant",
        "status": "completed",
        "type": "message",
    }
