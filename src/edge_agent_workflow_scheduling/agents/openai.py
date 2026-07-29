"""OpenAI Responses API backend for FunctionCallingAgent."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openai import OpenAI

from edge_agent_workflow_scheduling.agents.function_calling import LLMResponse
from edge_agent_workflow_scheduling.common import LLMCall
from edge_agent_workflow_scheduling.tools import ToolSpec

_PROTECTED_OPTIONS = {
    "background",
    "input",
    "instructions",
    "model",
    "parallel_tool_calls",
    "stream",
    "timeout",
    "tools",
}


@dataclass(slots=True)
class OpenAIResponsesBackend:
    """Call an injected OpenAI-compatible Responses client and normalize its response."""

    model: str
    client: OpenAI
    response_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(self.response_options, dict):
            raise ValueError("response_options must be a dictionary")
        conflicting_options = sorted(_PROTECTED_OPTIONS & self.response_options.keys())
        if conflicting_options:
            raise ValueError(f"response_options cannot override {conflicting_options}")

    def create_response(
        self,
        llm_call: LLMCall,
        *,
        system_instruction: str,
        tools: list[ToolSpec],
        timeout_sec: float,
    ) -> LLMResponse:
        input_items = [
            deepcopy(item) for item in llm_call.input_items if item.get("role") != "system"
        ]
        response = self.client.responses.create(
            input=input_items,
            instructions=system_instruction,
            model=self.model,
            parallel_tool_calls=True,
            tools=tools,
            timeout=timeout_sec,
            **self.response_options,
        )
        status = getattr(response, "status", None)
        if status not in {None, "completed"}:
            error = getattr(response, "error", None)
            message = getattr(error, "message", None) or f"response ended with status {status!r}"
            raise RuntimeError(message)
        output_items = [item.model_dump(mode="json", exclude_none=True) for item in response.output]
        metadata: dict[str, Any] = {}
        if status is not None:
            metadata["status"] = status
        if response.usage is not None:
            metadata["usage"] = response.usage.model_dump(mode="json", exclude_none=True)
        return LLMResponse(
            output_items=output_items,
            output_text=response.output_text,
            response_id=response.id,
            model=response.model,
            metadata=metadata,
        )
