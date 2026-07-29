"""Responses API executor for injected OpenAI-compatible clients."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openai import OpenAI

from edge_agent_workflow_scheduling.agents.openai import OpenAIResponsesBackend
from edge_agent_workflow_scheduling.common import LLMCall, LLMResult
from edge_agent_workflow_scheduling.executors.adapters import BackendLLMExecutor
from edge_agent_workflow_scheduling.resources import LLMInstanceProfile
from edge_agent_workflow_scheduling.tools import ToolSpec


@dataclass(slots=True)
class OpenAIResponsesExecutor:
    """Execute an LLMCall through an injected Responses-compatible client."""

    profile: LLMInstanceProfile
    client: OpenAI
    model_parameters: dict[str, Any] = field(default_factory=dict)
    _adapter: BackendLLMExecutor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        backend = OpenAIResponsesBackend(
            model=self.profile.model,
            client=self.client,
            response_options=self.model_parameters,
        )
        self._adapter = BackendLLMExecutor(profile=self.profile, backend=backend)

    def execute(
        self,
        llm_call: LLMCall,
        *,
        tools: list[ToolSpec] | None = None,
        timeout_sec: float | None = None,
    ) -> LLMResult:
        return self._adapter.execute(
            llm_call,
            tools=tools,
            timeout_sec=timeout_sec,
        )
