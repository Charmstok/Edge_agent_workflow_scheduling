"""Responses API executor for injected OpenAI-compatible clients."""

from __future__ import annotations

import os
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


def create_openai_responses_executor(
    profile: LLMInstanceProfile,
) -> OpenAIResponsesExecutor:
    """Create an OpenAI-compatible executor from a public profile and environment."""

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("the openai package is required for openai_responses executors") from exc

    requires_api_key = profile.deployment_config.get("requires_api_key", True)
    if not isinstance(requires_api_key, bool):
        raise ValueError("deployment_config.requires_api_key must be a boolean")
    if requires_api_key:
        if len(profile.secret_env_vars) != 1:
            raise ValueError(
                "openai_responses profiles requiring authentication must declare exactly "
                "one API key environment variable"
            )
        api_key_env = profile.secret_env_vars[0]
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"required environment variable {api_key_env!r} is not set")
    else:
        api_key = "local-no-auth"

    client_options: dict[str, Any] = {"api_key": api_key}
    if profile.base_url is not None:
        client_options["base_url"] = profile.base_url
    client = OpenAI(**client_options)

    model_parameters = profile.deployment_config.get("model_parameters", {})
    if not isinstance(model_parameters, dict):
        raise ValueError("deployment_config.model_parameters must be an object")
    return OpenAIResponsesExecutor(
        profile=profile,
        client=client,
        model_parameters=model_parameters,
    )
