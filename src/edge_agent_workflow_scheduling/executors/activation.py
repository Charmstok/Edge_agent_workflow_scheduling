"""Client activation checks shared by local and cloud LLM entry points."""

from __future__ import annotations

import os

from edge_agent_workflow_scheduling.resources import LLMInstanceProfile


def llm_activation_error(profile: LLMInstanceProfile) -> tuple[str, str] | None:
    enabled = profile.deployment_config.get("enabled", profile.platform != "cloud")
    if not isinstance(enabled, bool):
        raise ValueError("deployment_config.enabled must be a boolean")
    if not enabled:
        return "deployment_disabled", "LLM deployment must be explicitly enabled"
    requires_key = profile.deployment_config.get("requires_api_key", True)
    if not isinstance(requires_key, bool):
        raise ValueError("deployment_config.requires_api_key must be a boolean")
    if requires_key:
        if len(profile.secret_env_vars) != 1:
            raise ValueError(
                "authenticated LLM profiles must declare one API key environment variable"
            )
        name = profile.secret_env_vars[0]
        if not os.getenv(name, "").strip():
            return "credentials_missing", f"required environment variable {name!r} is not set"
    return None
