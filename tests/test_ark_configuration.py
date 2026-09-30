"""Credential-free checks for Ark profile routing and secret handling."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from edge_agent_workflow_scheduling.common import LLMCall
from edge_agent_workflow_scheduling.config import load_llm_profile, load_llm_profiles
from edge_agent_workflow_scheduling.executors.chat import create_openai_chat_executor
from edge_agent_workflow_scheduling.executors.openai import create_openai_client

CATALOG = Path(__file__).resolve().parents[1] / "configs/llm_profiles.toml"
CLOUD_IDS = ("online-glm-1", "online-glm-2")


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    for name in ("ARK_API_KEY", "ARK_BASE_URL", "ARK_PRIMARY_MODEL", "ARK_SECONDARY_MODEL"):
        monkeypatch.delenv(name, raising=False)


def test_cloud_profiles_load_without_credentials():
    profiles = load_llm_profiles(CATALOG)
    cloud = [profile for profile in profiles if profile.llm_id in CLOUD_IDS]
    assert len(cloud) == 2
    for profile in cloud:
        assert profile.provider == "volcengine"
        assert profile.model == "glm-5-3-flash-260828"
        assert profile.executor_type == "openai_chat"
        assert profile.secret_env_vars == ["ARK_API_KEY"]
        assert profile.metadata["function_calling_verified"] is False
        assert profile.quality_profile == profile.token_profile == profile.energy_profile == {}


@pytest.mark.parametrize("llm_id", CLOUD_IDS)
def test_missing_key_fails_before_client_creation(monkeypatch, llm_id):
    constructor = Mock()
    monkeypatch.setattr("openai.OpenAI", constructor)
    profile = load_llm_profile(CATALOG, llm_id)
    profile = replace(profile, deployment_config={**profile.deployment_config, "enabled": True})
    with pytest.raises(RuntimeError, match="ARK_API_KEY"):
        create_openai_client(profile)
    constructor.assert_not_called()


@pytest.mark.parametrize("llm_id", CLOUD_IDS)
def test_chat_request_uses_glm_and_ark_credentials(monkeypatch, llm_id):
    monkeypatch.setenv("ARK_API_KEY", "test-only-key")
    response = Mock()
    response.model_dump.return_value = {
        "id": "fake-response",
        "model": "glm-5-3-flash-260828",
        "choices": [{"finish_reason": "stop", "message": {"content": "42"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
    }
    create = Mock(return_value=response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    constructor = Mock(return_value=client)
    monkeypatch.setattr("openai.OpenAI", constructor)
    profile = load_llm_profile(CATALOG, llm_id)
    profile = replace(profile, deployment_config={**profile.deployment_config, "enabled": True})
    executor = create_openai_chat_executor(profile)
    result = executor.execute(
        LLMCall(
            llm_call_id="test-call",
            run_id="test-run",
            agent_id="test-agent",
            input_items=[{"role": "user", "content": "17 plus 25?"}],
        ),
        timeout_sec=10,
    )
    assert result.success
    assert result.output_text == "42"
    assert constructor.call_args.kwargs == {
        "api_key": "test-only-key",
        "max_retries": 0,
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
    }
    assert create.call_args.kwargs["model"] == "glm-5-3-flash-260828"
    assert "test-only-key" not in profile.to_json()
    assert "test-only-key" not in result.to_json()


def test_overrides_are_independent(monkeypatch):
    monkeypatch.setenv("ARK_PRIMARY_MODEL", "primary-test-model")
    monkeypatch.setenv("ARK_SECONDARY_MODEL", "secondary-test-model")
    monkeypatch.setenv("ARK_BASE_URL", "https://example.invalid/api/v3")
    first, second = [load_llm_profile(CATALOG, target) for target in CLOUD_IDS]
    assert first.model == "primary-test-model"
    assert second.model == "secondary-test-model"
    assert first.base_url == second.base_url == "https://example.invalid/api/v3"
