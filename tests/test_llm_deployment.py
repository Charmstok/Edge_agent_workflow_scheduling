"""Offline activation, secret-safe telemetry and real HTTP adapter integration."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from edge_agent_workflow_scheduling.common import LLMCall
from edge_agent_workflow_scheduling.config import load_llm_profiles, load_tool_profiles
from edge_agent_workflow_scheduling.executors.chat import (
    OpenAIChatExecutor,
    create_openai_chat_executor,
)
from edge_agent_workflow_scheduling.executors.llm_deployment import LLMDeployment
from edge_agent_workflow_scheduling.executors.openai import (
    OpenAIResponsesExecutor,
    create_openai_responses_executor,
)
from edge_agent_workflow_scheduling.profiler.llm_deployment import (
    catalog_report,
    live_checks,
    offline_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "configs/llm_profiles.toml"
CONFIG = json.loads((ROOT / "configs/llm_deployment_validation_v1.json").read_text())


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in (
        "ARK_API_KEY",
        "ARK_BASE_URL",
        "ARK_PRIMARY_MODEL",
        "ARK_SECONDARY_MODEL",
        "QWEN9B_BASE_URL",
        "QWEN27B_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)


def call():
    return LLMCall(
        llm_call_id="adapter-test",
        run_id="test",
        agent_id="test",
        input_items=[{"role": "user", "content": "17 plus 25?"}],
    )


def test_offline_contract_never_constructs_clients_and_masks_cloud(tmp_path, monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "test-only-offline-key")
    factory = Mock(side_effect=AssertionError("offline must not construct clients"))
    profiles = load_llm_profiles(CATALOG)
    with LLMDeployment(
        profiles, offline_rates=CONFIG["offline_tokens_per_sec"], client_factory=factory
    ) as deployment:
        report = offline_contract(deployment, tmp_path)
        assert report["passed"]
        assert set(report["selected_targets"]) == set(CONFIG["required_local_ids"])
        assert report["network_requests"] == 0
        for target in CONFIG["required_cloud_ids"]:
            assert report["action_mask"][target] is False
            assert report["activation"][target]["status"] == "skipped"
            assert report["states"][target]["is_online"] is False
        assert all(p.token_profile == {} for p in profiles)
        assert all(
            s.profile.metadata["hardware_performance_claim"] is False
            for s in deployment.resources.llm_snapshots()
            if s.state.is_online
        )
    factory.assert_not_called()
    assert "test-only-offline-key" not in (tmp_path / "execution_records.json").read_text()


def test_local_and_cloud_overrides_have_independent_provenance(monkeypatch):
    overrides = {
        "QWEN9B_BASE_URL": "http://localhost:9010/v1",
        "QWEN27B_BASE_URL": "http://localhost:9020/v1",
        "ARK_SECONDARY_MODEL": "secondary-model-test",
    }
    for key, value in overrides.items():
        monkeypatch.setenv(key, value)
    profiles = {p.llm_id: p for p in load_llm_profiles(CATALOG)}
    assert profiles["local-qwen35-9b"].base_url == overrides["QWEN9B_BASE_URL"]
    assert profiles["local-qwen38-27b"].base_url == overrides["QWEN27B_BASE_URL"]
    assert profiles["online-glm-2"].model == "secondary-model-test"
    assert profiles["online-glm-1"].model != "secondary-model-test"
    assert profiles["online-glm-2"].metadata["resolved_environment_overrides"] == {
        "model": "ARK_SECONDARY_MODEL"
    }
    report = catalog_report(list(profiles.values()), CONFIG)
    for item in report.values():
        assert all(not value["available"] for value in item["calibration"].values())
        assert item["capability_source"] and item["context_source"]


def test_disabled_local_profiles_fail_offline_acceptance_with_a_report(tmp_path):
    profiles = [
        replace(p, deployment_config={**p.deployment_config, "enabled": False})
        for p in load_llm_profiles(CATALOG)
    ]
    with LLMDeployment(profiles, offline_rates=CONFIG["offline_tokens_per_sec"]) as deployment:
        report = offline_contract(deployment, tmp_path)
        assert not report["passed"]
        assert report["selected_targets"] == []
        assert not any(report["action_mask"].values())


def test_declared_context_limits_mask_oversized_calls():
    with LLMDeployment(
        load_llm_profiles(CATALOG), offline_rates=CONFIG["offline_tokens_per_sec"]
    ) as deployment:
        oversized = replace(call(), context_length=32769)
        details = deployment.resources.action_mask_details(oversized)
        assert not any(details.values)
        for target in CONFIG["required_local_ids"]:
            assert "context_window_exceeded" in details.reasons_by_target()[target]


@pytest.mark.parametrize("factory", [create_openai_chat_executor, create_openai_responses_executor])
def test_cloud_key_alone_does_not_enable_client(monkeypatch, factory):
    monkeypatch.setenv("ARK_API_KEY", "test-only-key")
    constructor = Mock()
    monkeypatch.setattr("openai.OpenAI", constructor)
    cloud = next(p for p in load_llm_profiles(CATALOG) if p.llm_id == "online-glm-1")
    with pytest.raises(RuntimeError, match="explicitly enabled"):
        factory(cloud)
    constructor.assert_not_called()


@pytest.mark.parametrize(
    "endpoint",
    ["https://user:password@example.invalid/v1", "https://example.invalid/v1?api_key=secret"],
)
def test_endpoint_overrides_reject_embedded_credentials(monkeypatch, endpoint):
    monkeypatch.setenv("ARK_BASE_URL", endpoint)
    with pytest.raises(ValueError, match="without credentials"):
        load_llm_profiles(CATALOG)


def test_unreachable_live_models_are_offline_and_clients_are_closed():
    clients = []

    def factory(profile):
        client = SimpleNamespace(
            models=SimpleNamespace(list=Mock(side_effect=ConnectionError("offline"))), close=Mock()
        )
        clients.append(client)
        return client

    with LLMDeployment(
        load_llm_profiles(CATALOG), mode="live", client_factory=factory
    ) as deployment:
        assert not deployment.executors
        assert not any(deployment.resources.action_mask(call()).values())
        assert all(
            deployment.reports[target]["status"] == "offline"
            for target in CONFIG["required_local_ids"]
        )
        assert all(
            not deployment.reports[target]["client_created"]
            for target in CONFIG["required_cloud_ids"]
        )
    assert len(clients) == 2
    for client in clients:
        client.close.assert_called_once()


@pytest.mark.parametrize(
    "failure",
    [TimeoutError("Authorization test-only-key"), RuntimeError("Authorization test-only-key")],
)
def test_provider_failures_keep_metadata_without_echoing_credentials(monkeypatch, failure):
    monkeypatch.setenv("ARK_API_KEY", "test-only-key")
    profile = next(p for p in load_llm_profiles(CATALOG) if p.llm_id == "online-glm-1")
    create = Mock(side_effect=failure)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result = OpenAIChatExecutor(profile, client).execute(call(), timeout_sec=1)
    assert not result.success
    assert result.error_code == (
        "timeout" if isinstance(failure, TimeoutError) else "llm_execution_failed"
    )
    assert "test-only-key" not in result.to_json()
    assert result.metadata["provider"] == "volcengine"
    assert result.metadata["network_inclusive_request_time_sec"] == result.inference_time_sec
    assert result.metadata["network_only_time_sec"] is None
    assert result.input_transfer_time_sec == result.output_transfer_time_sec == 0


def test_responses_adapter_redacts_echoed_secrets_and_keeps_provenance(monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "test-only-key")
    profile = next(p for p in load_llm_profiles(CATALOG) if p.llm_id == "online-glm-1")
    output = Mock()
    output.model_dump.return_value = {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "test-only-key"}],
    }
    response = SimpleNamespace(
        status="completed",
        id="response-test",
        model=profile.model,
        output=[output],
        output_text="test-only-key",
        usage=None,
        model_dump=Mock(return_value={"Authorization": "Bearer test-only-key"}),
    )
    client = SimpleNamespace(responses=SimpleNamespace(create=Mock(return_value=response)))
    result = OpenAIResponsesExecutor(profile, client).execute(call(), timeout_sec=1)
    assert result.success
    assert "test-only-key" not in result.to_json()
    assert result.output_text == "[REDACTED]"
    assert result.metadata["provider"] == "volcengine"
    assert len(result.metadata["request_parameters_sha256"]) == 64


@pytest.fixture
def adapter_server():
    requests = []
    models = [p.model for p in load_llm_profiles(CATALOG)]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, value):
            data = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.reply(
                {"object": "list", "data": [{"id": name, "object": "model"} for name in models]}
            )

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            tool_result = next((m for m in payload["messages"] if m["role"] == "tool"), None)
            prompt = next(m["content"] for m in payload["messages"] if m["role"] == "user")
            if "Convert this local image" in prompt and tool_result is None:
                uri = re.search(r"file:\S+", prompt).group()
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "fn-image",
                            "type": "function",
                            "function": {
                                "name": "image_preprocess",
                                "arguments": json.dumps(
                                    {
                                        "input_uri": uri,
                                        "operations": ["grayscale", "resize"],
                                        "operation_repeat": 1,
                                    }
                                ),
                            },
                        }
                    ],
                }
                finish = "tool_calls"
            else:
                message = {
                    "role": "assistant",
                    "content": tool_result["content"] if tool_result else "42",
                }
                finish = "stop"
            self.reply(
                {
                    "id": "test-response",
                    "object": "chat.completion",
                    "created": 0,
                    "model": payload["model"],
                    "choices": [{"index": 0, "finish_reason": finish, "message": message}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
                }
            )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_live_http_function_calling_executes_real_image_replicas(tmp_path, adapter_server):
    endpoint, requests = adapter_server
    profiles = [
        replace(p, base_url=endpoint) if p.platform != "cloud" else p
        for p in load_llm_profiles(CATALOG)
    ]
    config = dict(CONFIG)
    tool_profiles = [
        p.to_dict()
        for p in load_tool_profiles(ROOT / "configs/tool_profiles.toml")
        if p.tool_name == "image_preprocess"
    ]
    catalog = tmp_path / "tools.json"
    catalog.write_text(json.dumps({"tool_replicas": tool_profiles}))
    samples = json.loads((ROOT / "configs/tool_deployment_samples_v1.json").read_text())
    samples["samples"] = [samples["samples"][0]]
    shutil.copy(ROOT / "configs/workload_fixtures_v1/alpha-small.png", tmp_path / "sample.png")
    samples["samples"][0]["arguments"]["input_uri"] = "sample.png"
    sample_path = tmp_path / "samples.json"
    sample_path.write_text(json.dumps(samples))
    config.update(tool_catalog=str(catalog), tool_consistency_samples=str(sample_path))
    result = live_checks(profiles, config, output_dir=tmp_path / "live", live_local=True)
    assert result["local_function_calling_verified"]
    assert all(
        result["checks"][target]["status"] == "passed" for target in CONFIG["required_local_ids"]
    )
    assert len(requests) == 6  # Tool request, real Tool feedback, then no-Tool request per model.
    assert any(any(m["role"] == "tool" for m in r["messages"]) for r in requests)
    for p in (tmp_path / "live").glob("local-*/*/trace.json"):
        trace = json.loads(p.read_text())
        llm_rows = [row for row in trace["calls"] if row["call_kind"] == "llm"]
        assert llm_rows
        assert all(row["result_metadata"]["provider"] == "vllm" for row in llm_rows)


def test_explicit_cloud_smoke_is_capped_and_does_not_require_real_local_models(
    tmp_path,
    adapter_server,
    monkeypatch,
):
    endpoint, requests = adapter_server
    monkeypatch.setenv("ARK_API_KEY", "test-only-key")
    # Only the loopback adapter fixture is contacted; this is not provider validation.
    profiles = [replace(p, base_url=endpoint) for p in load_llm_profiles(CATALOG)]
    result = live_checks(profiles, CONFIG, output_dir=tmp_path, cloud_smoke=True)
    assert len(requests) == 2
    assert all(r["max_tokens"] == CONFIG["max_output_tokens"] for r in requests)
    assert not result["local_function_calling_verified"]
    assert all(
        result["checks"][target]["status"] == "passed" for target in CONFIG["required_cloud_ids"]
    )
    assert "test-only-key" not in (tmp_path / "cloud_trace.jsonl").read_text()
