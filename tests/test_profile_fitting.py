from __future__ import annotations

import json
from pathlib import Path

import pytest

from edge_agent_workflow_scheduling.common import LLMCall, ToolCall
from edge_agent_workflow_scheduling.executors import ProfileToolExecutor
from edge_agent_workflow_scheduling.profiler.profile_fitting import (
    SyntheticEnergyConfig,
    fit_llm_profiles,
    fit_tool_profiles,
    write_profile_catalog,
)
from edge_agent_workflow_scheduling.resources import (
    ProfileScopeError,
    ToolReplicaProfile,
    resolve_llm_tokens_per_sec,
    resolve_tool_execution_time_sec,
)


def test_tool_fitting_exports_measured_and_synthetic_profiles(tmp_path: Path) -> None:
    run = _tool_run(tmp_path)
    energy = SyntheticEnergyConfig.from_mapping(
        {
            "version": "synthetic-energy-test-v1",
            "tool_joules_per_call": {"image_preprocess": 0.25},
            "llm_joules_per_token": {},
            "assumptions": "test-only synthetic constant",
        }
    )

    profiles = fit_tool_profiles(
        run,
        profile_version="test-profile-v1",
        synthetic_energy=energy,
    )

    assert len(profiles) == 2
    measured, synthetic = profiles
    assert measured.metadata["source_kind"] == "measured"
    assert measured.energy_profile == {}
    assert measured.latency_profile["execution_time_sec"] == pytest.approx(2.0)
    assert (
        measured.metadata["profile_lookup"]["latency_buckets"][0]["distribution"]["sample_count"]
        == 3
    )
    assert synthetic.metadata["source_kind"] == "synthetic"
    assert synthetic.metadata["physical_deployment"]["is_physical_device"] is False
    assert synthetic.latency_profile["execution_time_sec"] == pytest.approx(3.5)
    assert synthetic.energy_profile == {"joules_per_call": 0.25}


def test_bucket_lookup_is_shared_by_executor_and_rejects_unknown_scope(tmp_path: Path) -> None:
    measured = fit_tool_profiles(
        _tool_run(tmp_path),
        profile_version="test-profile-v1",
    )[0]
    call = _tool_call(input_size="small")

    resolved = resolve_tool_execution_time_sec(measured, call)
    result = ProfileToolExecutor(measured).execute(call)

    assert resolved.source == "bucket"
    assert resolved.value == pytest.approx(2.0)
    assert result.success is True
    assert result.execution_time_sec == pytest.approx(2.0)
    assert result.metadata["profile_metric"]["execution_time_sec_bucket_id"]

    with pytest.raises(ProfileScopeError, match="no latency_buckets entry"):
        resolve_tool_execution_time_sec(measured, _tool_call(input_size="extra-large"))


def test_aggregate_fallback_must_be_explicit(tmp_path: Path) -> None:
    profile = fit_tool_profiles(
        _tool_run(tmp_path),
        profile_version="test-profile-v1",
        fallback_policy="aggregate",
    )[0]

    resolved = resolve_tool_execution_time_sec(profile, _tool_call(input_size="unknown"))

    assert resolved.source == "aggregate"
    assert resolved.value == pytest.approx(2.0)


def test_llm_fitting_uses_task_size_buckets_and_token_scope(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        json.dumps(
            {
                "provenance": {},
                "llm_instances": [
                    {
                        "llm_id": "llm-a",
                        "provider": "benchmark",
                        "model": "model-a",
                        "node_id": "node-a",
                        "platform": "linux",
                        "executor_type": "profile",
                        "context_window_tokens": 4096,
                    }
                ],
                "samples": [
                    {
                        "llm_id": "llm-a",
                        "task_type": "summary",
                        "input_size": "small",
                        "input_tokens": 80,
                        "output_tokens": 20,
                        "request_time_sec": 2.0,
                        "success": True,
                        "energy_joules": None,
                    },
                    {
                        "llm_id": "llm-a",
                        "task_type": "summary",
                        "input_size": "small",
                        "input_tokens": 160,
                        "output_tokens": 40,
                        "request_time_sec": 2.0,
                        "success": True,
                        "energy_joules": None,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    profile = fit_llm_profiles(benchmark, profile_version="test-profile-v1")[0]
    call = LLMCall(
        llm_call_id="llm-call",
        run_id="run",
        agent_id="agent",
        input_tokens=80,
        estimated_output_tokens=20,
        metadata={"task_type": "summary", "input_size": "small"},
    )

    resolved = resolve_llm_tokens_per_sec(profile, call)

    assert resolved.source == "bucket"
    assert resolved.value == pytest.approx(75.0)
    with pytest.raises(ProfileScopeError, match="outside calibrated range"):
        resolve_llm_tokens_per_sec(
            profile,
            LLMCall(
                llm_call_id="outside",
                run_id="run",
                agent_id="agent",
                input_tokens=1000,
                estimated_output_tokens=20,
                metadata={"task_type": "summary", "input_size": "small"},
            ),
        )


def test_catalog_write_requires_explicit_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "profiles.json"
    write_profile_catalog({"version": "v1"}, output)

    with pytest.raises(FileExistsError, match="already exists"):
        write_profile_catalog({"version": "v2"}, output)

    write_profile_catalog({"version": "v2"}, output, overwrite=True)
    assert json.loads(output.read_text(encoding="utf-8"))["version"] == "v2"


def _tool_run(tmp_path: Path) -> Path:
    run = tmp_path / "sampling"
    combination = run / "image_preprocess-small-c1"
    combination.mkdir(parents=True)
    profile = ToolReplicaProfile(
        replica_id="image-linux-local",
        tool_name="image_preprocess",
        node_id="linux-local",
        platform="linux",
        implementation_version="pillow-test",
        executor_type="local",
    )
    manifest = {
        "sampling_config": {"sampling_id": "test"},
        "host": {"os": "Linux", "hostname_hash": "test-host"},
    }
    summary = {
        "combinations": [
            {
                "combination_id": "image_preprocess-small-c1",
                "tool_name": "image_preprocess",
                "scale": "small",
                "concurrency": 1,
                "status": "completed",
                "resource_profile": profile.to_dict(),
            }
        ]
    }
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    rows = [
        {
            "result": {
                "success": True,
                "execution_time_sec": value,
                "energy_joules": 0.0,
                "metadata": {},
            }
        }
        for value in (1.0, 2.0, 3.0)
    ]
    (combination / "measurement.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return run


def _tool_call(*, input_size: str) -> ToolCall:
    return ToolCall(
        tool_call_id=f"tool-{input_size}",
        call_id=f"function-{input_size}",
        run_id="run",
        agent_id="agent",
        tool_name="image_preprocess",
        arguments={},
        metadata={"input_size": input_size},
    )
