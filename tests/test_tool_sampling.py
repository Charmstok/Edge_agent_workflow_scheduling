from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from edge_agent_workflow_scheduling.profiler.tool_sampling import (
    ProcessTreeSampler,
    ToolSamplingConfig,
    collect_host_inventory,
    percentile,
    run_tool_sampling,
    summarize_measurements,
)


def _config() -> ToolSamplingConfig:
    return ToolSamplingConfig(
        sampling_id="test-sampling",
        tools=("image_preprocess",),
        scales=("small",),
        concurrency_levels=(1, 2),
        cold_start_runs=1,
        warmup_runs_per_worker=1,
        measurement_repetitions_per_worker=2,
        timeout_sec=10,
        resource_sample_interval_sec=0.01,
        input_templates={"image_preprocess": "fixture-{scale}.png"},
        image_preprocess={
            "operations": ["grayscale", "blur", "threshold", "edge_detect"],
            "operation_repeat": 2,
        },
    )


def _write_config(tmp_path: Path, config: ToolSamplingConfig) -> Path:
    path = tmp_path / "sampling.json"
    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    Image.new("RGB", (640, 480), "white").save(tmp_path / "fixture-small.png")
    return path


def test_sampling_matrix_separates_phases_and_measures_queue(tmp_path: Path) -> None:
    config = _config()
    config_path = _write_config(tmp_path, config)

    output = run_tool_sampling(
        config,
        config_path=config_path,
        output_root=tmp_path / "output",
        experiment_id="test-run",
    )

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["completed_combination_count"] == 2
    assert summary["skipped_combination_count"] == 0
    assert summary["measurement_call_count"] == 6
    assert summary["all_terminal_measurements_retained"] is True
    for concurrency, expected_counts in ((1, (1, 1, 2)), (2, (1, 2, 4))):
        directory = output / f"image_preprocess-small-c{concurrency}"
        cold = _read_jsonl(directory / "cold_start.jsonl")
        warmup = _read_jsonl(directory / "warmup.jsonl")
        measured = _read_jsonl(directory / "measurement.jsonl")
        assert (len(cold), len(warmup), len(measured)) == expected_counts
        assert all(record["phase"] == "measurement" for record in measured)
        assert all(record["running_at_start"] <= concurrency for record in measured)
        assert all(record["queue_depth_after_submit"] >= 1 for record in measured)
        assert all(record["result"]["queue_wait_time_sec"] >= 0 for record in measured)
        assert all(record["result"]["input_transfer_time_sec"] == 0 for record in measured)
        assert all(record["result"]["output_transfer_time_sec"] == 0 for record in measured)
        assert all(
            record["result"]["metadata"]["transfer_measurement"] == "local_same_host_no_transfer"
            for record in measured
        )
        combination = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        metrics = combination["phases"]["measurement"]["metrics"]
        assert metrics["sample_count"] == len(measured)
        assert metrics["success_rate"] == 1
        assert metrics["excluded_count"] == 0
        assert metrics["execution_time_sec"]["p95"] > 0
        assert metrics["throughput_calls_per_sec"] > 0
        assert (directory / "cold_start_resource_samples.jsonl").is_file()
        assert len(_read_jsonl(directory / "trace.jsonl")) == sum(expected_counts)


def test_sampling_config_round_trip_preserves_matrix(tmp_path: Path) -> None:
    path = _write_config(tmp_path, _config())
    loaded = ToolSamplingConfig.from_json(path)

    assert loaded == _config()
    assert loaded.to_dict()["concurrency_levels"] == [1, 2]


def test_sampling_fails_before_execution_for_missing_input(tmp_path: Path) -> None:
    config = _config()
    config_path = tmp_path / "sampling.json"
    config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="sampling input not found"):
        run_tool_sampling(
            config,
            config_path=config_path,
            output_root=tmp_path / "output",
            experiment_id="missing-input",
        )


def test_sampling_manifest_records_host_without_direct_identifiers(tmp_path: Path) -> None:
    config = replace(_config(), concurrency_levels=(1,))
    config_path = _write_config(tmp_path, config)
    output = run_tool_sampling(
        config,
        config_path=config_path,
        output_root=tmp_path / "output",
        experiment_id="host-run",
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["timing_method"] == "time.perf_counter monotonic wall clock"
    assert manifest["transfer_model"] == "local_same_host_no_transfer"
    assert manifest["host"]["hostname_hash"]
    serialized = json.dumps(manifest["host"]).casefold()
    assert "serial_number" not in serialized
    assert "platform_uuid" not in serialized
    assert "provisioning_udid" not in serialized


def test_resource_sampler_reports_unsupported_metrics_as_unavailable() -> None:
    sampler = ProcessTreeSampler(0.01, psutil_module=None)
    sampler.start()
    summary = sampler.stop()
    assert summary["cpu"]["status"] == "unavailable"
    assert summary["memory"]["status"] == "unavailable"
    assert summary["gpu_utilization"]["status"] == "unavailable"
    host = collect_host_inventory(psutil_module=None)
    assert host["memory_bytes"] is None
    assert host["physical_cpu_count"] is None
    assert host["resource_sampler"]["status"] == "unavailable"


def test_summary_retains_failed_and_timeout_samples() -> None:
    records = [
        _record(success=True, execution=1.0, queue=0.1),
        _record(success=False, execution=2.0, queue=0.2, error_code="timeout"),
        _record(success=False, execution=3.0, queue=0.3, error_code="invalid_input"),
    ]
    summary = summarize_measurements(records, wall_time_sec=4.0)
    assert summary["sample_count"] == 3
    assert summary["included_count"] == 3
    assert summary["excluded_count"] == 0
    assert summary["success_rate"] == pytest.approx(1 / 3)
    assert summary["timeout_count"] == 1
    assert summary["failure_codes"] == {"timeout": 1, "invalid_input": 1}
    assert summary["execution_time_sec"]["mean"] == 2
    assert summary["throughput_calls_per_sec"] == 0.75


def test_percentile_interpolates_and_rejects_invalid_input() -> None:
    assert percentile([2, 4], 0.95) == pytest.approx(3.9)
    assert percentile([3], 0.99) == 3
    with pytest.raises(ValueError, match="at least one"):
        percentile([], 0.5)
    with pytest.raises(ValueError, match="between 0 and 1"):
        percentile([1], 1.1)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("tools", ("unknown",), "unsupported"),
        ("scales", ("tiny",), "unsupported"),
        ("concurrency_levels", (0,), "concurrency level"),
        ("cold_start_runs", 0, "cold_start_runs"),
        ("measurement_repetitions_per_worker", 0, "measurement_repetitions"),
        ("timeout_sec", float("nan"), "timeout_sec"),
        ("resource_sample_interval_sec", 0, "resource_sample_interval_sec"),
    ],
)
def test_sampling_config_rejects_invalid_values(field: str, value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_config(), **{field: value})


def test_sampling_does_not_overwrite_existing_run(tmp_path: Path) -> None:
    config = replace(_config(), concurrency_levels=(1,))
    config_path = _write_config(tmp_path, config)
    run_tool_sampling(
        config,
        config_path=config_path,
        output_root=tmp_path / "output",
        experiment_id="same-run",
    )
    with pytest.raises(FileExistsError):
        run_tool_sampling(
            config,
            config_path=config_path,
            output_root=tmp_path / "output",
            experiment_id="same-run",
        )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _record(
    *, success: bool, execution: float, queue: float, error_code: str | None = None
) -> dict:
    return {
        "result": {
            "success": success,
            "queue_wait_time_sec": queue,
            "input_transfer_time_sec": 0,
            "execution_time_sec": execution,
            "output_transfer_time_sec": 0,
            "error_code": error_code,
        }
    }
