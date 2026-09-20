from __future__ import annotations

import json
from pathlib import Path

import pytest

from edge_agent_workflow_scheduling.profiler.profile_fitting import fit_tool_profiles
from edge_agent_workflow_scheduling.profiler.profile_validation import (
    ProfileValidationConfig,
    validate_profile_catalog,
    write_profile_validation_artifacts,
)
from edge_agent_workflow_scheduling.resources import ToolReplicaProfile


def test_profile_validation_reports_errors_thresholds_and_unverified_metrics(
    tmp_path: Path,
) -> None:
    calibration = _sampling_run(
        tmp_path / "calibration",
        source_hash="a" * 64,
        input_template="alpha-{scale}.png",
        values=(1.0, 2.0, 3.0),
    )
    holdout = _sampling_run(
        tmp_path / "holdout",
        source_hash="b" * 64,
        input_template="beta-{scale}.png",
        values=(1.9, 2.1),
    )
    catalog_path = _catalog(tmp_path, calibration)

    report, samples = validate_profile_catalog(catalog_path, holdout, _config())

    assert report["status"] == "passed"
    assert report["data_leakage_check"]["status"] == "passed"
    assert report["tool_execution_time"]["sample_count"] == 2
    assert report["tool_execution_time"]["overall"]["after"]["mae_sec"] == pytest.approx(
        0.1
    )
    assert report["tool_execution_time"]["thresholds"]["passed"] is True
    assert report["llm_request_time_or_throughput"]["status"] == "not_validated"
    assert report["energy"]["status"] == "not_validated"
    assert len(samples) == 2

    output = write_profile_validation_artifacts(tmp_path / "report", report, samples)
    assert (output / "report.json").is_file()
    assert len((output / "samples.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_profile_validation_rejects_overlapping_calibration_and_holdout_inputs(
    tmp_path: Path,
) -> None:
    calibration = _sampling_run(
        tmp_path / "calibration",
        source_hash="a" * 64,
        input_template="alpha-{scale}.png",
        values=(1.0, 2.0, 3.0),
    )
    holdout = _sampling_run(
        tmp_path / "holdout",
        source_hash="a" * 64,
        input_template="beta-{scale}.png",
        values=(1.9, 2.1),
    )

    with pytest.raises(ValueError, match="overlap by content hash"):
        validate_profile_catalog(_catalog(tmp_path, calibration), holdout, _config())


def _config() -> ProfileValidationConfig:
    return ProfileValidationConfig(
        validation_id="test-validation-v1",
        relative_error_denominator_epsilon=1e-9,
        comparison={
            "before_calibration": "aggregate_profile_mean",
            "after_calibration": "matching_bucket_mean",
            "distribution": "calibration_bucket_distribution_vs_holdout_distribution",
        },
        tool_execution_time_thresholds={
            "minimum_total_sample_count": 2,
            "minimum_group_sample_count": 2,
            "maximum_overall_mae_sec": 0.2,
            "maximum_overall_mean_relative_error": 0.2,
            "maximum_group_mean_relative_error": 0.2,
        },
        stochastic_validation={
            "seeds": [1, 2],
            "samples_per_seed_per_group": 3,
        },
        policy={"thresholds_declared_before_holdout_sampling": True},
    )


def _catalog(tmp_path: Path, calibration: Path) -> Path:
    profiles = fit_tool_profiles(calibration, profile_version="test-profile-v1")
    path = tmp_path / "profiles.json"
    path.write_text(
        json.dumps(
            {
                "version": "test-profile-v1",
                "resource_profiles": {
                    "llm_instances": [],
                    "tool_replicas": [
                        {"profile": profile.to_dict(), "state": {}} for profile in profiles
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _sampling_run(
    path: Path,
    *,
    source_hash: str,
    input_template: str,
    values: tuple[float, ...],
) -> Path:
    combination = path / "image_preprocess-small-c1"
    combination.mkdir(parents=True)
    sampling_config = {
        "tools": ["image_preprocess"],
        "scales": ["small"],
        "concurrency_levels": [1],
        "input_templates": {"image_preprocess": input_template},
        "image_preprocess": {
            "operations": ["grayscale"],
            "operation_repeat": 1,
        },
    }
    profile = ToolReplicaProfile(
        replica_id="image-preprocess-local",
        tool_name="image_preprocess",
        node_id="linux-local",
        platform="linux",
        implementation_version="pillow-test",
        executor_type="local",
    )
    manifest = {
        "host": {"hostname_hash": "test-host"},
        "sampling_config": sampling_config,
    }
    summary = {
        "experiment_id": path.name,
        "combinations": [
            {
                "combination_id": "image_preprocess-small-c1",
                "tool_name": "image_preprocess",
                "scale": "small",
                "concurrency": 1,
                "status": "completed",
                "source_sha256": source_hash,
                "resource_profile": profile.to_dict(),
            }
        ],
    }
    records = [
        {
            "call": {
                "tool_call_id": f"sample-{index}",
                "metadata": {"task_type": "tool_profile_sampling"},
            },
            "result": {
                "success": True,
                "execution_time_sec": value,
                "energy_joules": 0.0,
                "metadata": {"energy_source": "unavailable"},
            },
        }
        for index, value in enumerate(values)
    ]
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (combination / "measurement.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return path
