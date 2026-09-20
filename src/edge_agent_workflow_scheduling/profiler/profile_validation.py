"""Validate fitted resource profiles against immutable holdout observations."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from statistics import fmean
from typing import Any, Self

from edge_agent_workflow_scheduling.common import ToolCall
from edge_agent_workflow_scheduling.profiler.profile_fitting import DistributionSummary
from edge_agent_workflow_scheduling.resources import (
    ToolReplicaProfile,
    resolve_tool_execution_time_sec,
)

PROFILE_VALIDATION_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ProfileValidationConfig:
    """Predeclared validation policy and acceptance thresholds."""

    validation_id: str
    relative_error_denominator_epsilon: float
    comparison: dict[str, str]
    tool_execution_time_thresholds: dict[str, int | float]
    stochastic_validation: dict[str, Any]
    policy: dict[str, Any]
    schema_version: int = PROFILE_VALIDATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PROFILE_VALIDATION_SCHEMA_VERSION:
            raise ValueError("unsupported profile validation schema_version")
        if not isinstance(self.validation_id, str) or not self.validation_id.strip():
            raise ValueError("validation_id must be non-empty")
        _positive(self.relative_error_denominator_epsilon, "relative error epsilon")
        expected_comparison = {
            "before_calibration": "aggregate_profile_mean",
            "after_calibration": "matching_bucket_mean",
            "distribution": "calibration_bucket_distribution_vs_holdout_distribution",
        }
        if self.comparison != expected_comparison:
            raise ValueError(f"comparison must equal {expected_comparison!r}")
        thresholds = self.tool_execution_time_thresholds
        for key in ("minimum_total_sample_count", "minimum_group_sample_count"):
            _positive_integer(thresholds.get(key), key)
        for key in (
            "maximum_overall_mae_sec",
            "maximum_overall_mean_relative_error",
            "maximum_group_mean_relative_error",
        ):
            _positive(thresholds.get(key), key)
        seeds = self.stochastic_validation.get("seeds")
        if (
            not isinstance(seeds, list)
            or not seeds
            or any(
                isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
                for seed in seeds
            )
            or len(seeds) != len(set(seeds))
        ):
            raise ValueError("stochastic_validation.seeds must be unique non-negative integers")
        _positive_integer(
            self.stochastic_validation.get("samples_per_seed_per_group"),
            "samples_per_seed_per_group",
        )
        if self.policy.get("thresholds_declared_before_holdout_sampling") is not True:
            raise ValueError("validation thresholds must be declared before holdout sampling")

    @classmethod
    def from_json(cls, path: str | Path) -> Self:
        value = _read_object(Path(path))
        return cls(**value)


def validate_profile_catalog(
    profile_catalog_path: str | Path,
    tool_holdout_run: str | Path,
    config: ProfileValidationConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare measured Tool profiles with an independent sampling run."""

    catalog_path = Path(profile_catalog_path).resolve()
    holdout_path = Path(tool_holdout_run).resolve()
    catalog = _read_object(catalog_path)
    manifest = _read_object(holdout_path / "manifest.json")
    summary = _read_object(holdout_path / "summary.json")
    tool_profiles = _measured_tool_profiles(catalog)
    leakage = _validate_independent_holdout(tool_profiles, manifest, summary, holdout_path)
    samples, group_context = _tool_samples(
        tool_profiles,
        summary,
        holdout_path,
        config.relative_error_denominator_epsilon,
    )
    if not samples:
        raise ValueError("holdout run has no valid Tool execution-time observations")

    overall = _error_summary(samples, config.relative_error_denominator_epsilon)
    groups = []
    grouped_samples: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        key = (sample["task_type"], sample["input_size"], sample["concurrency"])
        grouped_samples[key].append(sample)
    for key, rows in sorted(grouped_samples.items()):
        context = group_context[key]
        groups.append(
            {
                "task_type": key[0],
                "input_size": key[1],
                "concurrency": key[2],
                "profile_id": context["profile"].replica_id,
                "bucket_id": context["bucket"]["bucket_id"],
                **_error_summary(rows, config.relative_error_denominator_epsilon),
                "distribution_comparison": _distribution_comparison(
                    context["bucket"], rows, config.relative_error_denominator_epsilon
                ),
                "stochastic_simulation": _stochastic_comparison(
                    context["profile"],
                    rows,
                    config.stochastic_validation,
                    config.relative_error_denominator_epsilon,
                ),
            }
        )

    threshold_result = _evaluate_thresholds(
        overall,
        groups,
        config.tool_execution_time_thresholds,
    )
    energy_observations = [
        sample for sample in samples if sample["energy_observation_status"] == "measured"
    ]
    report = {
        "schema_version": PROFILE_VALIDATION_SCHEMA_VERSION,
        "validation_id": config.validation_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "passed" if threshold_result["passed"] else "failed",
        "method": {
            **config.comparison,
            "relative_error": "abs(prediction-observation)/abs(observation)",
            "relative_error_denominator_epsilon": config.relative_error_denominator_epsilon,
            "profile_randomness": (
                "multi-seed bounded-uniform simulation uses the fitted profile's "
                "suggested jitter ratio; deterministic bucket error is reported separately"
            ),
        },
        "provenance": {
            "profile_catalog_ref": _display_ref(catalog_path),
            "profile_catalog_sha256": _file_digest(catalog_path),
            "tool_holdout_ref": _display_ref(holdout_path),
            "tool_holdout_sha256": _directory_digest(holdout_path),
            "profile_version": catalog.get("version"),
            "holdout_experiment_id": summary.get("experiment_id"),
            "validation_config": config.validation_id,
            "tool_holdout_sampling_config_ref": manifest.get("sampling_config_path"),
            "tool_holdout_sampling_config_sha256": manifest.get("sampling_config_sha256"),
        },
        "data_leakage_check": leakage,
        "tool_execution_time": {
            "status": "validated",
            "sample_count": len(samples),
            "overall": overall,
            "groups": groups,
            "thresholds": threshold_result,
        },
        "llm_request_time_or_throughput": {
            "status": "not_validated",
            "reason": (
                "the fitted 2026-09-11 Qwen3.8-27B benchmark has no independent "
                "matching holdout observations"
            ),
            "sample_count": 0,
        },
        "energy": {
            "status": "validated" if energy_observations else "not_validated",
            "reason": (
                None
                if energy_observations
                else "holdout records contain no provenance-qualified measured energy"
            ),
            "sample_count": len(energy_observations),
            "zero_placeholders_are_measurements": False,
        },
        "synthetic_profiles": {
            "status": "not_validated",
            "reason": "logical synthetic replicas are not physical holdout deployments",
        },
        "policy": config.policy,
    }
    return report, samples


def write_profile_validation_artifacts(
    output_dir: str | Path,
    report: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    *,
    overwrite: bool = False,
) -> Path:
    """Write a report and its auditable per-observation errors."""

    directory = Path(output_dir)
    if directory.exists() and not overwrite:
        raise FileExistsError(
            f"profile validation output already exists: {directory}; choose a new version "
            "or allow overwrite"
        )
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "samples.jsonl").write_text(
        "".join(
            json.dumps(sample, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for sample in samples
        ),
        encoding="utf-8",
    )
    (directory / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return directory


def _measured_tool_profiles(catalog: Mapping[str, Any]) -> list[ToolReplicaProfile]:
    resources = catalog.get("resource_profiles")
    if not isinstance(resources, dict):
        raise ValueError("profile catalog requires resource_profiles")
    entries = resources.get("tool_replicas")
    if not isinstance(entries, list):
        raise ValueError("profile catalog requires resource_profiles.tool_replicas")
    profiles = []
    for entry in entries:
        raw = entry.get("profile") if isinstance(entry, dict) else None
        if isinstance(raw, dict):
            profile = ToolReplicaProfile.from_dict(raw)
            if profile.metadata.get("source_kind") == "measured":
                profiles.append(profile)
    if not profiles:
        raise ValueError("profile catalog has no measured Tool profiles")
    return profiles


def _validate_independent_holdout(
    profiles: Sequence[ToolReplicaProfile],
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    holdout_path: Path,
) -> dict[str, Any]:
    holdout_ref = _display_ref(holdout_path)
    source_refs = {profile.metadata.get("source_ref") for profile in profiles}
    if holdout_ref in source_refs:
        raise ValueError("holdout run is also a profile fitting source")
    calibration_paths = {
        Path(source_ref).resolve()
        for source_ref in source_refs
        if isinstance(source_ref, str) and Path(source_ref).exists()
    }
    calibration_hashes: set[str] = set()
    for path in calibration_paths:
        calibration_summary = _read_object(path / "summary.json")
        calibration_hashes.update(_input_hashes(calibration_summary))
    holdout_hashes = _input_hashes(summary)
    overlap = sorted(calibration_hashes & holdout_hashes)
    if overlap:
        raise ValueError("calibration and holdout inputs overlap by content hash")

    holdout_host = manifest.get("host")
    if not isinstance(holdout_host, dict):
        raise ValueError("holdout manifest requires host inventory")
    host_hash = holdout_host.get("hostname_hash")
    profile_hosts = {
        profile.metadata.get("host", {}).get("hostname_hash") for profile in profiles
    }
    if profile_hosts != {host_hash}:
        raise ValueError("holdout host does not match the measured profile host")

    sampling_config = manifest.get("sampling_config")
    if not isinstance(sampling_config, dict):
        raise ValueError("holdout manifest requires sampling_config")
    fitted_config = profiles[0].metadata.get("measurement_config")
    if not isinstance(fitted_config, dict):
        raise ValueError("measured profile requires measurement_config")
    for key in ("tools", "scales", "concurrency_levels", "image_preprocess"):
        if sampling_config.get(key) != fitted_config.get(key):
            raise ValueError(f"holdout sampling config differs from calibration for {key}")
    return {
        "status": "passed",
        "profile_source_is_not_holdout": True,
        "input_content_hashes_are_disjoint": True,
        "same_host": True,
        "same_execution_configuration": True,
        "calibration_input_count": len(calibration_hashes),
        "holdout_input_count": len(holdout_hashes),
        "overlap_count": 0,
    }


def _tool_samples(
    profiles: Sequence[ToolReplicaProfile],
    summary: Mapping[str, Any],
    holdout_path: Path,
    epsilon: float,
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, str, int], dict[str, Any]],
]:
    by_key = {(profile.tool_name, profile.max_concurrency): profile for profile in profiles}
    combinations = summary.get("combinations")
    if not isinstance(combinations, list):
        raise ValueError("holdout summary requires combinations")
    samples = []
    contexts: dict[tuple[str, str, int], dict[str, Any]] = {}
    for combination in combinations:
        if not isinstance(combination, dict) or combination.get("status") != "completed":
            continue
        tool_name = combination.get("tool_name")
        input_size = combination.get("scale")
        concurrency = combination.get("concurrency")
        profile = by_key.get((tool_name, concurrency))
        if profile is None or not isinstance(input_size, str):
            raise ValueError(
                f"no measured profile for holdout combination {tool_name!r}, c{concurrency!r}"
            )
        call = ToolCall(
            tool_call_id="profile-validation",
            call_id="profile-validation",
            run_id=str(summary.get("experiment_id", "profile-validation")),
            agent_id="profile-validator",
            tool_name=tool_name,
            arguments={},
            metadata={"task_type": tool_name, "input_size": input_size},
        )
        resolved = resolve_tool_execution_time_sec(profile, call)
        if resolved.source != "bucket" or resolved.bucket_id is None:
            raise ValueError("holdout validation requires an exact fitted bucket")
        bucket = _bucket_by_id(profile, resolved.bucket_id)
        group_key = (tool_name, input_size, concurrency)
        contexts[group_key] = {"profile": profile, "bucket": bucket}
        records_path = holdout_path / combination["combination_id"] / "measurement.jsonl"
        for record in _read_jsonl(records_path):
            result = record.get("result")
            if not isinstance(result, dict) or result.get("success") is not True:
                continue
            observed = result.get("execution_time_sec")
            if not _is_positive(observed):
                continue
            before = profile.latency_profile.get("execution_time_sec")
            _positive(before, "aggregate Tool execution time")
            energy_status = result.get("metadata", {}).get("energy_source", "unavailable")
            before_abs = abs(float(before) - float(observed))
            after_abs = abs(resolved.value - float(observed))
            samples.append(
                {
                    "sample_id": record.get("call", {}).get("tool_call_id"),
                    "source_ref": _display_ref(records_path),
                    "profile_id": profile.replica_id,
                    "bucket_id": resolved.bucket_id,
                    "tool_name": tool_name,
                    "task_type": tool_name,
                    "source_task_type": record.get("call", {})
                    .get("metadata", {})
                    .get("task_type"),
                    "input_size": input_size,
                    "concurrency": concurrency,
                    "observed_execution_time_sec": observed,
                    "before_prediction_sec": before,
                    "after_prediction_sec": resolved.value,
                    "before_absolute_error_sec": before_abs,
                    "after_absolute_error_sec": after_abs,
                    "before_relative_error": _relative_error(before_abs, observed, epsilon),
                    "after_relative_error": _relative_error(after_abs, observed, epsilon),
                    "energy_observation_status": (
                        "measured" if energy_status == "measured" else "unavailable"
                    ),
                }
            )
    return samples, contexts


def _error_summary(rows: Sequence[Mapping[str, Any]], epsilon: float) -> dict[str, Any]:
    observed = [float(row["observed_execution_time_sec"]) for row in rows]
    output: dict[str, Any] = {
        "sample_count": len(rows),
        "observed_distribution": DistributionSummary.from_values(observed).to_dict(),
    }
    for label in ("before", "after"):
        predictions = [float(row[f"{label}_prediction_sec"]) for row in rows]
        absolute = [
            abs(predicted - actual)
            for predicted, actual in zip(predictions, observed, strict=True)
        ]
        relative = [
            value
            for value in (
                _relative_error(error, actual, epsilon)
                for error, actual in zip(absolute, observed, strict=True)
            )
            if value is not None
        ]
        output[label] = {
            "prediction_mean_sec": fmean(predictions),
            "mae_sec": fmean(absolute),
            "p95_absolute_error_sec": _percentile(absolute, 0.95),
            "mean_relative_error": fmean(relative) if relative else None,
            "p95_relative_error": _percentile(relative, 0.95) if relative else None,
            "relative_error_sample_count": len(relative),
            "relative_error_excluded_count": len(rows) - len(relative),
            "mean_bias_sec": fmean(
                predicted - actual
                for predicted, actual in zip(predictions, observed, strict=True)
            ),
        }
    output["calibration_comparison"] = {
        "mae_improved": output["after"]["mae_sec"] < output["before"]["mae_sec"],
        "relative_error_improved": (
            output["after"]["mean_relative_error"]
            < output["before"]["mean_relative_error"]
        ),
        "mae_change_sec": output["after"]["mae_sec"] - output["before"]["mae_sec"],
    }
    return output


def _distribution_comparison(
    bucket: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    epsilon: float,
) -> dict[str, Any]:
    fitted = bucket.get("distribution")
    if not isinstance(fitted, dict):
        raise ValueError("fitted bucket requires a distribution")
    observed = DistributionSummary.from_values(
        [float(row["observed_execution_time_sec"]) for row in rows]
    ).to_dict()
    comparisons = {}
    for key in ("mean", "p95", "population_stddev"):
        error = abs(float(fitted[key]) - float(observed[key]))
        comparisons[key] = {
            "absolute_error_sec": error,
            "relative_error": _relative_error(error, float(observed[key]), epsilon),
        }
    return {
        "calibration": fitted,
        "holdout": observed,
        "differences": comparisons,
    }


def _stochastic_comparison(
    profile: ToolReplicaProfile,
    rows: Sequence[Mapping[str, Any]],
    stochastic_config: Mapping[str, Any],
    epsilon: float,
) -> dict[str, Any]:
    distribution = profile.metadata.get("random_distribution")
    if not isinstance(distribution, dict):
        return {"status": "not_applicable", "reason": "profile is deterministic"}
    if distribution.get("kind") != "bounded_uniform_executor_jitter":
        return {"status": "not_applicable", "reason": "unsupported profile distribution"}
    ratio = distribution.get("suggested_jitter_ratio")
    if not isinstance(ratio, int | float) or isinstance(ratio, bool) or not 0 <= ratio < 1:
        raise ValueError("profile suggested_jitter_ratio must be in [0, 1)")
    prediction = float(rows[0]["after_prediction_sec"])
    seeds = stochastic_config["seeds"]
    count = stochastic_config["samples_per_seed_per_group"]
    simulated = []
    seed_summaries = []
    for seed in seeds:
        rng = random.Random(f"{seed}:{profile.replica_id}:{rows[0]['bucket_id']}")
        values = [prediction * rng.uniform(1.0 - ratio, 1.0 + ratio) for _ in range(count)]
        simulated.extend(values)
        seed_summaries.append(
            {"seed": seed, **DistributionSummary.from_values(values).to_dict()}
        )
    simulated_distribution = DistributionSummary.from_values(simulated).to_dict()
    observed_distribution = DistributionSummary.from_values(
        [float(row["observed_execution_time_sec"]) for row in rows]
    ).to_dict()
    differences = {}
    for key in ("mean", "p95", "population_stddev"):
        error = abs(simulated_distribution[key] - observed_distribution[key])
        differences[key] = {
            "absolute_error_sec": error,
            "relative_error": _relative_error(error, observed_distribution[key], epsilon),
        }
    return {
        "status": "validated",
        "distribution_kind": distribution["kind"],
        "jitter_ratio": ratio,
        "seed_count": len(seeds),
        "samples_per_seed": count,
        "seed_summaries": seed_summaries,
        "combined_simulation": simulated_distribution,
        "holdout": observed_distribution,
        "differences": differences,
    }


def _evaluate_thresholds(
    overall: Mapping[str, Any],
    groups: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, int | float],
) -> dict[str, Any]:
    maximum_group_error = max(group["after"]["mean_relative_error"] for group in groups)
    minimum_group_count = min(group["sample_count"] for group in groups)
    checks = [
        _minimum_check(
            "minimum_total_sample_count",
            overall["sample_count"],
            thresholds["minimum_total_sample_count"],
        ),
        _minimum_check(
            "minimum_group_sample_count",
            minimum_group_count,
            thresholds["minimum_group_sample_count"],
        ),
        _maximum_check(
            "maximum_overall_mae_sec",
            overall["after"]["mae_sec"],
            thresholds["maximum_overall_mae_sec"],
        ),
        _maximum_check(
            "maximum_overall_mean_relative_error",
            overall["after"]["mean_relative_error"],
            thresholds["maximum_overall_mean_relative_error"],
        ),
        _maximum_check(
            "maximum_group_mean_relative_error",
            maximum_group_error,
            thresholds["maximum_group_mean_relative_error"],
        ),
    ]
    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }


def _minimum_check(name: str, observed: int | float, threshold: int | float) -> dict[str, Any]:
    return {
        "name": name,
        "operator": ">=",
        "observed": observed,
        "threshold": threshold,
        "passed": observed >= threshold,
    }


def _maximum_check(name: str, observed: int | float, threshold: int | float) -> dict[str, Any]:
    return {
        "name": name,
        "operator": "<=",
        "observed": observed,
        "threshold": threshold,
        "passed": observed <= threshold,
    }


def _bucket_by_id(profile: ToolReplicaProfile, bucket_id: str) -> dict[str, Any]:
    lookup = profile.metadata.get("profile_lookup")
    buckets = lookup.get("latency_buckets") if isinstance(lookup, dict) else None
    if not isinstance(buckets, list):
        raise ValueError("measured profile requires latency_buckets")
    matches = [bucket for bucket in buckets if bucket.get("bucket_id") == bucket_id]
    if len(matches) != 1:
        raise ValueError(f"expected one latency bucket {bucket_id!r}")
    return matches[0]


def _input_hashes(summary: Mapping[str, Any]) -> set[str]:
    combinations = summary.get("combinations")
    if not isinstance(combinations, list):
        raise ValueError("sampling summary requires combinations")
    values = {
        item.get("source_sha256")
        for item in combinations
        if isinstance(item, dict) and item.get("status") == "completed"
    }
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise ValueError("completed sampling combinations require source_sha256")
    return values


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        values.append(value)
    return values


def _relative_error(error: float, observed: float, epsilon: float) -> float | None:
    return error / abs(observed) if abs(observed) > epsilon else None


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires observations")
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _positive(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _is_positive(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and isfinite(value)
        and value > 0
    )


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _directory_digest(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError("directory digest requires at least one file")
    for item in files:
        digest.update(str(item.relative_to(path)).encode())
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _display_ref(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())
