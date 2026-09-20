"""Fit traceable bucketed resource profiles from saved sampling artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from statistics import fmean, median, pstdev
from typing import Any, Literal

from edge_agent_workflow_scheduling.resources import (
    LLMInstanceProfile,
    LLMInstanceState,
    ToolReplicaProfile,
    ToolReplicaState,
)

FallbackPolicy = Literal["error", "aggregate"]
PROFILE_CATALOG_SCHEMA_VERSION = 1
FIT_METHOD = "bucketed arithmetic mean with interpolated percentiles"


@dataclass(frozen=True, slots=True)
class DistributionSummary:
    sample_count: int
    mean: float
    median: float
    p95: float
    p99: float
    minimum: float
    maximum: float
    population_stddev: float

    @classmethod
    def from_values(cls, values: Sequence[float]) -> DistributionSummary:
        checked = [_positive(value, "observation") for value in values]
        if not checked:
            raise ValueError("distribution requires at least one positive observation")
        return cls(
            sample_count=len(checked),
            mean=fmean(checked),
            median=median(checked),
            p95=_percentile(checked, 0.95),
            p99=_percentile(checked, 0.99),
            minimum=min(checked),
            maximum=max(checked),
            population_stddev=pstdev(checked),
        )

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SyntheticEnergyConfig:
    version: str
    tool_joules_per_call: dict[str, float]
    llm_joules_per_token: dict[str, float]
    assumptions: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SyntheticEnergyConfig:
        version = value.get("version")
        assumptions = value.get("assumptions")
        if not isinstance(version, str) or not version.strip():
            raise ValueError("synthetic energy config requires a non-empty version")
        if not isinstance(assumptions, str) or not assumptions.strip():
            raise ValueError("synthetic energy config requires non-empty assumptions")
        tools = _positive_mapping(value.get("tool_joules_per_call", {}), "tool_joules_per_call")
        llms = _positive_mapping(value.get("llm_joules_per_token", {}), "llm_joules_per_token")
        return cls(version, tools, llms, assumptions)


def fit_profile_catalog(
    *,
    profile_version: str,
    tool_sampling_run: str | Path | Sequence[str | Path] | None = None,
    llm_benchmark: str | Path | None = None,
    fallback_policy: FallbackPolicy = "error",
    synthetic_tool_latency_multiplier: float = 1.75,
    synthetic_energy: SyntheticEnergyConfig | None = None,
) -> dict[str, Any]:
    """Fit a directly loadable resource snapshot catalog from immutable observations."""

    _non_empty(profile_version, "profile_version")
    if fallback_policy not in {"error", "aggregate"}:
        raise ValueError("fallback_policy must be 'error' or 'aggregate'")
    multiplier = _positive(
        synthetic_tool_latency_multiplier,
        "synthetic_tool_latency_multiplier",
    )
    tool_run_paths = _path_sequence(tool_sampling_run)
    if not tool_run_paths and llm_benchmark is None:
        raise ValueError("at least one Tool sampling run or LLM benchmark is required")

    tool_profiles: list[ToolReplicaProfile] = []
    llm_profiles: list[LLMInstanceProfile] = []
    sources: list[dict[str, Any]] = []
    for run_path in tool_run_paths:
        tool_profiles.extend(
            fit_tool_profiles(
                run_path,
                profile_version=profile_version,
                fallback_policy=fallback_policy,
                synthetic_latency_multiplier=multiplier,
                synthetic_energy=synthetic_energy,
            )
        )
        sources.append(_source_descriptor(run_path, "tool_sampling"))
    if llm_benchmark is not None:
        benchmark_path = Path(llm_benchmark).resolve()
        llm_profiles = fit_llm_profiles(
            benchmark_path,
            profile_version=profile_version,
            fallback_policy=fallback_policy,
        )
        sources.append(_source_descriptor(benchmark_path, "llm_benchmark"))

    _require_unique_profile_ids(tool_profiles, llm_profiles)
    captured_at = datetime.now(UTC).isoformat()
    return {
        "schema_version": PROFILE_CATALOG_SCHEMA_VERSION,
        "version": profile_version,
        "generated_at": captured_at,
        "fit_method": FIT_METHOD,
        "fallback_policy": fallback_policy,
        "sources": sources,
        "energy_policy": {
            "missing_observations": "leave energy_profile empty",
            "zero_placeholder_is_measurement": False,
            "dependent_policy_behavior": "energy-aware configuration validation must fail",
            "synthetic_energy_version": synthetic_energy.version if synthetic_energy else None,
        },
        "resource_profiles": {
            "llm_instances": [
                {
                    "profile": profile.to_dict(),
                    "state": LLMInstanceState(
                        llm_id=profile.llm_id,
                        updated_at=captured_at,
                    ).to_dict(),
                }
                for profile in llm_profiles
            ],
            "tool_replicas": [
                {
                    "profile": profile.to_dict(),
                    "state": ToolReplicaState(
                        replica_id=profile.replica_id,
                        updated_at=captured_at,
                    ).to_dict(),
                }
                for profile in tool_profiles
            ],
        },
    }


def fit_tool_profiles(
    run_path: str | Path,
    *,
    profile_version: str,
    fallback_policy: FallbackPolicy = "error",
    synthetic_latency_multiplier: float = 1.75,
    synthetic_energy: SyntheticEnergyConfig | None = None,
) -> list[ToolReplicaProfile]:
    """Fit one measured profile per Tool/concurrency and one explicit synthetic replica."""

    run_path = Path(run_path).resolve()
    summary_path = run_path / "summary.json"
    manifest_path = run_path / "manifest.json"
    summary = _read_object(summary_path)
    manifest = _read_object(manifest_path)
    host = manifest.get("host", {})
    host_hash = host.get("hostname_hash") if isinstance(host, dict) else None
    host_key = host_hash[:8] if isinstance(host_hash, str) and host_hash else "unknown-host"
    combinations = summary.get("combinations")
    if not isinstance(combinations, list):
        raise ValueError("Tool sampling summary requires combinations")
    source_digest = _tree_digest(
        [summary_path, manifest_path]
        + [
            run_path / item["combination_id"] / "measurement.jsonl"
            for item in combinations
            if isinstance(item, dict) and item.get("status") == "completed"
        ]
    )

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for combination in combinations:
        if not isinstance(combination, dict) or combination.get("status") != "completed":
            continue
        tool_name = combination.get("tool_name")
        scale = combination.get("scale")
        concurrency = combination.get("concurrency")
        if not isinstance(tool_name, str) or not isinstance(scale, str):
            raise ValueError("completed Tool combination requires tool_name and scale")
        if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
            raise ValueError("completed Tool combination requires positive concurrency")
        records_path = run_path / combination["combination_id"] / "measurement.jsonl"
        records = _read_jsonl(records_path)
        values = [
            record["result"]["execution_time_sec"]
            for record in records
            if record.get("result", {}).get("success") is True
            and _is_positive(record.get("result", {}).get("execution_time_sec"))
        ]
        if not values:
            raise ValueError(f"{combination['combination_id']} has no successful timed samples")
        grouped[(tool_name, concurrency)].append(
            {
                "combination": combination,
                "scale": scale,
                "records_path": records_path,
                "record_count": len(records),
                "values": values,
                "excluded_count": len(records) - len(values),
            }
        )

    measured: list[ToolReplicaProfile] = []
    for (tool_name, concurrency), groups in sorted(grouped.items()):
        base_raw = groups[0]["combination"].get("resource_profile")
        if not isinstance(base_raw, dict):
            raise ValueError("Tool combination requires resource_profile")
        base = ToolReplicaProfile.from_dict(base_raw)
        all_values = [value for group in groups for value in group["values"]]
        aggregate = DistributionSummary.from_values(all_values)
        buckets = []
        for group in sorted(groups, key=lambda item: item["scale"]):
            distribution = DistributionSummary.from_values(group["values"])
            buckets.append(
                {
                    "bucket_id": f"{tool_name}-{group['scale']}-c{concurrency}",
                    "selectors": {
                        "input_size": group["scale"],
                        "concurrency": concurrency,
                    },
                    "value": distribution.mean,
                    "unit": "seconds_per_call",
                    "distribution": distribution.to_dict(),
                    "included_count": distribution.sample_count,
                    "excluded_count": group["excluded_count"],
                    "exclusion_rule": "failed calls and missing/non-positive timing",
                    "source_ref": _display_ref(group["records_path"]),
                }
            )

        profile_id = f"{tool_name}-linux-{host_key}-c{concurrency}-{_slug(profile_version)}"
        energy_profile, energy_provenance = _tool_measured_energy(groups, tool_name)
        profile = replace(
            base,
            replica_id=profile_id,
            node_id=f"linux-{host_key}",
            platform="linux",
            executor_type="profile",
            max_concurrency=concurrency,
            latency_profile={"execution_time_sec": aggregate.mean},
            energy_profile=energy_profile,
            deployment_config={},
            secret_env_vars=[],
            metadata={
                "profile_id": profile_id,
                "profile_version": profile_version,
                "source_kind": "measured",
                "source_ref": _display_ref(run_path),
                "source_sha256": source_digest,
                "fit_method": FIT_METHOD,
                "sample_count": aggregate.sample_count,
                "excluded_count": sum(group["excluded_count"] for group in groups),
                "scope": {
                    "input_sizes": sorted(group["scale"] for group in groups),
                    "concurrency": concurrency,
                    "tool_name": tool_name,
                },
                "host": deepcopy(host),
                "measurement_config": deepcopy(manifest.get("sampling_config", {})),
                "physical_deployment": {
                    "kind": "single_local_host",
                    "node_id": f"linux-{host_key}",
                    "logical_replica_count": 1,
                },
                "energy_provenance": energy_provenance,
                "random_distribution": {
                    "kind": "bounded_uniform_executor_jitter",
                    "suggested_jitter_ratio": _jitter_ratio(aggregate),
                    "calibration_statistic": "population_stddev/mean capped below 1",
                },
                "profile_lookup": {
                    "fallback_policy": fallback_policy,
                    "latency_buckets": buckets,
                    "energy_buckets": [],
                },
            },
        )
        measured.append(profile)

    synthetic = _synthetic_tool_profiles(
        measured,
        profile_version=profile_version,
        latency_multiplier=synthetic_latency_multiplier,
        synthetic_energy=synthetic_energy,
    )
    return [*measured, *synthetic]


def fit_llm_profiles(
    benchmark_path: str | Path,
    *,
    profile_version: str,
    fallback_policy: FallbackPolicy = "error",
) -> list[LLMInstanceProfile]:
    """Fit aggregate and task/size bucket token rates from a benchmark interchange file."""

    benchmark_path = Path(benchmark_path).resolve()
    data = _read_object(benchmark_path)
    raw_profiles = data.get("llm_instances")
    rows = data.get("samples")
    if not isinstance(raw_profiles, list) or not isinstance(rows, list):
        raise ValueError("LLM benchmark requires llm_instances and samples lists")
    profiles = {
        profile.llm_id: profile for profile in map(LLMInstanceProfile.from_dict, raw_profiles)
    }
    source_digest = _file_digest(benchmark_path)
    exported: list[LLMInstanceProfile] = []
    for llm_id, base in profiles.items():
        selected = [row for row in rows if _valid_llm_row(row, llm_id)]
        if not selected:
            continue
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            groups[(row["task_type"], row["input_size"])].append(row)
        buckets = []
        for (task_type, input_size), group in sorted(groups.items()):
            total_tokens = sum(_row_total_tokens(row) for row in group)
            total_time = sum(float(row["request_time_sec"]) for row in group)
            rates = [_row_total_tokens(row) / float(row["request_time_sec"]) for row in group]
            distribution = DistributionSummary.from_values(rates)
            buckets.append(
                {
                    "bucket_id": f"{llm_id}-{task_type}-{input_size}-c1",
                    "selectors": {
                        "task_type": task_type,
                        "input_size": input_size,
                        "concurrency": 1,
                    },
                    "value": total_tokens / total_time,
                    "unit": "total_tokens_per_client_request_second",
                    "scope": {
                        "input_token_range": [
                            min(row["input_tokens"] for row in group),
                            max(row["input_tokens"] for row in group),
                        ],
                        "output_token_range": [
                            min(row["output_tokens"] for row in group),
                            max(row["output_tokens"] for row in group),
                        ],
                    },
                    "distribution": distribution.to_dict(),
                    "included_count": len(group),
                    "excluded_count": 0,
                    "exclusion_rule": "failed or missing/non-positive timing or token counts",
                }
            )
        total_tokens = sum(_row_total_tokens(row) for row in selected)
        total_time = sum(float(row["request_time_sec"]) for row in selected)
        energy_profile, energy_provenance = _llm_measured_energy(
            selected,
            data.get("provenance"),
        )
        source_kinds = {row.get("source_kind", "benchmark") for row in selected}
        if any(not isinstance(item, str) or not item.strip() for item in source_kinds):
            raise ValueError("LLM sample source_kind values must be non-empty strings")
        if len(source_kinds) != 1:
            raise ValueError("one fitted LLM profile cannot mix source kinds")
        source_kind = next(iter(source_kinds))
        profile_id = f"{llm_id}-{_slug(profile_version)}"
        exported.append(
            replace(
                base,
                llm_id=profile_id,
                executor_type="profile",
                max_concurrency=1,
                token_profile={"tokens_per_sec": total_tokens / total_time},
                energy_profile=energy_profile,
                deployment_config={},
                secret_env_vars=[],
                metadata={
                    **deepcopy(base.metadata),
                    "profile_id": profile_id,
                    "profile_version": profile_version,
                    "source_kind": source_kind,
                    "source_ref": _display_ref(benchmark_path),
                    "source_sha256": source_digest,
                    "fit_method": FIT_METHOD,
                    "rate_definition": (
                        "sum(input_tokens+output_tokens)/sum(client_request_seconds)"
                    ),
                    "timing_scope": "client_request_including_network_and_server_wait",
                    "sample_count": len(selected),
                    "excluded_count": sum(row.get("llm_id") == llm_id for row in rows)
                    - len(selected),
                    "scope": {
                        "task_types": sorted({row["task_type"] for row in selected}),
                        "input_sizes": sorted({row["input_size"] for row in selected}),
                        "concurrency": 1,
                        "input_token_range": [
                            min(row["input_tokens"] for row in selected),
                            max(row["input_tokens"] for row in selected),
                        ],
                        "output_token_range": [
                            min(row["output_tokens"] for row in selected),
                            max(row["output_tokens"] for row in selected),
                        ],
                    },
                    "energy_provenance": energy_provenance,
                    "profile_lookup": {
                        "fallback_policy": fallback_policy,
                        "token_rate_buckets": buckets,
                        "energy_buckets": [],
                    },
                },
            )
        )
    return exported


def write_profile_catalog(
    catalog: Mapping[str, Any],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    output = Path(path)
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"profile catalog already exists: {output}; choose a new version or allow overwrite"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def _synthetic_tool_profiles(
    measured: Sequence[ToolReplicaProfile],
    *,
    profile_version: str,
    latency_multiplier: float,
    synthetic_energy: SyntheticEnergyConfig | None,
) -> list[ToolReplicaProfile]:
    by_tool: dict[str, ToolReplicaProfile] = {}
    for profile in measured:
        current = by_tool.get(profile.tool_name)
        if current is None or profile.max_concurrency < current.max_concurrency:
            by_tool[profile.tool_name] = profile
    output = []
    for tool_name, source in sorted(by_tool.items()):
        metadata = deepcopy(source.metadata)
        lookup = deepcopy(metadata["profile_lookup"])
        for bucket in lookup["latency_buckets"]:
            bucket["value"] *= latency_multiplier
            bucket["distribution"] = {
                key: value * latency_multiplier if key != "sample_count" else value
                for key, value in bucket["distribution"].items()
            }
            bucket["bucket_id"] += "-synthetic"
        energy_profile: dict[str, float] = {}
        energy_provenance: dict[str, Any] = {
            "status": "unavailable",
            "reason": "no synthetic energy configuration was supplied",
        }
        if synthetic_energy and tool_name in synthetic_energy.tool_joules_per_call:
            energy_profile = {"joules_per_call": synthetic_energy.tool_joules_per_call[tool_name]}
            energy_provenance = {
                "status": "synthetic",
                "source_kind": "synthetic",
                "config_version": synthetic_energy.version,
                "derivation": "configured constant joules_per_call",
                "assumptions": synthetic_energy.assumptions,
                "unit": "joules_per_call",
            }
        source_host = source.metadata.get("host", {}).get("hostname_hash", "unknown-host")
        host_key = source_host[:8] if isinstance(source_host, str) else "unknown-host"
        profile_id = f"{tool_name}-logical-{host_key}-synthetic-{_slug(profile_version)}"
        metadata.update(
            {
                "profile_id": profile_id,
                "source_kind": "synthetic",
                "derived_from_profile_id": source.replica_id,
                "fit_method": "measured bucket means multiplied by configured factor",
                "latency_multiplier": latency_multiplier,
                "energy_provenance": energy_provenance,
                "physical_deployment": {
                    "kind": "logical_simulation",
                    "is_physical_device": False,
                },
                "profile_lookup": lookup,
            }
        )
        output.append(
            replace(
                source,
                replica_id=profile_id,
                node_id="logical-synthetic",
                platform="simulated",
                latency_profile={
                    "execution_time_sec": (
                        source.latency_profile["execution_time_sec"] * latency_multiplier
                    )
                },
                energy_profile=energy_profile,
                metadata=metadata,
            )
        )
    return output


def _tool_measured_energy(
    groups: Sequence[dict[str, Any]],
    tool_name: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    observed: list[float] = []
    sources: set[str] = set()
    measurement_specs: list[dict[str, Any]] = []
    for group in groups:
        for record in _read_jsonl(group["records_path"]):
            result = record.get("result", {})
            metadata = result.get("metadata", {})
            source = metadata.get("energy_source")
            value = result.get("energy_joules")
            if source in {"measured", "proxy", "estimated"} and _is_positive(value):
                measurement = metadata.get("energy_measurement")
                _validate_energy_measurement(measurement, source)
                observed.append(float(value))
                sources.add(source)
                measurement_specs.append(measurement)
    if not observed:
        return {}, {
            "status": "unavailable",
            "reason": f"no provenance-qualified energy observations for {tool_name}",
            "zero_placeholder_is_measurement": False,
        }
    if len(sources) != 1:
        raise ValueError("Tool energy observations mix incompatible source kinds")
    source = next(iter(sources))
    if any(spec != measurement_specs[0] for spec in measurement_specs[1:]):
        raise ValueError("Tool energy observations mix incompatible measurement settings")
    return {"joules_per_call": fmean(observed)}, {
        "status": source,
        "source_kind": source,
        **deepcopy(measurement_specs[0]),
        "sample_count": len(observed),
        "derivation": "arithmetic mean of per-call joule observations",
        "unit": "joules_per_call",
    }


def _llm_measured_energy(
    rows: Sequence[dict[str, Any]],
    provenance: Any,
) -> tuple[dict[str, float], dict[str, Any]]:
    measurement = provenance.get("energy_measurement") if isinstance(provenance, dict) else None
    observed = [row for row in rows if _is_positive(row.get("energy_joules"))]
    if not observed or not isinstance(measurement, dict):
        return {}, {
            "status": "unavailable",
            "reason": "benchmark has no provenance-qualified energy observations",
            "zero_placeholder_is_measurement": False,
        }
    source_kind = measurement["source_kind"]
    _validate_energy_measurement(measurement, source_kind)
    total_energy = sum(float(row["energy_joules"]) for row in observed)
    total_tokens = sum(_row_total_tokens(row) for row in observed)
    return {"joules_per_token": total_energy / total_tokens}, {
        "status": source_kind,
        **deepcopy(measurement),
        "sample_count": len(observed),
        "derivation": "sum(per-request joules)/sum(input_tokens+output_tokens)",
        "unit": "joules_per_token",
    }


def _valid_llm_row(row: Any, llm_id: str) -> bool:
    return (
        isinstance(row, dict)
        and row.get("llm_id") == llm_id
        and row.get("success") is True
        and isinstance(row.get("task_type"), str)
        and isinstance(row.get("input_size"), str)
        and isinstance(row.get("input_tokens"), int)
        and not isinstance(row.get("input_tokens"), bool)
        and isinstance(row.get("output_tokens"), int)
        and not isinstance(row.get("output_tokens"), bool)
        and row["input_tokens"] + row["output_tokens"] > 0
        and _is_positive(row.get("request_time_sec"))
    )


def _row_total_tokens(row: Mapping[str, Any]) -> int:
    return int(row["input_tokens"]) + int(row["output_tokens"])


def _source_descriptor(path: Path, source_kind: str) -> dict[str, str]:
    return {
        "source_kind": source_kind,
        "source_ref": _display_ref(path),
        "source_sha256": _directory_digest(path) if path.is_dir() else _file_digest(path),
    }


def _path_sequence(
    value: str | Path | Sequence[str | Path] | None,
) -> list[Path]:
    if value is None:
        return []
    values = [value] if isinstance(value, (str, Path)) else list(value)
    paths = [Path(item).resolve() for item in values]
    if len(set(paths)) != len(paths):
        raise ValueError("Tool sampling run paths must not contain duplicates")
    return paths


def _require_unique_profile_ids(
    tools: Sequence[ToolReplicaProfile],
    llms: Sequence[LLMInstanceProfile],
) -> None:
    tool_ids = [profile.replica_id for profile in tools]
    llm_ids = [profile.llm_id for profile in llms]
    if len(tool_ids) != len(set(tool_ids)):
        raise ValueError("fitted Tool replica IDs must be unique across sampling runs")
    if len(llm_ids) != len(set(llm_ids)):
        raise ValueError("fitted LLM IDs must be unique")


def _tree_digest(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in paths if path.is_file())
    if not files:
        raise ValueError("source digest requires at least one file")
    for path in files:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _directory_digest(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError("source digest requires at least one file")
    for item in files:
        digest.update(str(item.relative_to(path)).encode())
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display_ref(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} must contain one JSON object per line")
    return rows


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _jitter_ratio(distribution: DistributionSummary) -> float:
    return min(0.99, distribution.population_stddev / distribution.mean)


def _positive_mapping(value: Any, name: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return {str(key): _positive(item, f"{name}.{key}") for key, item in value.items()}


def _validate_energy_measurement(value: Any, source_kind: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("energy observations require metadata.energy_measurement")
    if source_kind not in {"measured", "proxy", "estimated"}:
        raise ValueError("energy source_kind must be measured, proxy, or estimated")
    required = (
        "sampling_interval_sec",
        "integration_method",
        "idle_baseline",
        "allocation_method",
    )
    if any(key not in value for key in required):
        raise ValueError(f"energy_measurement requires {required!r}")
    _positive(value["sampling_interval_sec"], "energy sampling_interval_sec")
    for key in ("integration_method", "idle_baseline", "allocation_method"):
        _non_empty(value[key], f"energy_measurement.{key}")
    if source_kind in {"proxy", "estimated"}:
        for key in ("conversion", "assumptions"):
            _non_empty(value.get(key), f"energy_measurement.{key}")


def _positive(value: Any, name: str) -> float:
    if not _is_positive(value):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _is_positive(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and isfinite(value)
        and value > 0
    )


def _non_empty(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
