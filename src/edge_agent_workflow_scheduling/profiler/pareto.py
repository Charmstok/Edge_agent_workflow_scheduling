"""Weight scans and small pairwise Pareto-frontier experiments."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from math import isfinite
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

from edge_agent_workflow_scheduling.profiler.baseline_experiment import (
    BaselineExperimentResult,
    BaselineRunResult,
    run_baseline_experiment,
)
from edge_agent_workflow_scheduling.profiler.models import TraceBundle
from edge_agent_workflow_scheduling.profiler.replay import load_trace_bundle
from edge_agent_workflow_scheduling.scheduler.objectives import (
    ObjectiveNormalization,
    ObjectiveWeights,
)

REFERENCE_POLICIES = (
    "random",
    "round_robin",
    "least_queue",
    "earliest_finish_time",
    "quality_aware",
    "energy_aware",
    "quality_constrained_earliest_finish_time",
)
MINIMIZED_OBJECTIVES = (
    "latency_sec",
    "energy_joules",
    "deadline_miss",
    "load_imbalance",
)
MAXIMIZED_OBJECTIVES = ("quality",)
OBJECTIVE_NAMES = (*MINIMIZED_OBJECTIVES, *MAXIMIZED_OBJECTIVES)


@dataclass(frozen=True, slots=True)
class WeightConfiguration:
    """One named scalarization used by the weighted-objective policy."""

    name: str
    weights: ObjectiveWeights

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("weight configuration name must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "weights": asdict(self.weights)}


@dataclass(frozen=True, slots=True)
class WeightSet:
    """A versioned, explicit collection of weight configurations."""

    version: str
    configurations: tuple[WeightConfiguration, ...]

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("weight set version must be non-empty")
        if not self.configurations:
            raise ValueError("weight set must contain at least one configuration")
        names = [configuration.name for configuration in self.configurations]
        if len(names) != len(set(names)):
            raise ValueError("weight configuration names must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "configurations": [configuration.to_dict() for configuration in self.configurations],
        }


@dataclass(frozen=True, slots=True)
class ParetoPoint:
    """One traceable policy run and its raw measured objective vector."""

    point_id: str
    experiment_id: str
    policy_name: str
    point_kind: str
    weight_set_version: str | None
    weight_name: str | None
    weights: dict[str, float] | None
    seed: int
    profile_seed: int
    latency_sec: float
    energy_joules: float
    deadline_miss: float
    load_imbalance: float
    quality: float
    weighted_cost: float | None
    is_pareto: bool
    manifest_path: str
    summary_path: str
    trace_path: str

    def objective_vector(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in OBJECTIVE_NAMES}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ParetoExperimentResult:
    """Artifacts and points produced by one complete weight scan."""

    experiment_id: str
    dataset_id: str
    output_dir: str
    weight_set_version: str
    points: tuple[ParetoPoint, ...]
    weight_scan_distinct_objective_count: int
    weight_scan_has_tradeoff: bool
    weight_scan_note: str

    @property
    def pareto_points(self) -> tuple[ParetoPoint, ...]:
        return tuple(point for point in self.points if point.is_pareto)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "dataset_id": self.dataset_id,
            "mode": "replay_profile",
            "output_dir": self.output_dir,
            "weight_set_version": self.weight_set_version,
            "point_count": len(self.points),
            "pareto_point_count": len(self.pareto_points),
            "weight_scan_distinct_objective_count": (
                self.weight_scan_distinct_objective_count
            ),
            "weight_scan_has_tradeoff": self.weight_scan_has_tradeoff,
            "weight_scan_note": self.weight_scan_note,
            "points": [point.to_dict() for point in self.points],
        }


def load_weight_set(path: str | Path) -> WeightSet:
    """Load and validate a versioned JSON weight set."""

    value = _read_object(path)
    version = value.get("version")
    configurations = value.get("configurations")
    if not isinstance(version, str) or not isinstance(configurations, list):
        raise ValueError("weight set requires version and configurations")
    parsed: list[WeightConfiguration] = []
    for item in configurations:
        if not isinstance(item, dict) or not isinstance(item.get("weights"), dict):
            raise ValueError("each weight configuration requires name and weights")
        name = item.get("name")
        if not isinstance(name, str):
            raise ValueError("weight configuration name must be a string")
        parsed.append(
            WeightConfiguration(
                name=name,
                weights=ObjectiveWeights(**item["weights"]),
            )
        )
    return WeightSet(version=version, configurations=tuple(parsed))


def load_resource_profile_set(path: str | Path) -> tuple[str, dict[str, Any]]:
    """Load the version and resource_profiles mapping from a JSON profile set."""

    value = _read_object(path)
    version = value.get("version")
    profiles = value.get("resource_profiles")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("resource profile set requires a non-empty version")
    if not isinstance(profiles, dict):
        raise ValueError("resource profile set requires resource_profiles")
    return version, profiles


def pareto_flags(vectors: Sequence[Mapping[str, float]]) -> list[bool]:
    """Mark non-dominated raw vectors using a simple pairwise comparison."""

    normalized = [_validated_vector(vector) for vector in vectors]
    return [
        not any(
            other_index != index and _dominates(other, vector)
            for other_index, other in enumerate(normalized)
        )
        for index, vector in enumerate(normalized)
    ]


def run_pareto_experiment(
    trace: TraceBundle | str | Path,
    *,
    output_dir: str | Path,
    weight_set: WeightSet | str | Path,
    objective_normalization: ObjectiveNormalization | Mapping[str, float],
    resource_profiles: Mapping[str, Any],
    resource_profile_version: str,
    reference_policies: Sequence[str] = REFERENCE_POLICIES,
    random_seeds: Sequence[int] = (0, 1, 2),
    seed: int = 0,
    profile_seed: int = 0,
    profile_jitter_ratio: float = 0.0,
    min_quality: float = 0.8,
    experiment_id: str | None = None,
) -> ParetoExperimentResult:
    """Run the weighted scan and reference policies on one fixed profile workload."""

    source = load_trace_bundle(trace) if isinstance(trace, (str, Path)) else trace
    resolved_weights = (
        load_weight_set(weight_set)
        if isinstance(weight_set, (str, Path))
        else weight_set
    )
    normalization = (
        objective_normalization
        if isinstance(objective_normalization, ObjectiveNormalization)
        else ObjectiveNormalization(**dict(objective_normalization))
    )
    policies = _validate_reference_policies(reference_policies)
    random_seed_values = _validate_seeds(random_seeds, "random_seeds")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not resource_profile_version.strip():
        raise ValueError("resource_profile_version must be non-empty")

    experiment_name = experiment_id or f"{source.manifest.experiment_id}-pareto"
    experiment_dir = (
        Path(output_dir)
        / f"workload-{_slug(source.manifest.dataset_id)}"
        / f"profile-{_slug(resource_profile_version)}"
        / f"experiment-{_slug(experiment_name)}"
    ).resolve()
    experiment_dir.mkdir(parents=True, exist_ok=True)
    points: list[ParetoPoint] = []

    for configuration in resolved_weights.configurations:
        point_id = f"weight-{_slug(configuration.name)}-seed-{seed}"
        batch = run_baseline_experiment(
            source,
            output_dir=experiment_dir / "runs" / point_id,
            policies=("weighted_objective",),
            seeds=(seed,),
            profile_seed=profile_seed,
            profile_jitter_ratio=profile_jitter_ratio,
            objective_weights=configuration.weights,
            objective_normalization=normalization,
            resource_profiles=resource_profiles,
            experiment_id=f"{experiment_name}-{point_id}",
        )
        points.append(
            _point_from_run(
                point_id,
                batch,
                batch.runs[0],
                point_kind="weight_scan",
                weight_set_version=resolved_weights.version,
                weight_name=configuration.name,
                weights=asdict(configuration.weights),
            )
        )

    for policy_name in policies:
        policy_seeds = random_seed_values if policy_name == "random" else (seed,)
        for policy_seed in policy_seeds:
            point_id = f"reference-{policy_name}-seed-{policy_seed}"
            threshold = (
                min_quality
                if policy_name == "quality_constrained_earliest_finish_time"
                else None
            )
            batch = run_baseline_experiment(
                source,
                output_dir=experiment_dir / "runs" / point_id,
                policies=(policy_name,),
                seeds=(policy_seed,),
                profile_seed=profile_seed,
                profile_jitter_ratio=profile_jitter_ratio,
                resource_profiles=resource_profiles,
                min_quality=threshold,
                experiment_id=f"{experiment_name}-{point_id}",
            )
            points.append(
                _point_from_run(
                    point_id,
                    batch,
                    batch.runs[0],
                    point_kind="reference",
                    weight_set_version=None,
                    weight_name=None,
                    weights=None,
                )
            )

    flags = pareto_flags([point.objective_vector() for point in points])
    marked_points = tuple(
        replace(point, is_pareto=flag)
        for point, flag in zip(points, flags, strict=True)
    )
    weighted_vectors = {
        tuple(round(value, 12) for value in point.objective_vector().values())
        for point in marked_points
        if point.point_kind == "weight_scan"
    }
    distinct_count = len(weighted_vectors)
    has_tradeoff = distinct_count >= 2
    note = (
        f"The weight scan produced {distinct_count} distinct measured objective vectors."
        if has_tradeoff
        else (
            "All weight configurations produced the same measured objective vector; "
            "this workload/profile exposes no observable objective conflict."
        )
    )
    result = ParetoExperimentResult(
        experiment_id=experiment_name,
        dataset_id=source.manifest.dataset_id,
        output_dir=str(experiment_dir),
        weight_set_version=resolved_weights.version,
        points=marked_points,
        weight_scan_distinct_objective_count=distinct_count,
        weight_scan_has_tradeoff=has_tradeoff,
        weight_scan_note=note,
    )
    _write_outputs(
        result,
        normalization=normalization,
        resource_profile_version=resource_profile_version,
        min_quality=min_quality,
    )
    return result


def _point_from_run(
    point_id: str,
    batch: BaselineExperimentResult,
    run: BaselineRunResult,
    *,
    point_kind: str,
    weight_set_version: str | None,
    weight_name: str | None,
    weights: dict[str, float] | None,
) -> ParetoPoint:
    vector = _validated_vector(run.evaluation.objective_vector)
    run_dir = Path(batch.output_dir) / f"{run.policy_name}-seed-{run.seed}"
    return ParetoPoint(
        point_id=point_id,
        experiment_id=run.evaluation.experiment_ids[0],
        policy_name=run.policy_name,
        point_kind=point_kind,
        weight_set_version=weight_set_version,
        weight_name=weight_name,
        weights=weights,
        seed=run.seed,
        profile_seed=run.profile_seed,
        latency_sec=vector["latency_sec"],
        energy_joules=vector["energy_joules"],
        deadline_miss=vector["deadline_miss"],
        load_imbalance=vector["load_imbalance"],
        quality=vector["quality"],
        weighted_cost=run.evaluation.weighted_cost,
        is_pareto=False,
        manifest_path=str((run_dir / "manifest.json").resolve()),
        summary_path=str((run_dir / "summary.json").resolve()),
        trace_path=str((run_dir / "trace.json").resolve()),
    )


def _dominates(first: Mapping[str, float], second: Mapping[str, float]) -> bool:
    no_worse = all(first[name] <= second[name] for name in MINIMIZED_OBJECTIVES) and all(
        first[name] >= second[name] for name in MAXIMIZED_OBJECTIVES
    )
    strictly_better = any(
        first[name] < second[name] for name in MINIMIZED_OBJECTIVES
    ) or any(first[name] > second[name] for name in MAXIMIZED_OBJECTIVES)
    return no_worse and strictly_better


def _validated_vector(vector: Mapping[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for name in OBJECTIVE_NAMES:
        value = vector.get(name)
        if isinstance(value, bool) or not isinstance(value, int | float) or not isfinite(value):
            raise ValueError(f"objective {name!r} must be a finite measured number")
        values[name] = float(value)
    return values


def _write_outputs(
    result: ParetoExperimentResult,
    *,
    normalization: ObjectiveNormalization,
    resource_profile_version: str,
    min_quality: float,
) -> None:
    directory = Path(result.output_dir)
    point_rows = [_flat_point(point) for point in result.points]
    pareto_rows = [_flat_point(point) for point in result.pareto_points]
    manifest = {
        "experiment_id": result.experiment_id,
        "dataset_id": result.dataset_id,
        "mode": "replay_profile",
        "weight_set_version": result.weight_set_version,
        "resource_profile_version": resource_profile_version,
        "objective_normalization": asdict(normalization),
        "pareto_objectives": {
            "minimize": list(MINIMIZED_OBJECTIVES),
            "maximize": list(MAXIMIZED_OBJECTIVES),
        },
        "weighted_cost_used_for_dominance": False,
        "quality_constrained_min_quality": min_quality,
    }
    _write_json(directory / "manifest.json", manifest)
    _write_json(directory / "points.json", result.to_dict())
    _write_csv(directory / "points.csv", point_rows)
    _write_json(
        directory / "pareto_frontier.json",
        {"points": [point.to_dict() for point in result.pareto_points]},
    )
    _write_csv(directory / "pareto_frontier.csv", pareto_rows)
    for name in ("latency_energy", "latency_quality", "energy_quality"):
        _write_csv(directory / f"{name}.csv", point_rows)
    statistics = _point_statistics(result.points)
    _write_json(directory / "statistics.json", {"groups": statistics})
    _write_csv(directory / "statistics.csv", statistics)
    _write_json(directory / "summary.json", result.to_dict())


def _flat_point(point: ParetoPoint) -> dict[str, Any]:
    row = point.to_dict()
    row["weights"] = json.dumps(point.weights, sort_keys=True) if point.weights else None
    return row


def _point_statistics(points: Sequence[ParetoPoint]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str | None], list[ParetoPoint]] = {}
    for point in points:
        groups.setdefault((point.policy_name, point.weight_name), []).append(point)
    rows: list[dict[str, Any]] = []
    for (policy_name, weight_name), group in sorted(
        groups.items(), key=lambda item: (item[0][0], item[0][1] or "")
    ):
        row: dict[str, Any] = {
            "policy_name": policy_name,
            "weight_name": weight_name,
            "run_count": len(group),
        }
        for objective in OBJECTIVE_NAMES:
            values = [float(getattr(point, objective)) for point in group]
            row[f"{objective}_mean"] = fmean(values)
            row[f"{objective}_std"] = pstdev(values) if len(values) > 1 else 0.0
        rows.append(row)
    return rows


def _validate_reference_policies(values: Sequence[str]) -> tuple[str, ...]:
    policies = tuple(values)
    if any(not isinstance(value, str) or not value.strip() for value in policies):
        raise ValueError("reference_policies must contain non-empty names")
    if len(set(policies)) != len(policies):
        raise ValueError("reference_policies must not contain duplicates")
    return policies


def _validate_seeds(values: Sequence[int], field_name: str) -> tuple[int, ...]:
    seeds = tuple(values)
    if not seeds or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in seeds
    ):
        raise ValueError(f"{field_name} must contain non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{field_name} must not contain duplicates")
    return seeds


def _read_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _slug(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in value
    )
