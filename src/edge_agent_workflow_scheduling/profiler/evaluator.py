"""Lightweight metric aggregation for replay and live experiment traces."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from math import isclose, isfinite
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Literal

from edge_agent_workflow_scheduling.profiler.models import (
    AgentRunTrace,
    CallTrace,
    TraceBundle,
)

EvaluationLevel = Literal["call", "agent_run", "experiment"]


@dataclass(frozen=True, slots=True)
class CallEvaluation:
    """Measured metrics and profile metadata for one completed call."""

    run_id: str
    call_id: str
    call_kind: str
    selected_target: str
    policy_name: str
    success: bool
    timeout: bool
    queue_wait_time_sec: float
    input_transfer_time_sec: float
    execution_time_sec: float
    output_transfer_time_sec: float
    total_latency_sec: float
    energy_joules: float
    deadline_sec: float | None
    deadline_miss: bool
    profile_quality: float | None
    quality_source: str
    estimated_objectives: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AgentRunEvaluation:
    """End-to-end metrics and quality summary for one AgentRun."""

    run_id: str
    agent_id: str
    task_id: str
    status: str
    success: bool
    end_to_end_latency_sec: float
    total_energy_joules: float
    llm_call_count: int
    tool_call_count: int
    call_count: int
    average_profile_quality: float | None
    profile_quality_count: int
    profile_quality_coverage: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExperimentEvaluation:
    """Aggregated call, AgentRun, and experiment-level evaluation results."""

    experiment_ids: list[str]
    dataset_id: str
    modes: list[str]
    call_evaluations: list[CallEvaluation]
    agent_runs: list[AgentRunEvaluation]
    average_end_to_end_latency_sec: float
    p95_end_to_end_latency_sec: float
    p99_end_to_end_latency_sec: float
    deadline_miss_rate: float
    deadline_call_count: int
    deadline_miss_count: int
    total_energy_joules: float
    average_profile_quality: float | None
    profile_quality_count: int
    profile_quality_coverage: float
    success_rate: float
    throughput_runs_per_sec: float
    evaluation_window_sec: float
    target_selection_counts: dict[str, int]
    load_imbalance: float
    load_imbalance_by_group: dict[str, float]
    objective_vector: dict[str, float | None]
    weighted_cost: float | None

    def __post_init__(self) -> None:
        if not self.experiment_ids:
            raise ValueError("experiment_ids must not be empty")
        if not self.agent_runs:
            raise ValueError("agent_runs must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_ids": list(self.experiment_ids),
            "dataset_id": self.dataset_id,
            "modes": list(self.modes),
            "calls": [evaluation.to_dict() for evaluation in self.call_evaluations],
            "agent_runs": [evaluation.to_dict() for evaluation in self.agent_runs],
            "metrics": {
                "average_end_to_end_latency_sec": self.average_end_to_end_latency_sec,
                "p95_end_to_end_latency_sec": self.p95_end_to_end_latency_sec,
                "p99_end_to_end_latency_sec": self.p99_end_to_end_latency_sec,
                "deadline_miss_rate": self.deadline_miss_rate,
                "deadline_call_count": self.deadline_call_count,
                "deadline_miss_count": self.deadline_miss_count,
                "total_energy_joules": self.total_energy_joules,
                "average_profile_quality": self.average_profile_quality,
                "profile_quality_count": self.profile_quality_count,
                "profile_quality_coverage": self.profile_quality_coverage,
                "success_rate": self.success_rate,
                "throughput_runs_per_sec": self.throughput_runs_per_sec,
                "evaluation_window_sec": self.evaluation_window_sec,
                "target_selection_counts": dict(self.target_selection_counts),
                "load_imbalance": self.load_imbalance,
                "load_imbalance_by_group": dict(self.load_imbalance_by_group),
                "objective_vector": dict(self.objective_vector),
                "weighted_cost": self.weighted_cost,
            },
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=2)

    def to_csv(self, path: str | Path, *, level: EvaluationLevel = "experiment") -> None:
        """Write flat rows for one evaluation level to CSV."""

        if level not in {"call", "agent_run", "experiment"}:
            raise ValueError("level must be 'call', 'agent_run', or 'experiment'")
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = self._csv_rows(level)
        fieldnames = sorted({key for row in rows for key in row})
        with output_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _csv_rows(self, level: EvaluationLevel) -> list[dict[str, Any]]:
        if level == "call":
            return [evaluation.to_dict() for evaluation in self.call_evaluations]
        if level == "agent_run":
            return [evaluation.to_dict() for evaluation in self.agent_runs]
        metrics = self.to_dict()["metrics"]
        return [
            {
                "experiment_ids": json.dumps(self.experiment_ids),
                "dataset_id": self.dataset_id,
                "modes": json.dumps(self.modes),
                **{
                    key: json.dumps(value)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in metrics.items()
                },
            }
        ]


def evaluate_trace_bundle(trace: TraceBundle) -> ExperimentEvaluation:
    """Evaluate one complete trace without executing or modifying it."""

    return evaluate_traces([trace])


def evaluate_trace_path(path: str | Path) -> ExperimentEvaluation:
    """Load and evaluate one JSON trace bundle."""

    trace = TraceBundle.from_json(Path(path).read_text(encoding="utf-8"))
    return evaluate_trace_bundle(trace)


def write_evaluation_artifacts(
    evaluation: ExperimentEvaluation,
    output_dir: str | Path,
) -> None:
    """Write one JSON summary and flat CSV files for all evaluation levels."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(
        f"{evaluation.to_json()}\n",
        encoding="utf-8",
    )
    evaluation.to_csv(directory / "calls.csv", level="call")
    evaluation.to_csv(directory / "agent_runs.csv", level="agent_run")
    evaluation.to_csv(directory / "experiment.csv", level="experiment")


def evaluate_traces(traces: list[TraceBundle]) -> ExperimentEvaluation:
    """Aggregate multiple traces from one comparable experiment."""

    if not traces:
        raise ValueError("traces must not be empty")
    dataset_ids = {trace.manifest.dataset_id for trace in traces}
    if len(dataset_ids) != 1:
        raise ValueError("all traces must use the same dataset_id")

    call_evaluations: list[CallEvaluation] = []
    agent_runs: list[AgentRunEvaluation] = []
    target_selection_counts: dict[str, int] = {}
    for trace in traces:
        evaluated_calls = [
            _evaluate_call(call, trace) for call in trace.calls
        ]
        call_evaluations.extend(evaluated_calls)
        agent_runs.append(_evaluate_agent_run(trace.run, evaluated_calls))
        for call in evaluated_calls:
            target_selection_counts[call.selected_target] = (
                target_selection_counts.get(call.selected_target, 0) + 1
            )

    latencies = [run.end_to_end_latency_sec for run in agent_runs]
    deadline_calls = [call for call in call_evaluations if call.deadline_sec is not None]
    deadline_misses = sum(call.deadline_miss for call in deadline_calls)
    quality_values = [
        call.profile_quality
        for call in call_evaluations
        if call.profile_quality is not None
    ]
    total_call_count = len(call_evaluations)
    quality_coverage = len(quality_values) / total_call_count if total_call_count else 0.0
    window_sec = _evaluation_window_sec(traces)
    load_imbalance_by_group = _load_imbalance_by_group(traces, call_evaluations)
    aggregate_objectives = {
        "latency_sec": fmean(latencies),
        "energy_joules": sum(call.energy_joules for call in call_evaluations)
        / len(agent_runs),
        "deadline_miss": (
            deadline_misses / len(deadline_calls) if deadline_calls else 0.0
        ),
        "load_imbalance": (
            fmean(load_imbalance_by_group.values())
            if load_imbalance_by_group
            else 0.0
        ),
        "quality": fmean(quality_values) if quality_values else None,
    }
    weighted_cost = _aggregate_weighted_cost(traces, aggregate_objectives)
    return ExperimentEvaluation(
        experiment_ids=[trace.manifest.experiment_id for trace in traces],
        dataset_id=next(iter(dataset_ids)),
        modes=sorted({trace.manifest.mode for trace in traces}),
        call_evaluations=call_evaluations,
        agent_runs=agent_runs,
        average_end_to_end_latency_sec=fmean(latencies),
        p95_end_to_end_latency_sec=_percentile(latencies, 0.95),
        p99_end_to_end_latency_sec=_percentile(latencies, 0.99),
        deadline_miss_rate=(
            deadline_misses / len(deadline_calls) if deadline_calls else 0.0
        ),
        deadline_call_count=len(deadline_calls),
        deadline_miss_count=deadline_misses,
        total_energy_joules=sum(call.energy_joules for call in call_evaluations),
        average_profile_quality=fmean(quality_values) if quality_values else None,
        profile_quality_count=len(quality_values),
        profile_quality_coverage=quality_coverage,
        success_rate=sum(run.success for run in agent_runs) / len(agent_runs),
        throughput_runs_per_sec=len(agent_runs) / window_sec,
        evaluation_window_sec=window_sec,
        target_selection_counts=target_selection_counts,
        load_imbalance=(
            fmean(load_imbalance_by_group.values())
            if load_imbalance_by_group
            else 0.0
        ),
        load_imbalance_by_group=load_imbalance_by_group,
        objective_vector=aggregate_objectives,
        weighted_cost=weighted_cost,
    )


def _aggregate_weighted_cost(
    traces: list[TraceBundle],
    objectives: dict[str, float | None],
) -> float | None:
    """Compute measured aggregate cost from fixed manifest weights and scales."""

    configurations = [
        (
            trace.manifest.scheduler_parameters.get("objective_weights"),
            trace.manifest.scheduler_parameters.get("objective_normalization"),
        )
        for trace in traces
    ]
    if not configurations or any(
        weights is None or normalization is None
        for weights, normalization in configurations
    ):
        return None
    weights, normalization = configurations[0]
    if any(config != configurations[0] for config in configurations[1:]):
        raise ValueError("all traces must use the same objective weights and normalization")
    if not isinstance(weights, dict) or not isinstance(normalization, dict):
        raise ValueError("objective weights and normalization must be objects")
    required_weights = ("latency", "energy", "deadline_miss", "load_imbalance", "quality")
    weight_values = [weights.get(key) for key in required_weights]
    if any(
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not isfinite(value)
        or value < 0
        for value in weight_values
    ):
        raise ValueError("objective weights are incomplete")
    if not isclose(sum(weight_values), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("objective weights must sum to 1.0")
    latency_ref = normalization.get("latency_ref_sec")
    energy_ref = normalization.get("energy_ref_joules")
    if (
        isinstance(latency_ref, bool)
        or not isinstance(latency_ref, int | float)
        or not isfinite(latency_ref)
        or latency_ref <= 0
        or isinstance(energy_ref, bool)
        or not isinstance(energy_ref, int | float)
        or not isfinite(energy_ref)
        or energy_ref <= 0
    ):
        raise ValueError("objective normalization scales must be positive")
    quality = objectives["quality"]
    if quality is None:
        return None
    return (
        float(weights["latency"]) * float(objectives["latency_sec"]) / float(latency_ref)
        + float(weights["energy"]) * float(objectives["energy_joules"]) / float(energy_ref)
        + float(weights["deadline_miss"]) * float(objectives["deadline_miss"])
        + float(weights["load_imbalance"]) * float(objectives["load_imbalance"])
        + float(weights["quality"]) * (1.0 - float(quality))
    )


def _evaluate_call(call: CallTrace, trace: TraceBundle) -> CallEvaluation:
    deadline_sec = call.call_payload.get("deadline_sec")
    if deadline_sec is not None and (
        isinstance(deadline_sec, bool)
        or not isinstance(deadline_sec, int | float)
        or not isfinite(deadline_sec)
        or deadline_sec < 0
    ):
        raise ValueError(f"call {call.call_id!r} has an invalid deadline_sec")
    profile_quality, quality_source = _profile_quality(call, trace)
    return CallEvaluation(
        run_id=call.run_id,
        call_id=call.call_id,
        call_kind=call.call_kind,
        selected_target=call.selected_target,
        policy_name=call.policy_name,
        success=call.success,
        timeout=call.timeout,
        queue_wait_time_sec=call.queue_wait_time_sec,
        input_transfer_time_sec=call.input_transfer_time_sec,
        execution_time_sec=call.execution_time_sec,
        output_transfer_time_sec=call.output_transfer_time_sec,
        total_latency_sec=call.total_latency_sec,
        energy_joules=call.energy_joules,
        deadline_sec=float(deadline_sec) if deadline_sec is not None else None,
        deadline_miss=deadline_sec is not None and call.total_latency_sec > deadline_sec,
        profile_quality=profile_quality,
        quality_source=quality_source,
        estimated_objectives=call.estimated_objectives,
    )


def _evaluate_agent_run(
    run: AgentRunTrace,
    calls: list[CallEvaluation],
) -> AgentRunEvaluation:
    qualities = [call.profile_quality for call in calls if call.profile_quality is not None]
    return AgentRunEvaluation(
        run_id=run.run_id,
        agent_id=run.agent_id,
        task_id=run.task_id,
        status=run.status,
        success=run.status == "completed",
        end_to_end_latency_sec=run.end_to_end_latency_sec,
        total_energy_joules=sum(call.energy_joules for call in calls),
        llm_call_count=run.llm_call_count,
        tool_call_count=run.tool_call_count,
        call_count=len(calls),
        average_profile_quality=fmean(qualities) if qualities else None,
        profile_quality_count=len(qualities),
        profile_quality_coverage=len(qualities) / len(calls) if calls else 0.0,
    )


def _profile_quality(call: CallTrace, trace: TraceBundle) -> tuple[float | None, str]:
    profile = _selected_profile(call, trace)
    if profile is None:
        return None, "unavailable"
    quality_profile = profile.get("quality_profile")
    if not isinstance(quality_profile, dict):
        return None, "unavailable"
    if call.call_kind == "tool" and not quality_profile:
        return 1.0, "equivalent_tool_default"
    metadata = call.call_payload.get("metadata", {})
    task_type = metadata.get("task_type", "default") if isinstance(metadata, dict) else "default"
    if not isinstance(task_type, str) or not task_type:
        return None, "unavailable"
    quality = quality_profile.get(task_type)
    if quality is None:
        return None, "unavailable"
    if isinstance(quality, bool) or not isinstance(quality, int | float) or not 0 <= quality <= 1:
        raise ValueError(f"target {call.selected_target!r} has invalid profile quality")
    return float(quality), "selected_profile"


def _selected_profile(call: CallTrace, trace: TraceBundle) -> dict[str, Any] | None:
    key = "llm_instances" if call.call_kind == "llm" else "tool_replicas"
    resources = trace.manifest.resource_profiles.get(key, [])
    if not isinstance(resources, list):
        raise ValueError(f"resource_profiles.{key} must be a list")
    id_key = "llm_id" if call.call_kind == "llm" else "replica_id"
    for item in resources:
        if not isinstance(item, dict) or not isinstance(item.get("profile"), dict):
            raise ValueError(f"resource_profiles.{key} entries must contain profile objects")
        if item["profile"].get(id_key) == call.selected_target:
            return item["profile"]
    raise ValueError(f"selected target {call.selected_target!r} is missing from resource profiles")


def _evaluation_window_sec(traces: list[TraceBundle]) -> float:
    starts = [_parse_timestamp(trace.run.started_at) for trace in traces]
    finishes = [_parse_timestamp(trace.run.finished_at) for trace in traces]
    window = (max(finishes) - min(starts)).total_seconds()
    if window > 0:
        return window
    return max(sum(trace.run.end_to_end_latency_sec for trace in traces), 1e-12)


def _load_imbalance_by_group(
    traces: list[TraceBundle],
    calls: list[CallEvaluation],
) -> dict[str, float]:
    loads: dict[str, dict[str, float]] = {}
    capacities: dict[str, dict[str, float]] = {}
    for trace in traces:
        profiles = trace.manifest.resource_profiles
        _add_resource_groups(profiles.get("llm_instances", []), "llm", "llm_id", loads, capacities)
        _add_tool_groups(profiles.get("tool_replicas", []), loads, capacities)
    for call in calls:
        group = "llm" if call.call_kind == "llm" else _tool_group(call, traces)
        if group not in loads or call.selected_target not in loads[group]:
            raise ValueError(
                f"selected target {call.selected_target!r} is missing from load groups"
            )
        matching_trace = next(trace for trace in traces if trace.run.run_id == call.run_id)
        execution = next(
            item.execution_time_sec
            for item in matching_trace.calls
            if item.call_id == call.call_id
        )
        loads[group][call.selected_target] += execution / capacities[group][call.selected_target]
    return {
        group: _coefficient_of_variation(group_loads)
        for group, group_loads in loads.items()
        if len(group_loads) >= 2
    }


def _add_resource_groups(
    resources: Any,
    group: str,
    id_key: str,
    loads: dict[str, dict[str, float]],
    capacities: dict[str, dict[str, float]],
) -> None:
    if not isinstance(resources, list):
        raise ValueError("resource profile groups must be lists")
    loads.setdefault(group, {})
    capacities.setdefault(group, {})
    for item in resources:
        profile = item.get("profile") if isinstance(item, dict) else None
        if not isinstance(profile, dict) or not isinstance(profile.get(id_key), str):
            raise ValueError("resource profile entry is invalid")
        capacity = profile.get("max_concurrency")
        if isinstance(capacity, bool) or not isinstance(capacity, int | float) or capacity <= 0:
            raise ValueError("resource profile max_concurrency must be positive")
        loads[group].setdefault(profile[id_key], 0.0)
        capacities[group][profile[id_key]] = float(capacity)


def _add_tool_groups(
    resources: Any,
    loads: dict[str, dict[str, float]],
    capacities: dict[str, dict[str, float]],
) -> None:
    if not isinstance(resources, list):
        raise ValueError("resource_profiles.tool_replicas must be a list")
    for item in resources:
        profile = item.get("profile") if isinstance(item, dict) else None
        if not isinstance(profile, dict) or not isinstance(profile.get("replica_id"), str):
            raise ValueError("Tool resource profile entry is invalid")
        tool_name = profile.get("tool_name")
        if not isinstance(tool_name, str) or not tool_name:
            raise ValueError("Tool resource profile must contain tool_name")
        group = f"tool:{tool_name}"
        _add_resource_groups([item], group, "replica_id", loads, capacities)


def _tool_group(call: CallEvaluation, traces: list[TraceBundle]) -> str:
    for trace in traces:
        if trace.run.run_id != call.run_id:
            continue
        for item in trace.calls:
            if item.call_id == call.call_id:
                if item.tool_name is None:
                    raise ValueError("Tool call trace must contain tool_name")
                return f"tool:{item.tool_name}"
    raise ValueError(f"call {call.call_id!r} is missing from traces")


def _coefficient_of_variation(loads: dict[str, float]) -> float:
    values = list(loads.values())
    mean_load = fmean(values)
    if mean_load == 0:
        return 0.0
    return pstdev(values) / mean_load


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _parse_timestamp(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"invalid trace timestamp: {value!r}") from exc
