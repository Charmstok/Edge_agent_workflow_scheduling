"""Small, reproducible batch runner for baseline scheduler comparisons."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from time import perf_counter
from typing import Any

from edge_agent_workflow_scheduling.common import LLMCall, ToolCall
from edge_agent_workflow_scheduling.executors import ProfileLLMExecutor, ProfileToolExecutor
from edge_agent_workflow_scheduling.profiler.evaluator import (
    ExperimentEvaluation,
    evaluate_trace_bundle,
    write_evaluation_artifacts,
)
from edge_agent_workflow_scheduling.profiler.models import (
    CallTrace,
    ExperimentManifest,
    TraceBundle,
)
from edge_agent_workflow_scheduling.profiler.privacy import content_digest
from edge_agent_workflow_scheduling.profiler.replay import (
    load_trace_bundle,
    reconstruct_call,
    resources_from_manifest,
)
from edge_agent_workflow_scheduling.profiler.trace import TraceBundleStore
from edge_agent_workflow_scheduling.resources import ResourceRegistry, SchedulingConstraints
from edge_agent_workflow_scheduling.scheduler import BaselineScheduler, SchedulerPolicyConfig
from edge_agent_workflow_scheduling.scheduler.objectives import (
    ObjectiveNormalization,
    ObjectiveWeights,
)

DEFAULT_BASELINE_POLICIES = (
    "random",
    "round_robin",
    "least_queue",
    "earliest_finish_time",
    "quality_aware",
    "energy_aware",
    "weighted_objective",
    "quality_constrained_earliest_finish_time",
)

DEFAULT_OBJECTIVE_WEIGHTS = ObjectiveWeights(
    latency=0.2,
    energy=0.2,
    deadline_miss=0.2,
    load_imbalance=0.2,
    quality=0.2,
)
DEFAULT_OBJECTIVE_NORMALIZATION = ObjectiveNormalization(
    latency_ref_sec=1.0,
    energy_ref_joules=1.0,
)
DEFAULT_MIN_QUALITY = 0.8


@dataclass(frozen=True, slots=True)
class BaselineRunResult:
    """One policy/seed run and its measured scheduler overhead."""

    policy_name: str
    seed: int
    profile_seed: int
    source_run_id: str
    input_fingerprint: str
    decision_count: int
    decision_time_total_sec: float
    decision_time_mean_sec: float
    evaluation: ExperimentEvaluation

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "seed": self.seed,
            "profile_seed": self.profile_seed,
            "source_run_id": self.source_run_id,
            "input_fingerprint": self.input_fingerprint,
            "decision_count": self.decision_count,
            "decision_time_total_sec": self.decision_time_total_sec,
            "decision_time_mean_sec": self.decision_time_mean_sec,
            "evaluation": self.evaluation.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class BaselineExperimentResult:
    """All policy/seed runs from one fixed replay workload."""

    experiment_id: str
    dataset_id: str
    output_dir: str
    policies: tuple[str, ...]
    seeds: tuple[int, ...]
    runs: tuple[BaselineRunResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "dataset_id": self.dataset_id,
            "mode": "replay_profile",
            "output_dir": self.output_dir,
            "policies": list(self.policies),
            "seeds": list(self.seeds),
            "runs": [run.to_dict() for run in self.runs],
        }


def run_baseline_experiment(
    trace: TraceBundle | str | Path,
    *,
    output_dir: str | Path,
    policies: Sequence[str] = DEFAULT_BASELINE_POLICIES,
    seeds: Sequence[int] = (0,),
    profile_seed: int = 0,
    profile_jitter_ratio: float = 0.0,
    profile_failure_rate: float = 0.0,
    objective_weights: ObjectiveWeights | Mapping[str, float] | None = None,
    objective_normalization: ObjectiveNormalization | Mapping[str, float] | None = None,
    resource_profiles: Mapping[str, Any] | None = None,
    min_quality: float | None = None,
    experiment_id: str | None = None,
) -> BaselineExperimentResult:
    """Run all requested policies against one immutable replay call stream.

    Each policy/seed receives a fresh resource registry and fresh seeded profile
    executors. The source trace is never modified; generated traces contain the
    same call IDs and order with the new decisions and profile measurements.
    """

    source_trace = _load_trace(trace)
    requested_policy_names = _validate_names(policies, "policies")
    run_seeds = _validate_seeds(seeds)
    if profile_seed < 0:
        raise ValueError("profile_seed must be non-negative")
    weights = _coerce_weights(objective_weights)
    normalization = _coerce_normalization(objective_normalization)
    if (
        tuple(requested_policy_names) == DEFAULT_BASELINE_POLICIES
        and weights is None
        and normalization is None
    ):
        weights = DEFAULT_OBJECTIVE_WEIGHTS
        normalization = DEFAULT_OBJECTIVE_NORMALIZATION
    if (weights is None) != (normalization is None):
        raise ValueError(
            "objective_weights and objective_normalization must be configured together"
        )
    _validate_profile_parameters(profile_jitter_ratio, profile_failure_rate)

    experiment_name = experiment_id or f"{source_trace.manifest.experiment_id}-baseline"
    profile_snapshot = deepcopy(resource_profiles or source_trace.manifest.resource_profiles)
    skipped_policies: dict[str, str] = {}
    policy_names = requested_policy_names
    if tuple(requested_policy_names) == DEFAULT_BASELINE_POLICIES:
        policy_names, skipped_policies = _policies_supported_by_profiles(
            requested_policy_names,
            profile_snapshot,
        )
    base_trace = _trace_with_resource_profiles(source_trace, profile_snapshot)
    workload_fingerprint = content_digest([call.call_digest for call in base_trace.calls])
    profile_version = _profile_version(profile_snapshot)
    experiment_dir = (
        Path(output_dir)
        / f"workload-{_slug(base_trace.manifest.dataset_id)}"
        / f"profile-{_slug(profile_version)}"
        / f"experiment-{_slug(experiment_name)}"
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        experiment_dir / "manifest.json",
        {
            "experiment_id": experiment_name,
            "dataset_id": base_trace.manifest.dataset_id,
            "mode": "replay_profile",
            "source_run_id": base_trace.run.run_id,
            "source_trace_fingerprint": workload_fingerprint,
            "policies": list(policy_names),
            "requested_policies": list(requested_policy_names),
            "skipped_policies": skipped_policies,
            "seeds": list(run_seeds),
            "profile_seed": profile_seed,
            "profile_jitter_ratio": profile_jitter_ratio,
            "profile_failure_rate": profile_failure_rate,
            "objective_weights": _json_config(weights),
            "objective_normalization": _json_config(normalization),
            "resource_profile_version": profile_version,
            "min_quality": (
                DEFAULT_MIN_QUALITY
                if min_quality is None
                and "quality_constrained_earliest_finish_time" in policy_names
                else min_quality
            ),
        },
    )

    runs: list[BaselineRunResult] = []
    for policy_name in policy_names:
        for seed in run_seeds:
            policy_min_quality = (
                DEFAULT_MIN_QUALITY
                if min_quality is None
                and policy_name == "quality_constrained_earliest_finish_time"
                else min_quality
            )
            run_dir = experiment_dir / f"{_slug(policy_name)}-seed-{seed}"
            result = _run_one_policy(
                base_trace,
                policy_name=policy_name,
                seed=seed,
                profile_seed=profile_seed,
                profile_jitter_ratio=profile_jitter_ratio,
                profile_failure_rate=profile_failure_rate,
                objective_weights=weights,
                objective_normalization=normalization,
                min_quality=policy_min_quality,
                experiment_id=experiment_name,
                output_dir=run_dir,
            )
            runs.append(result)

    experiment = BaselineExperimentResult(
        experiment_id=experiment_name,
        dataset_id=base_trace.manifest.dataset_id,
        output_dir=str(experiment_dir),
        policies=policy_names,
        seeds=run_seeds,
        runs=tuple(runs),
    )
    _write_merged_csv(experiment_dir / "baseline_results.csv", runs)
    _write_json(experiment_dir / "summary.json", experiment.to_dict())
    return experiment


def _run_one_policy(
    source_trace: TraceBundle,
    *,
    policy_name: str,
    seed: int,
    profile_seed: int,
    profile_jitter_ratio: float,
    profile_failure_rate: float,
    objective_weights: ObjectiveWeights | None,
    objective_normalization: ObjectiveNormalization | None,
    min_quality: float | None,
    experiment_id: str,
    output_dir: Path,
) -> BaselineRunResult:
    resources = resources_from_manifest(source_trace)
    scheduler = BaselineScheduler(
        policy_name,
        constraints=SchedulingConstraints(min_quality=min_quality),
        policy_config=SchedulerPolicyConfig(
            random_seed=seed,
            record_objectives=True,
            objective_weights=objective_weights,
            objective_normalization=objective_normalization,
        ),
    )
    profile_pool = _ProfileExecutorPool(
        profile_seed=profile_seed,
        run_seed=seed,
        jitter_ratio=profile_jitter_ratio,
        failure_rate=profile_failure_rate,
    )
    calls: list[CallTrace] = []
    decisions: list[dict[str, Any]] = []
    decision_times: list[float] = []
    elapsed_profile_sec = 0.0
    for source_batch in _call_batches(source_trace.calls):
        assignments: list[tuple[CallTrace, LLMCall | ToolCall, Any]] = []
        for source_call in source_batch:
            call = reconstruct_call(source_call)
            started = perf_counter()
            decision = scheduler.schedule(call, resources=resources)
            decision_time_sec = perf_counter() - started
            decision_times.append(decision_time_sec)
            decisions.append({**decision.to_dict(), "decision_time_sec": decision_time_sec})
            _enqueue_resource_state(resources, call, decision.selected_target)
            assignments.append((source_call, call, decision))

        results = _execute_batch(profile_pool, assignments, resources)
        batch_duration = 0.0
        for (source_call, call, decision), result in zip(assignments, results, strict=True):
            call_duration = _result_duration(call, result)
            batch_duration = max(batch_duration, call_duration)
            calls.append(
                replace(
                    _call_trace_from_profile(source_call, call, decision, result),
                    finished_at=(
                        datetime.fromisoformat(source_trace.run.started_at)
                        + timedelta(seconds=elapsed_profile_sec + call_duration)
                    ).isoformat(),
                )
            )
            _update_resource_state(resources, call, decision.selected_target, result)
        elapsed_profile_sec += batch_duration

    generated_trace = _build_trace_bundle(source_trace, calls, policy_name)
    manifest = _build_policy_manifest(
        source_trace.manifest,
        experiment_id=experiment_id,
        policy_name=policy_name,
        seed=seed,
        profile_seed=profile_seed,
        objective_weights=objective_weights,
        objective_normalization=objective_normalization,
        min_quality=min_quality,
    )
    generated_trace = replace(generated_trace, manifest=manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    TraceBundleStore(output_dir / "trace.json").write(generated_trace)
    _write_json(output_dir / "manifest.json", manifest.to_dict())
    _write_json(output_dir / "decisions.json", {"decisions": decisions})
    evaluation = evaluate_trace_bundle(generated_trace)
    write_evaluation_artifacts(evaluation, output_dir)
    decision_total = sum(decision_times)
    summary = {
        "mode": "replay_profile",
        "policy_name": policy_name,
        "seed": seed,
        "profile_seed": profile_seed,
        "source_run_id": source_trace.run.run_id,
        "input_fingerprint": content_digest([call.call_digest for call in source_trace.calls]),
        "decision_time_total_sec": decision_total,
        "decision_time_mean_sec": decision_total / len(decision_times) if decision_times else 0.0,
        "evaluation": evaluation.to_dict(),
    }
    _write_json(output_dir / "summary.json", summary)
    return BaselineRunResult(
        policy_name=policy_name,
        seed=seed,
        profile_seed=profile_seed,
        source_run_id=source_trace.run.run_id,
        input_fingerprint=summary["input_fingerprint"],
        decision_count=len(decision_times),
        decision_time_total_sec=decision_total,
        decision_time_mean_sec=summary["decision_time_mean_sec"],
        evaluation=evaluation,
    )


class _ProfileExecutorPool:
    def __init__(
        self,
        *,
        profile_seed: int,
        run_seed: int,
        jitter_ratio: float,
        failure_rate: float,
    ) -> None:
        self.profile_seed = profile_seed
        self.run_seed = run_seed
        self.jitter_ratio = jitter_ratio
        self.failure_rate = failure_rate
        self._llm: dict[str, ProfileLLMExecutor] = {}
        self._tool: dict[str, ProfileToolExecutor] = {}

    def execute(self, call: LLMCall | ToolCall, target_id: str, resources: ResourceRegistry):
        if isinstance(call, LLMCall):
            profile = resources.llm_snapshot(target_id).profile
            executor = self._llm.get(target_id)
            if executor is None:
                executor = ProfileLLMExecutor(
                    profile=profile,
                    seed=self._seed_for(target_id),
                    jitter_ratio=self.jitter_ratio,
                    failure_rate=self.failure_rate,
                )
                self._llm[target_id] = executor
            return executor.execute(call)
        profile = resources.tool_snapshot(target_id).profile
        executor = self._tool.get(target_id)
        if executor is None:
            executor = ProfileToolExecutor(
                profile=profile,
                seed=self._seed_for(target_id),
                output={"mode": "replay_profile", "replica_id": target_id},
                jitter_ratio=self.jitter_ratio,
                failure_rate=self.failure_rate,
            )
            self._tool[target_id] = executor
        return executor.execute(call)

    def _seed_for(self, target_id: str) -> int:
        digest = hashlib.sha256(target_id.encode("utf-8")).digest()
        target_offset = int.from_bytes(digest[:4], "big")
        return (self.profile_seed + self.run_seed * 1_000_003 + target_offset) % (2**32)


def _call_trace_from_profile(
    source: CallTrace,
    call: LLMCall | ToolCall,
    decision: Any,
    result: Any,
) -> CallTrace:
    success = result.success
    status = "succeeded" if success else "failed"
    total_latency = (
        result.queue_wait_time_sec
        + result.input_transfer_time_sec
        + (result.inference_time_sec if isinstance(call, LLMCall) else result.execution_time_sec)
        + result.output_transfer_time_sec
    )
    common = {
        "selected_target": decision.selected_target,
        "policy_name": decision.policy_name,
        "status": status,
        "success": success,
        "timeout": result.error_code == "timeout",
        "queue_wait_time_sec": result.queue_wait_time_sec,
        "input_transfer_time_sec": result.input_transfer_time_sec,
        "execution_time_sec": (
            result.inference_time_sec
            if isinstance(call, LLMCall)
            else result.execution_time_sec
        ),
        "output_transfer_time_sec": result.output_transfer_time_sec,
        "total_latency_sec": total_latency,
        "energy_joules": result.energy_joules,
        "decided_at": decision.decided_at,
        "finished_at": result.finished_at,
        "result_metadata": result.metadata,
        "error_code": result.error_code,
        "error_message": result.error_message,
        "estimated_objectives": decision.estimated_objectives,
    }
    if isinstance(call, LLMCall):
        return replace(
            source,
            **common,
            model_name=result.response_model or source.model_name,
            raw_response_items=deepcopy(result.output_items),
        )
    from edge_agent_workflow_scheduling.tools import build_function_call_output

    return replace(
        source,
        **common,
        function_call_output=build_function_call_output(call.call_id, result),
    )


def _build_trace_bundle(
    source: TraceBundle,
    calls: list[CallTrace],
    policy_name: str,
) -> TraceBundle:
    started = datetime.fromisoformat(source.run.started_at)
    finish_times = [datetime.fromisoformat(call.finished_at) for call in calls]
    finished = max(finish_times, default=started)
    total_latency = max((finished - started).total_seconds(), 0.0)
    if not calls:
        total_latency = 0.0
    success = all(call.success for call in calls)
    status = "completed" if success else "failed"
    raw_items = [item for call in calls for item in call.raw_response_items]
    function_outputs = [call.function_call_output for call in calls if call.function_call_output]
    run = replace(
        source.run,
        status=status,
        state_transitions=[*source.run.state_transitions[:-1], status],
        final_output=source.run.final_output if success else None,
        total_rounds=max((call.turn_index for call in calls), default=source.run.total_rounds),
        llm_call_count=sum(call.call_kind == "llm" for call in calls),
        tool_call_count=sum(call.call_kind == "tool" for call in calls),
        end_to_end_latency_sec=total_latency,
        started_at=started.isoformat(),
        finished_at=finished.isoformat(),
        raw_response_items=raw_items,
        function_call_outputs=function_outputs,
        error_code=None if success else "profile_execution_failed",
        error_message=None if success else "one or more profile calls failed",
    )
    return TraceBundle(manifest=source.manifest, run=run, calls=calls)


def _build_policy_manifest(
    source: ExperimentManifest,
    *,
    experiment_id: str,
    policy_name: str,
    seed: int,
    profile_seed: int,
    objective_weights: ObjectiveWeights | None,
    objective_normalization: ObjectiveNormalization | None,
    min_quality: float | None,
) -> ExperimentManifest:
    scheduler_parameters = {
        "random_seed": seed,
        "profile_seed": profile_seed,
        "record_objectives": True,
        "execution_mode": "profile",
        "objective_weights": _json_config(objective_weights),
        "objective_normalization": _json_config(objective_normalization),
        "constraints": {"min_quality": min_quality, "allowed_node_ids": None},
    }
    return replace(
        source,
        experiment_id=f"{experiment_id}-{policy_name}-seed-{seed}",
        scheduler_name=policy_name,
        scheduler_parameters=scheduler_parameters,
        scheduler_seed=seed,
        mode="replay",
        run_started_at=source.run_started_at,
    )


def _update_resource_state(
    resources: ResourceRegistry,
    call: LLMCall | ToolCall,
    target_id: str,
    result: Any,
) -> None:
    if isinstance(call, LLMCall):
        snapshot = resources.llm_snapshot(target_id)
        resources.update_llm_state(
            replace(
                snapshot.state,
                queue_len=max(snapshot.state.queue_len - 1, 0),
                running_requests=0,
                avg_latency_sec=result.inference_time_sec,
                tokens_per_sec=(
                    (call.input_tokens + call.estimated_output_tokens) / result.inference_time_sec
                    if result.inference_time_sec > 0
                    else snapshot.state.tokens_per_sec
                ),
            )
        )
    else:
        snapshot = resources.tool_snapshot(target_id)
        resources.update_tool_state(
            replace(
                snapshot.state,
                queue_len=max(snapshot.state.queue_len - 1, 0),
                running_tasks=0,
                avg_execution_time_sec=result.execution_time_sec,
            )
        )


def _call_batches(calls: Sequence[CallTrace]) -> list[list[CallTrace]]:
    """Group contiguous same-turn calls that can arrive in one scheduling round."""

    batches: list[list[CallTrace]] = []
    for call in calls:
        if (
            batches
            and batches[-1][0].turn_index == call.turn_index
            and batches[-1][0].call_kind == call.call_kind
        ):
            batches[-1].append(call)
        else:
            batches.append([call])
    return batches


def _enqueue_resource_state(
    resources: ResourceRegistry,
    call: LLMCall | ToolCall,
    target_id: str,
) -> None:
    """Reserve a queued slot while a same-round batch is being assigned."""

    if isinstance(call, LLMCall):
        snapshot = resources.llm_snapshot(target_id)
        resources.update_llm_state(
            replace(
                snapshot.state,
                queue_len=snapshot.state.queue_len + 1,
            )
        )
    else:
        snapshot = resources.tool_snapshot(target_id)
        resources.update_tool_state(
            replace(
                snapshot.state,
                queue_len=snapshot.state.queue_len + 1,
            )
        )


def _execute_batch(
    profile_pool: _ProfileExecutorPool,
    assignments: Sequence[tuple[CallTrace, LLMCall | ToolCall, Any]],
    resources: ResourceRegistry,
) -> list[Any]:
    """Execute one scheduling batch concurrently, preserving input order."""

    if len(assignments) == 1:
        _, call, decision = assignments[0]
        return [profile_pool.execute(call, decision.selected_target, resources)]
    grouped: dict[str, list[tuple[int, LLMCall | ToolCall]]] = {}
    for index, (_, call, decision) in enumerate(assignments):
        grouped.setdefault(decision.selected_target, []).append((index, call))

    def execute_target(group: list[tuple[int, LLMCall | ToolCall]]) -> list[tuple[int, Any]]:
        results: list[tuple[int, Any]] = []
        queued_duration = 0.0
        for index, call in group:
            result = profile_pool.execute(call, assignments[index][2].selected_target, resources)
            if queued_duration:
                result = replace(
                    result,
                    queue_wait_time_sec=result.queue_wait_time_sec + queued_duration,
                )
            results.append((index, result))
            queued_duration += _result_duration(call, result)
        return results

    completed: list[Any | None] = [None] * len(assignments)
    with ThreadPoolExecutor(max_workers=len(grouped)) as pool:
        futures = [pool.submit(execute_target, group) for group in grouped.values()]
        for future in futures:
            for index, result in future.result():
                completed[index] = result
    return [result for result in completed if result is not None]


def _result_duration(call: LLMCall | ToolCall, result: Any) -> float:
    return (
        result.queue_wait_time_sec
        + result.input_transfer_time_sec
        + (result.inference_time_sec if isinstance(call, LLMCall) else result.execution_time_sec)
        + result.output_transfer_time_sec
    )


def _write_merged_csv(path: Path, runs: Sequence[BaselineRunResult]) -> None:
    rows: list[dict[str, Any]] = []
    for run in runs:
        metrics = run.evaluation.to_dict()["metrics"]
        rows.append(
            {
                "policy_name": run.policy_name,
                "seed": run.seed,
                "profile_seed": run.profile_seed,
                "source_run_id": run.source_run_id,
                "input_fingerprint": run.input_fingerprint,
                "decision_count": run.decision_count,
                "decision_time_total_sec": run.decision_time_total_sec,
                "decision_time_mean_sec": run.decision_time_mean_sec,
                **{
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in metrics.items()
                },
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_trace(trace: TraceBundle | str | Path) -> TraceBundle:
    return load_trace_bundle(trace) if isinstance(trace, (str, Path)) else trace


def _trace_with_resource_profiles(trace: TraceBundle, profiles: Mapping[str, Any]) -> TraceBundle:
    if not isinstance(profiles, Mapping):
        raise ValueError("resource_profiles must be a mapping")
    return replace(
        trace,
        manifest=replace(
            trace.manifest,
            resource_profiles=deepcopy(dict(profiles)),
        ),
    )


def _policies_supported_by_profiles(
    policies: Sequence[str],
    profiles: Mapping[str, Any],
) -> tuple[tuple[str, ...], dict[str, str]]:
    """Keep the default command usable with older traces lacking optional profiles."""

    resources = [
        item.get("profile")
        for group in ("llm_instances", "tool_replicas")
        for item in profiles.get(group, [])
        if isinstance(item, Mapping) and isinstance(item.get("profile"), Mapping)
    ]
    missing_energy = any(
        not isinstance(profile.get("energy_profile"), Mapping)
        or (
            "joules_per_token" not in profile["energy_profile"]
            and "joules_per_call" not in profile["energy_profile"]
        )
        for profile in resources
    )
    missing_quality = any(
        profile.get("llm_id") is not None
        and (
            not isinstance(profile.get("quality_profile"), Mapping)
            or "default" not in profile["quality_profile"]
        )
        for profile in resources
    )
    skipped: dict[str, str] = {}
    supported: list[str] = []
    for policy in policies:
        reason = None
        if policy in {"energy_aware", "weighted_objective"} and missing_energy:
            reason = "required energy profile is missing"
        elif policy in {
            "quality_aware",
            "quality_constrained_earliest_finish_time",
            "weighted_objective",
        } and missing_quality:
            reason = "required quality profile is missing"
        if reason is None:
            supported.append(policy)
        else:
            skipped[policy] = reason
    return tuple(supported), skipped


def _coerce_weights(
    value: ObjectiveWeights | Mapping[str, float] | None,
) -> ObjectiveWeights | None:
    if value is None or isinstance(value, ObjectiveWeights):
        return value
    return ObjectiveWeights(**dict(value))


def _coerce_normalization(
    value: ObjectiveNormalization | Mapping[str, float] | None,
) -> ObjectiveNormalization | None:
    if value is None or isinstance(value, ObjectiveNormalization):
        return value
    return ObjectiveNormalization(**dict(value))


def _json_config(value: Any) -> dict[str, float] | None:
    return asdict(value) if value is not None else None


def _validate_names(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    names = tuple(values)
    if not names or any(not isinstance(value, str) or not value.strip() for value in names):
        raise ValueError(f"{field_name} must contain at least one non-empty name")
    if len(set(names)) != len(names):
        raise ValueError(f"{field_name} must not contain duplicates")
    return names


def _validate_seeds(values: Sequence[int]) -> tuple[int, ...]:
    seeds = tuple(values)
    if not seeds or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in seeds
    ):
        raise ValueError("seeds must contain non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must not contain duplicates")
    return seeds


def _validate_profile_parameters(jitter_ratio: float, failure_rate: float) -> None:
    if not 0.0 <= jitter_ratio < 1.0:
        raise ValueError("profile_jitter_ratio must be between 0.0 and 1.0")
    if not 0.0 <= failure_rate <= 1.0:
        raise ValueError("profile_failure_rate must be between 0.0 and 1.0")


def _profile_version(profiles: Mapping[str, Any]) -> str:
    versions: set[str] = set()
    for group in ("llm_instances", "tool_replicas"):
        for item in profiles.get(group, []):
            profile = item.get("profile", {}) if isinstance(item, dict) else {}
            metadata = profile.get("metadata", {}) if isinstance(profile, dict) else {}
            if isinstance(metadata, dict) and isinstance(metadata.get("profile_version"), str):
                versions.add(metadata["profile_version"])
    return "+".join(sorted(versions)) if versions else "unspecified"


def _slug(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in value
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
