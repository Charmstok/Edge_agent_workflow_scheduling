"""Fair profile replay comparison of Double DQN with all eight baselines."""

from __future__ import annotations

import csv
import itertools
import json
from dataclasses import asdict, replace
from pathlib import Path
from statistics import fmean, pstdev
from time import perf_counter
from typing import Any

import numpy as np

from edge_agent_workflow_scheduling.common import ScheduleDecision
from edge_agent_workflow_scheduling.profiler.baseline_experiment import (
    DEFAULT_BASELINE_POLICIES,
    run_replay_policy,
)
from edge_agent_workflow_scheduling.profiler.pareto import OBJECTIVE_NAMES, pareto_flags
from edge_agent_workflow_scheduling.profiler.privacy import content_digest
from edge_agent_workflow_scheduling.profiler.replay import (
    load_trace_bundle,
    reconstruct_call,
    resources_from_manifest,
)
from edge_agent_workflow_scheduling.resources import SchedulingConstraints
from edge_agent_workflow_scheduling.rl.dqn import DoubleDQNAgent, DQNConfig
from edge_agent_workflow_scheduling.rl.environment import SchedulingEnv
from edge_agent_workflow_scheduling.rl.runner import train_agent
from edge_agent_workflow_scheduling.scheduler.baseline import NoFeasibleTargetError
from edge_agent_workflow_scheduling.scheduler.objectives import (
    ObjectiveNormalization,
    ObjectiveWeights,
    estimate_objectives_dict,
)
from edge_agent_workflow_scheduling.scheduler.types import (
    call_id_for,
    call_kind_for,
    candidate_from_snapshot,
)


class DQNScheduler:
    """Greedy checkpoint policy using the exact training observation encoder."""

    def __init__(
        self,
        env: SchedulingEnv,
        agent: DoubleDQNAgent,
        checkpoint: dict,
        *,
        allow_missing_optional_profiles: bool = False,
    ) -> None:
        self.env = env
        self.agent = agent
        self.index = 0
        self.checkpoint = checkpoint
        self.allow_missing_optional_profiles = allow_missing_optional_profiles

    def manifest_parameters(self) -> dict:
        return {
            "checkpoint": self.checkpoint,
            "evaluation_epsilon": 0.0,
            "environment_schema": "scheduling-env-v1",
        }

    def schedule(self, call, *, resources) -> ScheduleDecision:
        if resources is not self.env.resources:
            raise ValueError("RL encoder must share the replay resource registry")
        details = resources.action_mask_details(call, constraints=self.env.constraints)
        mask = self.env.action_mask_for(call)
        if not mask.any():
            raise NoFeasibleTargetError(call_kind_for(call), details)
        observation = self.env.observation_for(call, self.index)
        action = self.agent.select_action(observation, mask, epsilon=0.0)
        candidates = [
            candidate_from_snapshot(item)
            for item in resources.eligible_snapshots(call, constraints=self.env.constraints)
        ]
        target = self.env.target_ids[action]
        selected = next(item for item in candidates if item.target_id == target)
        self.index += 1
        return ScheduleDecision(
            call_id=call_id_for(call),
            call_kind=call_kind_for(call),
            selected_target=target,
            policy_name="double_dqn",
            reason="greedy Double DQN with shared replay action mask",
            candidate_target_ids=list(details.target_ids),
            action_mask=list(details.values),
            rejection_reasons={
                key: list(value) for key, value in details.reasons_by_target().items()
            },
            estimated_objectives=estimate_objectives_dict(
                call,
                selected,
                candidates,
                allow_missing_optional_profiles=self.allow_missing_optional_profiles,
            ),
        )


def checkpoint_metadata(source, env, config, *, seed, profile_seed, code_version):
    """Version/content checks, including arrival timestamps and initial states."""
    return {
        "environment_schema": "scheduling-env-v1",
        "observation_size": env.observation_space.shape[0],
        "environment_config": asdict(env.config),
        "action_ids": list(env.target_ids),
        "workload_version": source.manifest.dataset_id,
        "workload_fingerprint": content_digest(
            [{"payload": item.call_payload, "created_at": item.created_at} for item in source.calls]
        ),
        "workload_parameters": source.manifest.workload_parameters,
        "resource_profiles_fingerprint": content_digest(source.manifest.resource_profiles),
        "resource_version": {
            "llm": source.manifest.llm_profile_version,
            "tool": source.manifest.tool_profile_version,
        },
        "reward_weights": asdict(env.reward.weights),
        "objective_normalization": asdict(env.reward.normalization),
        "constraints": {"min_quality": env.constraints.min_quality},
        "training_config": asdict(config),
        "training_seed": seed,
        "training_profile_seed": profile_seed,
        "code_version": code_version,
    }


def load_comparison_checkpoint(path, expected_metadata):
    agent, metadata = DoubleDQNAgent.load(path)
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f"checkpoint metadata mismatch: {key}")
    if agent.online.weights["w1"].shape[0] != metadata["observation_size"]:
        raise ValueError("checkpoint observation dimension mismatch")
    if agent.online.weights["w2"].shape[1] != len(metadata["action_ids"]):
        raise ValueError("checkpoint action dimension mismatch")
    return agent


def run_rl_comparison(
    trace_path: str | Path,
    *,
    output_dir: str | Path,
    config: dict[str, Any],
    resource_profiles: dict[str, Any],
    code_version: str,
    checkpoint_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train or reload checkpoints, then evaluate every policy on identical inputs.

    Scheduler seeds never affect profile noise. Statistical units are profile
    seeds; scheduler/training seeds are averaged inside each paired noise unit.
    """
    seeds = _seeds(config["scheduler_seeds"])
    profile_seeds = _seeds(config["profile_seeds"])
    if len(seeds) < 2 or len(profile_seeds) < 2:
        raise ValueError("comparison requires at least two scheduler and profile seeds")
    if len(profile_seeds) > 16:
        raise ValueError("exact sign-flip test supports at most 16 profile seeds")
    weights = ObjectiveWeights(**config["objective_weights"])
    normalization = ObjectiveNormalization(**config["objective_normalization"])
    dqn_config = DQNConfig(**config["training"])
    jitter = config["profile_jitter_ratio"]
    failure_rate = config.get("profile_failure_rate", 0.0)
    if not 0 <= jitter < 1 or not 0 <= failure_rate <= 1:
        raise ValueError("invalid profile simulation parameters")
    # The inference protocol is fixed before any result is examined.
    statistics = config["statistical_protocol"]
    if statistics != {
        "test": "paired_exact_sign_flip",
        "unit": "profile_seed",
        "alternative": "rl_cost_lower",
        "alpha": 0.05,
        "correction": "bonferroni_8",
    }:
        raise ValueError("unsupported statistical protocol")
    source = load_trace_bundle(trace_path)
    source = replace(
        source,
        manifest=replace(
            source.manifest,
            resource_profiles=resource_profiles,
            code_version=code_version,
            llm_profile_version=config["resource_profile_version"],
            tool_profile_version=config["resource_profile_version"],
        ),
    )
    calls = [reconstruct_call(item) for item in source.calls]
    if len({item.call_id for item in source.calls}) != len(calls):
        raise ValueError("replay call IDs must be unique")
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "source_trace.json").write_text(source.to_json() + "\n", encoding="utf-8")
    manifest = {
        "config": config,
        "mode": "replay_profile",
        "code_version": code_version,
        "source_trace": str(Path(trace_path).resolve()),
        "input_fingerprint": content_digest([item.call_digest for item in source.calls]),
        "resource_profiles_fingerprint": content_digest(resource_profiles),
        "call_ids": [item.call_id for item in source.calls],
        "arrival_information": {
            "run_started_at": source.run.started_at,
            "call_created_at": [item.created_at for item in source.calls],
            "workload_parameters": source.manifest.workload_parameters,
        },
        "profile_noise_semantics": "sha256(call_id,target_id)+profile_seed",
        "baseline_policies": list(DEFAULT_BASELINE_POLICIES),
        "quality_constraint_scope": "quality_constrained_EFT_only; per_call_constraints_shared",
        "percentile_population": "repeated_profile_runs_of_one_fixed_AgentRun",
        "measurement_scope": (
            "synthetic_profiles; no real hardware energy or task quality measurement"
        ),
        "training_scope": (
            "same fixed call stream; estimated per_call reward; no generalization claim"
        ),
    }
    _write_json(directory / "manifest.json", manifest)
    preflight_call_id = None
    try:
        resources = resources_from_manifest(source)
        for call in calls:
            preflight_call_id = call_id_for(call)
            for threshold in (None, config["min_quality"]):
                constraints = SchedulingConstraints(min_quality=threshold)
                details = resources.action_mask_details(call, constraints=constraints)
                candidates = [
                    candidate_from_snapshot(item)
                    for item in resources.eligible_snapshots(call, constraints=constraints)
                ]
                if not candidates:
                    raise NoFeasibleTargetError(call_kind_for(call), details)
                for candidate in candidates:
                    estimate_objectives_dict(call, candidate, candidates)
    except (ValueError, KeyError, TypeError) as exc:
        _write_json(
            directory / "preflight_failure.json",
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "failed_call_id": preflight_call_id,
                "call_ids": manifest["call_ids"],
                "executor_dispatched": False,
            },
        )
        raise
    agents = {}
    checkpoints = {}
    training_summary = []
    for seed in seeds:
        env = SchedulingEnv(
            calls,
            resources_from_manifest(source),
            reward_weights=weights,
            reward_normalization=normalization,
            profile_seed=config["training_profile_seed"],
        )
        metadata = checkpoint_metadata(
            source,
            env,
            dqn_config,
            seed=seed,
            profile_seed=config["training_profile_seed"],
            code_version=code_version,
        )
        train_dir = directory / "training" / f"seed-{seed}"
        train_dir.mkdir(parents=True, exist_ok=True)
        if checkpoint_dir is None:
            agent = DoubleDQNAgent(
                env.observation_space.shape[0], env.action_space.n, config=dqn_config, seed=seed
            )
            started = perf_counter()
            history = train_agent(env, agent, dqn_config.episodes, seed=seed)
            training_summary.append({"seed": seed, "wall_time_sec": perf_counter() - started})
            _write_json(train_dir / "history.json", history)
            agent.save(train_dir / "checkpoint.json", metadata)
            path = train_dir / "checkpoint.json"
        else:
            path = Path(checkpoint_dir) / f"seed-{seed}" / "checkpoint.json"
        # Evaluation always uses restored weights, never the in-memory training model.
        agents[seed] = load_comparison_checkpoint(path, metadata)
        checkpoints[seed] = {
            "path": str(path.resolve()),
            "content_digest": content_digest(json.loads(path.read_text(encoding="utf-8"))),
            "training_seed": seed,
        }
        _write_json(train_dir / "evaluation_checkpoint.json", {"path": str(path.resolve())})

    rows = []
    for profile_seed in profile_seeds:
        for seed in seeds:
            for policy in (*DEFAULT_BASELINE_POLICIES, "double_dqn"):
                run_dir = directory / "runs" / f"profile-{profile_seed}" / f"{policy}-seed-{seed}"
                threshold = (
                    config["min_quality"] if policy == DEFAULT_BASELINE_POLICIES[-1] else None
                )
                factory = None
                if policy == "double_dqn":

                    def factory(resources, agent=agents[seed], checkpoint=checkpoints[seed]):
                        return DQNScheduler(
                            SchedulingEnv(
                                calls,
                                resources,
                                reward_weights=weights,
                                reward_normalization=normalization,
                            ),
                            agent,
                            checkpoint,
                        )

                try:
                    run = run_replay_policy(
                        source,
                        policy_name=policy,
                        seed=seed,
                        profile_seed=profile_seed,
                        profile_jitter_ratio=jitter,
                        profile_failure_rate=failure_rate,
                        objective_weights=weights,
                        objective_normalization=normalization,
                        min_quality=threshold,
                        experiment_id=config["experiment_id"],
                        output_dir=run_dir,
                        scheduler_factory=factory,
                    )
                    generated = load_trace_bundle(run_dir / "trace.json")
                    if [(c.call_id, c.call_digest, c.created_at) for c in generated.calls] != [
                        (c.call_id, c.call_digest, c.created_at) for c in source.calls
                    ]:
                        raise ValueError("comparison replay input changed")
                    row = {
                        "policy_name": policy,
                        "seed": seed,
                        "profile_seed": profile_seed,
                        "status": "evaluated",
                        "trace_path": str(run_dir / "trace.json"),
                        "input_fingerprint": run.input_fingerprint,
                        "objective_vector": run.evaluation.objective_vector,
                        "weighted_cost": run.evaluation.weighted_cost,
                        "metrics": run.evaluation.to_dict()["metrics"],
                        "decision_count": run.decision_count,
                        "decision_time_mean_sec": run.decision_time_mean_sec,
                        "decision_time_total_sec": run.decision_time_total_sec,
                    }
                except (ValueError, KeyError, TypeError) as exc:
                    row = {
                        "policy_name": policy,
                        "seed": seed,
                        "profile_seed": profile_seed,
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "call_ids": manifest["call_ids"],
                    }
                    _write_json(run_dir / "failure.json", row)
                rows.append(row)
    evaluated = [row for row in rows if row["status"] == "evaluated"]
    flags = pareto_flags([row["objective_vector"] for row in evaluated])
    for row, flag in zip(evaluated, flags, strict=True):
        row["is_pareto"] = flag
    aggregates = aggregate_results(evaluated)
    complete = len(evaluated) == len(rows)
    comparisons = compare_rl(aggregates, evaluated, statistics) if complete else []
    result = {
        "experiment_id": config["experiment_id"],
        "complete": complete,
        "run_count": len(rows),
        "failed_run_count": len(rows) - len(evaluated),
        "manifest_path": str(directory / "manifest.json"),
        "training_wall_times": training_summary,
        "runs": rows,
        "policy_statistics": aggregates,
        "rl_comparisons": comparisons,
        "superiority_claim": False,
        "interpretation": "Report all objective regressions; significance is conditional on this "
        "fixed synthetic workload. No general baseline superiority or hardware claim is made.",
    }
    _write_json(directory / "summary.json", result)
    _write_csv(directory / "runs.csv", rows)
    _write_csv(directory / "policy_statistics.csv", aggregates)
    _write_csv(directory / "rl_comparisons.csv", comparisons)
    _write_json(
        directory / "pareto_frontier.json",
        {
            "run_points": [row for row in evaluated if row["is_pareto"]],
            "policy_mean_points": [row for row in aggregates if row["is_pareto"]],
            "dominance_uses": list(OBJECTIVE_NAMES),
            "weighted_cost_used": False,
        },
    )
    return result


def aggregate_results(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["policy_name"], []).append(row)
    results = []
    for policy, group in groups.items():
        vector = {
            name: fmean(row["objective_vector"][name] for row in group) for name in OBJECTIVE_NAMES
        }
        latencies = [row["objective_vector"]["latency_sec"] for row in group]
        counts = {}
        for row in group:
            for target, count in row["metrics"]["target_selection_counts"].items():
                counts[target] = counts.get(target, 0) + count
        results.append(
            {
                "policy_name": policy,
                "run_count": len(group),
                "objective_vector": vector,
                "objective_std": {
                    name: pstdev(row["objective_vector"][name] for row in group)
                    for name in OBJECTIVE_NAMES
                },
                "p95_latency_sec": float(np.quantile(latencies, 0.95)),
                "p99_latency_sec": float(np.quantile(latencies, 0.99)),
                "success_rate": fmean(row["metrics"]["success_rate"] for row in group),
                "throughput_runs_per_sec": len(group)
                / sum(row["metrics"]["evaluation_window_sec"] for row in group),
                "target_selection_counts": counts,
                "decision_time_mean_sec": sum(row["decision_time_total_sec"] for row in group)
                / sum(row["decision_count"] for row in group),
                "weighted_cost": fmean(row["weighted_cost"] for row in group),
            }
        )
    flags = pareto_flags([row["objective_vector"] for row in results])
    for row, flag in zip(results, flags, strict=True):
        row["is_pareto"] = flag
    return results


def paired_sign_flip_pvalue(differences):
    """Exact one-sided paired randomization test; negative means RL improves."""
    observed = fmean(differences)
    null_means = [
        fmean(sign * value for sign, value in zip(signs, differences, strict=True))
        for signs in itertools.product((-1, 1), repeat=len(differences))
    ]
    return sum(value <= observed + 1e-12 for value in null_means) / len(null_means)


def compare_rl(aggregates, rows, protocol):
    by_policy = {row["policy_name"]: row for row in aggregates}
    rl = by_policy["double_dqn"]
    profile_seeds = sorted({row["profile_seed"] for row in rows})
    results = []
    for policy in DEFAULT_BASELINE_POLICIES:
        baseline = by_policy[policy]
        delta = {
            name: rl["objective_vector"][name] - baseline["objective_vector"][name]
            for name in OBJECTIVE_NAMES
        }
        regressions = [
            name
            for name, value in delta.items()
            if (value < -1e-12 if name == "quality" else value > 1e-12)
        ]
        for name in (
            "p95_latency_sec",
            "p99_latency_sec",
            "decision_time_mean_sec",
            "success_rate",
            "throughput_runs_per_sec",
        ):
            delta[name] = rl[name] - baseline[name]
            if (
                delta[name] < -1e-12
                if name in {"success_rate", "throughput_runs_per_sec"}
                else delta[name] > 1e-12
            ):
                regressions.append(name)
        differences = []
        for profile_seed in profile_seeds:

            def mean_cost(name, noise_seed=profile_seed):
                return fmean(
                    row["weighted_cost"]
                    for row in rows
                    if row["policy_name"] == name and row["profile_seed"] == noise_seed
                )

            differences.append(mean_cost("double_dqn") - mean_cost(policy))
        pvalue = paired_sign_flip_pvalue(differences)
        results.append(
            {
                "baseline": policy,
                "rl_minus_baseline": delta,
                "regressions": regressions,
                "weighted_cost_delta": rl["weighted_cost"] - baseline["weighted_cost"],
                "paired_profile_seed_differences": differences,
                "independent_pair_count": len(differences),
                "pvalue": pvalue,
                "adjusted_pvalue": min(pvalue * 8, 1.0),
                "significant_cost_improvement": fmean(differences) < 0
                and pvalue * 8 < protocol["alpha"],
            }
        )
    return results


def _seeds(values):
    if (
        not values
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values
        )
        or len(set(values)) != len(values)
    ):
        raise ValueError("seeds must be distinct non-negative integers")
    return tuple(values)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def _write_csv(path, rows):
    flat = [
        {
            key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
            for key, value in row.items()
        }
        for row in rows
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in flat for key in row}))
        writer.writeheader()
        writer.writerows(flat)
