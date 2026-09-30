import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from edge_agent_workflow_scheduling.profiler.baseline_experiment import (
    DEFAULT_BASELINE_POLICIES,
    _execute_batch,
    _ProfileExecutorPool,
    run_replay_policy,
)
from edge_agent_workflow_scheduling.profiler.replay import (
    load_trace_bundle,
    reconstruct_call,
    resources_from_manifest,
)
from edge_agent_workflow_scheduling.rl.comparison import (
    load_comparison_checkpoint,
    paired_sign_flip_pvalue,
    run_rl_comparison,
)
from edge_agent_workflow_scheduling.scheduler.objectives import (
    ObjectiveNormalization,
    ObjectiveWeights,
)

ROOT = Path(__file__).resolve().parents[1]


def inputs():
    config = json.loads((ROOT / "configs/rl_comparison_v1.json").read_text())
    config["training"] = {"episodes": 2, "batch_size": 1, "warmup_steps": 1}
    config["profile_seeds"] = [0, 1]
    profiles = json.loads((ROOT / config["resource_profile"]).read_text())["resource_profiles"]
    return config, profiles


def run(tmp_path, config, profiles, **kwargs):
    return run_rl_comparison(
        ROOT / "configs/rl_comparison_trace_v1.json",
        output_dir=tmp_path,
        config=config,
        resource_profiles=profiles,
        code_version="test-v1",
        **kwargs,
    )


def deterministic_rows(result):
    return [
        {
            key: row[key]
            for key in (
                "policy_name",
                "seed",
                "profile_seed",
                "input_fingerprint",
                "objective_vector",
                "weighted_cost",
                "metrics",
                "decision_count",
                "is_pareto",
            )
        }
        for row in result["runs"]
    ]


def test_all_policies_share_inputs_noise_and_restored_checkpoints(tmp_path):
    config, profiles = inputs()
    first = run(tmp_path / "first", config, profiles)
    restored = run(
        tmp_path / "restored", config, profiles, checkpoint_dir=tmp_path / "first" / "training"
    )
    repeat = run(tmp_path / "repeat", config, profiles)
    assert first["complete"] and first["run_count"] == 36
    assert len(first["policy_statistics"]) == 9
    assert len(first["rl_comparisons"]) == 8
    assert {row["policy_name"] for row in first["runs"]} == {
        *DEFAULT_BASELINE_POLICIES,
        "double_dqn",
    }
    assert len({row["input_fingerprint"] for row in first["runs"]}) == 1
    assert deterministic_rows(first) == deterministic_rows(restored) == deterministic_rows(repeat)
    for seed in config["scheduler_seeds"]:
        a = json.loads((tmp_path / f"first/training/seed-{seed}/checkpoint.json").read_text())
        b = json.loads((tmp_path / f"repeat/training/seed-{seed}/checkpoint.json").read_text())
        assert a == b
    for noise_seed in (0, 1):
        eft = [
            row
            for row in first["runs"]
            if row["policy_name"] == "earliest_finish_time" and row["profile_seed"] == noise_seed
        ]
        assert eft[0]["metrics"] == eft[1]["metrics"]
    assert all(row["independent_pair_count"] == 2 for row in first["rl_comparisons"])
    # Overhead is wall-clock data, excluded from reproducibility assertions.
    assert all(row["decision_time_mean_sec"] > 0 for row in first["runs"])
    for row in first["runs"]:
        trace = load_trace_bundle(row["trace_path"])
        decisions = json.loads(Path(row["trace_path"]).with_name("decisions.json").read_text())
        assert len(trace.calls) == 7
        for decision in decisions["decisions"]:
            targets = decision["candidate_target_ids"]
            assert decision["action_mask"][targets.index(decision["selected_target"])]
    with pytest.raises(ValueError, match="metadata mismatch: action_ids"):
        payload = json.loads((tmp_path / "first/training/seed-0/checkpoint.json").read_text())
        metadata = deepcopy(payload["metadata"])
        metadata["action_ids"].reverse()
        load_comparison_checkpoint(tmp_path / "first/training/seed-0/checkpoint.json", metadata)


def test_execution_failures_remain_in_metrics_and_trace(tmp_path):
    config, profiles = inputs()
    config["profile_failure_rate"] = 1.0
    result = run(tmp_path, config, profiles)
    assert result["complete"]
    assert all(row["metrics"]["success_rate"] == 0 for row in result["runs"])
    for row in result["runs"]:
        calls = load_trace_bundle(row["trace_path"]).calls
        assert len(calls) == 7
        assert all(not call.success and call.error_code == "profile_failure" for call in calls)
    # These are failed executions, not omitted experiments.
    assert result["failed_run_count"] == 0


def test_infeasible_input_and_missing_metrics_fail_before_training(tmp_path):
    config, profiles = inputs()
    offline = deepcopy(profiles)
    for resource in offline["llm_instances"]:
        resource["state"]["is_online"] = False
    with pytest.raises(ValueError, match="no feasible"):
        run(tmp_path / "offline", config, offline)
    failure = json.loads((tmp_path / "offline/preflight_failure.json").read_text())
    assert not failure["executor_dispatched"] and len(failure["call_ids"]) == 7
    missing = deepcopy(profiles)
    missing["llm_instances"][0]["profile"]["energy_profile"] = {}
    with pytest.raises(ValueError, match="joules_per_token"):
        run(tmp_path / "missing", config, missing)
    assert (tmp_path / "missing/preflight_failure.json").exists()
    assert not (tmp_path / "missing/training").exists()


def test_invalid_scheduler_target_is_rejected_before_executor(tmp_path):
    config, _ = inputs()
    source = load_trace_bundle(ROOT / config["trace"])

    class InvalidScheduler:
        def schedule(self, call, *, resources):
            return SimpleNamespace(selected_target="image-profile")

    with pytest.raises(ValueError, match="masked target before Executor"):
        run_replay_policy(
            source,
            policy_name="invalid",
            seed=0,
            profile_seed=0,
            profile_jitter_ratio=0.0,
            profile_failure_rate=0.0,
            objective_weights=ObjectiveWeights(**config["objective_weights"]),
            objective_normalization=ObjectiveNormalization(**config["objective_normalization"]),
            min_quality=None,
            experiment_id="invalid",
            output_dir=tmp_path,
            scheduler_factory=lambda resources: InvalidScheduler(),
        )
    assert not (tmp_path / "trace.json").exists()


def test_profile_noise_is_call_target_specific_and_batch_capacity_is_respected():
    config, _ = inputs()
    source = load_trace_bundle(ROOT / config["trace"])
    resources = resources_from_manifest(source)
    tool_source = source.calls[1]
    call = reconstruct_call(tool_source)
    pool = _ProfileExecutorPool(profile_seed=3, jitter_ratio=0.1, failure_rate=0.0)
    first = pool.execute(call, "image-profile", resources)
    pool.execute(replace(call, tool_call_id="other-call"), "image-profile", resources)
    second = pool.execute(call, "image-profile", resources)
    assert first.execution_time_sec == second.execution_time_sec
    pool = _ProfileExecutorPool(profile_seed=0, jitter_ratio=0.0, failure_rate=0.0)
    assignments = [(tool_source, call, SimpleNamespace(selected_target="image-profile"))] * 3
    results = _execute_batch(pool, assignments, resources)
    assert [result.queue_wait_time_sec for result in results] == pytest.approx([0, 0.04, 0.08])
    concurrent_source = deepcopy(source)
    concurrent_source.manifest.resource_profiles["tool_replicas"][0]["profile"][
        "max_concurrency"
    ] = 2
    concurrent_resources = resources_from_manifest(concurrent_source)
    results = _execute_batch(pool, assignments, concurrent_resources)
    assert [result.queue_wait_time_sec for result in results] == pytest.approx([0, 0, 0.04])


def test_paired_statistics_and_tradeoff_reporting():
    assert paired_sign_flip_pvalue([-1.0, -2.0]) == 0.25
    assert paired_sign_flip_pvalue([0.0, 0.0]) == 1.0
    assert paired_sign_flip_pvalue([1.0, 2.0]) == 1.0
