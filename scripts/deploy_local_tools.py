#!/usr/bin/env python3
"""Start all eight real Tool replicas and validate the local deployment offline."""

from __future__ import annotations

import argparse
import json
import platform
from collections import Counter
from dataclasses import replace
from pathlib import Path

from edge_agent_workflow_scheduling.executors.local_deployment import LocalToolDeployment
from edge_agent_workflow_scheduling.profiler.local_deployment import (
    failure_probes,
    load_deployment_samples,
    run_scheduled_batch,
    self_check_deployment,
)
from edge_agent_workflow_scheduling.profiler.trace import JsonlTraceLogger, build_tool_trace_record
from edge_agent_workflow_scheduling.resources import ResourceRegistry
from edge_agent_workflow_scheduling.rl import DoubleDQNAgent, DQNConfig, SchedulingEnv
from edge_agent_workflow_scheduling.rl.comparison import DQNScheduler
from edge_agent_workflow_scheduling.rl.runner import train_agent
from edge_agent_workflow_scheduling.scheduler import BaselineScheduler


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/local_tool_deployment_v1.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/local_tool_deployment"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    samples = load_deployment_samples(config["consistency_samples"])
    records = []
    directory = args.output_dir.resolve()
    report = {
        "version": config["version"],
        "host": platform.platform(),
        "deployment_kind": "same_host_logical_replica",
        "cloud_calls": False,
        "config": config,
        "energy_source": "unavailable",
        "quality_source": "consistency_only",
        "rl_scope": "latency_only_scheduling_smoke; no_performance_claim",
    }
    with LocalToolDeployment.from_catalog(
        config["tool_profile_catalog"],
        input_root=Path(config["consistency_samples"]).parent,
        output_dir=directory,
        timeout_sec=config["timeout_sec"],
    ) as deployment:
        counts = Counter(worker.profile.tool_name for worker in deployment.workers.values())
        if set(config["required_tools"]) != set(counts) or any(
            count < config["minimum_replicas_per_tool"] for count in counts.values()
        ):
            raise ValueError(
                "deployment must include all four Tools with at least two replicas each"
            )
        report["replica_counts"] = dict(counts)
        report["startup"] = deployment.startup
        _write(directory / "startup.json", deployment.startup)
        checks, group = self_check_deployment(deployment, samples)
        report["self_check"] = checks
        records.extend(group)
        if checks["passed"]:
            for policy in ("round_robin", "least_queue"):
                group = run_scheduled_batch(
                    deployment, samples, BaselineScheduler(policy), run_id=f"local-{policy}-v1"
                )
                records.extend(group)
                selections = Counter(record.decision.selected_target for record in group)
                report[policy] = {
                    "selection_counts": dict(selections),
                    "passed": len(selections) == len(deployment.workers)
                    and all(record.result.success for record in group),
                }
            # Train on measured one-fixture latency profiles; execution during training
            # stays profile-based. Evaluate the restored greedy network on real workers.
            training_resources = ResourceRegistry()
            for snapshot in deployment.resources.tool_snapshots():
                training_resources.register_tool_replica(
                    snapshot.profile, replace(snapshot.state, queue_len=0, running_tasks=0)
                )
            calls = [record.call for record in group]
            env = SchedulingEnv(calls, training_resources, profile_seed=config["rl_smoke_seed"])
            dqn_config = DQNConfig(
                episodes=config["rl_smoke_episodes"],
                batch_size=16,
                warmup_steps=16,
                epsilon_decay_steps=len(calls) * config["rl_smoke_episodes"],
            )
            agent = DoubleDQNAgent(
                env.observation_space.shape[0],
                env.action_space.n,
                config=dqn_config,
                seed=config["rl_smoke_seed"],
            )
            history = train_agent(env, agent, dqn_config.episodes, seed=config["rl_smoke_seed"])
            checkpoint = directory / "rl_checkpoint.json"
            agent.save(
                checkpoint,
                {
                    "environment_schema": "scheduling-env-v1",
                    "action_ids": list(env.target_ids),
                    "deployment_version": config["version"],
                    "reward": "latency_only",
                    "hardware_performance_claim": False,
                },
            )
            agent, _ = DoubleDQNAgent.load(checkpoint)
            _write(directory / "rl_training_history.json", history)
            real_env = SchedulingEnv(calls, deployment.resources)
            scheduler = DQNScheduler(
                real_env, agent, {"path": str(checkpoint)}, allow_missing_optional_profiles=True
            )
            group = run_scheduled_batch(
                deployment, samples, scheduler, run_id="local-double-dqn-v1"
            )
            records.extend(group)
            report["double_dqn"] = {
                "passed": all(record.result.success for record in group),
                "selection_counts": dict(Counter(r.decision.selected_target for r in group)),
            }
            # Explicit mask probes verify both replicas can be chosen by RL, even if
            # the greedy policy prefers one of two equivalent same-host replicas.
            probes = []
            for target, worker in deployment.workers.items():
                others = [
                    snapshot.state
                    for snapshot in deployment.resources.tool_snapshots(
                        tool_name=worker.profile.tool_name
                    )
                    if snapshot.profile.replica_id != target
                ]
                for state in others:
                    deployment.resources.update_tool_state(replace(state, is_online=False))
                try:
                    one_sample = {worker.profile.tool_name: samples[worker.profile.tool_name]}
                    selected = run_scheduled_batch(
                        deployment, one_sample, scheduler, run_id=f"rl-mask-probe-{target}"
                    )
                    records.extend(selected)
                    probes.append(
                        {
                            "target": target,
                            "selection_mode": "single_replica_action_mask_probe",
                            "passed": all(
                                r.decision.selected_target == target and r.result.success
                                for r in selected
                            ),
                        }
                    )
                finally:
                    for state in others:
                        deployment.resources.update_tool_state(state)
            report["rl_mask_probes"] = probes
            if config.get("fault_probes", True):
                faults, group = failure_probes(deployment, samples)
                records.extend(group)
                report["fault_probes"] = faults
        report["online_before_shutdown"] = {
            snapshot.profile.replica_id: snapshot.state.is_online
            for snapshot in deployment.resources.tool_snapshots()
        }
        report["resource_profiles"] = {
            "tool_replicas": [
                {"profile": snapshot.profile.to_dict(), "state": snapshot.state.to_dict()}
                for snapshot in deployment.resources.tool_snapshots()
            ],
            "llm_instances": [],
        }
        report["passed"] = (
            checks["passed"]
            and all(
                report.get(policy, {}).get("passed", False)
                for policy in ("round_robin", "least_queue", "double_dqn")
            )
            and all(item["passed"] for item in report.get("rl_mask_probes", []))
            and all(item["passed"] for item in report.get("fault_probes", {}).values())
        )
    report["workers_stopped_after_validation"] = all(
        slot.process.poll() is not None
        for worker in deployment.workers.values()
        for slot in worker.slots
    )
    logger = JsonlTraceLogger(directory / "trace.jsonl")
    logger.clear()
    with (directory / "execution_records.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            logger.write(
                build_tool_trace_record(
                    tool_call=record.call,
                    decision=record.decision,
                    result=record.result,
                    timeout=record.result.error_code == "timeout",
                )
            )
            stream.write(
                json.dumps(
                    {
                        "call": record.call.to_dict(),
                        "decision": record.decision.to_dict(),
                        "result": record.result.to_dict(),
                    }
                )
                + "\n"
            )
    _write(directory / "state_events.json", deployment.events)
    _write(directory / "resource_profiles.json", report["resource_profiles"])
    report["record_count"] = len(records)
    _write(directory / "summary.json", report)
    print(f"replica_counts={dict(counts)} passed={report['passed']} records={len(records)}")
    print(f"summary={directory / 'summary.json'}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
