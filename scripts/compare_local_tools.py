#!/usr/bin/env python3
"""Simple real-Tool scheduling smoke on all eight local replicas."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from statistics import fmean
from time import perf_counter

from edge_agent_workflow_scheduling.common import CallStatus
from edge_agent_workflow_scheduling.executors.local_deployment import LocalToolDeployment
from edge_agent_workflow_scheduling.profiler.local_deployment import (
    load_deployment_samples,
    sample_call,
    self_check_deployment,
)
from edge_agent_workflow_scheduling.profiler.trace import JsonlTraceLogger, build_tool_trace_record
from edge_agent_workflow_scheduling.resources import ResourceRegistry
from edge_agent_workflow_scheduling.rl import DoubleDQNAgent, DQNConfig, SchedulingEnv
from edge_agent_workflow_scheduling.rl.comparison import DQNScheduler
from edge_agent_workflow_scheduling.rl.runner import train_agent
from edge_agent_workflow_scheduling.scheduler import BaselineScheduler
from edge_agent_workflow_scheduling.tools.deployment import canonical_digest, canonical_tool_output


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_calls(samples, count, run_id):
    # Interleave Tool types; all policies receive the same fixture order in one burst.
    return [
        sample_call(sample, run_id=run_id, suffix=str(index))
        for index in range(count)
        for sample in samples.values()
    ]


class TimedScheduler:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.times = []

    def schedule(self, call, *, resources):
        started = perf_counter()
        decision = self.scheduler.schedule(call, resources=resources)
        self.times.append(perf_counter() - started)
        return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, default=Path("configs/tool_profiles.toml"))
    parser.add_argument(
        "--samples", type=Path, default=Path("configs/tool_deployment_samples_v1.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/tool_scheduler_smoke"))
    parser.add_argument("--calls-per-tool", type=int, default=8)
    parser.add_argument("--train-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=57)
    args = parser.parse_args()
    if args.calls_per_tool < 1 or args.train_episodes < 1:
        parser.error("calls and training episodes must be positive")
    directory = args.output_dir.resolve()
    samples = load_deployment_samples(args.samples)
    report = {
        "scope": "single_burst_real_tool_smoke_not_formal_experiment",
        "deployment_kind": "same_host_logical_replica",
        "calls_per_tool": args.calls_per_tool,
        "seed": args.seed,
        "cloud_calls": False,
        "llm_calls": False,
        "energy_measurement": None,
        "task_quality_measurement": None,
        "rl_training_scope": "short_latency_only_training_on_measured_startup_fixture_profiles",
        "rl_evaluation_epsilon": 0.0,
        "policy_order": ["round_robin", "least_queue", "earliest_finish_time", "double_dqn"],
        "policy_statistics": [],
    }
    with LocalToolDeployment.from_catalog(
        args.profiles,
        input_root=args.samples.parent,
        output_dir=directory / "deployment",
        timeout_sec=30.0,
    ) as deployment:
        counts = Counter(w.profile.tool_name for w in deployment.workers.values())
        if set(counts) != set(samples) or any(count != 2 for count in counts.values()):
            raise ValueError("smoke requires all four Tool types with two replicas each")
        report["replica_counts"] = dict(counts)
        report["replica_ids"] = sorted(deployment.workers)
        checks, _ = self_check_deployment(deployment, samples)
        write_json(directory / "startup.json", deployment.startup)
        write_json(directory / "self_check.json", checks)
        if not checks["passed"]:
            write_json(directory / "summary.json", {**report, "passed": False})
            raise SystemExit("real Tool startup/self-check failed")
        initial_states = deployment.resources.snapshot_states()
        profiles = [
            {"profile": s.profile.to_dict(), "state": s.state.to_dict()}
            for s in deployment.resources.tool_snapshots()
        ]
        write_json(directory / "resource_profiles.json", {"tool_replicas": profiles})
        training_resources = ResourceRegistry()
        for snapshot in deployment.resources.tool_snapshots():
            training_resources.register_tool_replica(
                snapshot.profile, replace(snapshot.state, queue_len=0, running_tasks=0)
            )
        training_calls = make_calls(samples, args.calls_per_tool, "tool-smoke-training")
        env = SchedulingEnv(training_calls, training_resources)
        config = DQNConfig(
            episodes=args.train_episodes,
            batch_size=16,
            warmup_steps=16,
            epsilon_decay_steps=len(training_calls) * args.train_episodes,
        )
        agent = DoubleDQNAgent(
            env.observation_space.shape[0], env.action_space.n, config=config, seed=args.seed
        )
        history = train_agent(env, agent, config.episodes, seed=args.seed)
        checkpoint = directory / "checkpoint.json"
        agent.save(
            checkpoint,
            {
                "action_ids": list(env.target_ids),
                "scope": report["rl_training_scope"],
                "reward": "latency_only",
                "training_episodes": config.episodes,
                "resource_profiles": profiles,
            },
        )
        write_json(directory / "training_history.json", history)
        agent, metadata = DoubleDQNAgent.load(checkpoint)
        if metadata["action_ids"] != list(env.target_ids):
            raise ValueError("restored action IDs do not match the deployed replicas")
        digests_by_tool = {name: set() for name in samples}
        for policy in report["policy_order"]:
            deployment.resources.restore_states(initial_states)
            calls = make_calls(samples, args.calls_per_tool, f"tool-smoke-{policy}")
            if policy == "double_dqn":
                scheduler = DQNScheduler(
                    SchedulingEnv(calls, deployment.resources),
                    agent,
                    {"path": str(checkpoint)},
                    allow_missing_optional_profiles=True,
                )
            else:
                scheduler = BaselineScheduler(policy)
            scheduler = TimedScheduler(scheduler)
            started = perf_counter()
            submissions = deployment.submit_batch(calls, scheduler)
            rows = []
            for call, decision, future in submissions:
                result = future.result()
                call.transition_to(CallStatus.SUCCEEDED if result.success else CallStatus.FAILED)
                rows.append((call, decision, result))
            elapsed = perf_counter() - started
            # Artifact consistency is checked after the execution timer stops.
            for call, _, result in rows:
                if result.success:
                    canonical = canonical_tool_output(
                        call.tool_name,
                        result.output,
                        artifact_root=deployment.workers[result.replica_id].output_dir,
                    )
                    digests_by_tool[call.tool_name].add(canonical_digest(canonical))
            selected = Counter(decision.selected_target for _, decision, _ in rows)
            statistics = {
                "policy": policy,
                "call_count": len(rows),
                "success_count": sum(result.success for _, _, result in rows),
                "burst_wall_time_sec": elapsed,
                "mean_queue_wait_sec": fmean(result.queue_wait_time_sec for _, _, result in rows),
                "mean_execution_time_sec": fmean(
                    result.execution_time_sec for _, _, result in rows
                ),
                "mean_scheduler_time_sec": fmean(scheduler.times),
                "available_replica_count": len(deployment.workers),
                "selected_replica_count": len(selected),
                "selection_counts": {
                    target: selected[target] for target in sorted(deployment.workers)
                },
                "calls_by_tool": dict(Counter(call.tool_name for call, _, _ in rows)),
                "failures": [result.to_dict() for _, _, result in rows if not result.success],
            }
            report["policy_statistics"].append(statistics)
            output = directory / policy
            logger = JsonlTraceLogger(output / "trace.jsonl")
            logger.clear()
            write_json(
                output / "execution_records.json",
                [
                    {
                        "call": call.to_dict(),
                        "decision": decision.to_dict(),
                        "result": result.to_dict(),
                    }
                    for call, decision, result in rows
                ],
            )
            for call, decision, result in rows:
                logger.write(
                    build_tool_trace_record(tool_call=call, decision=decision, result=result)
                )
            write_json(output / "summary.json", statistics)
            print(
                f"{policy}: success={statistics['success_count']}/{len(rows)} "
                f"replicas={len(selected)}/8 burst={elapsed:.3f}s"
            )
        report["output_consistency"] = {
            name: len(digests) == 1 for name, digests in digests_by_tool.items()
        }
        report["passed"] = all(
            row["success_count"] == row["call_count"] for row in report["policy_statistics"]
        ) and all(report["output_consistency"].values())
        write_json(directory / "state_events.json", deployment.events)
    report["workers_stopped"] = all(
        slot.process.poll() is not None
        for worker in deployment.workers.values()
        for slot in worker.slots
    )
    write_json(directory / "summary.json", report)
    print(f"passed={report['passed']} summary={directory / 'summary.json'}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
