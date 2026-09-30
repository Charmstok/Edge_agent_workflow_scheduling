#!/usr/bin/env python3
"""Train and evaluate the local Double-DQN scheduling prototype."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from edge_agent_workflow_scheduling.profiler.replay import (
    load_trace_bundle,
    reconstruct_call,
    resources_from_manifest,
)
from edge_agent_workflow_scheduling.rl import DoubleDQNAgent, DQNConfig, SchedulingEnv
from edge_agent_workflow_scheduling.rl.runner import evaluate_agent, train_agent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/rl_training"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile-seed", type=int, default=0)
    args = parser.parse_args()
    trace = load_trace_bundle(args.trace)
    calls = [reconstruct_call(call) for call in trace.calls]
    env = SchedulingEnv(calls, resources_from_manifest(trace), profile_seed=args.profile_seed)
    config = DQNConfig(episodes=args.episodes)
    agent = DoubleDQNAgent(
        env.observation_space.shape[0], env.action_space.n, config=config, seed=args.seed
    )
    history = train_agent(env, agent, args.episodes, seed=args.seed)
    evaluation = evaluate_agent(env, agent, args.eval_episodes, seed=args.seed + args.episodes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "resource_version": trace.manifest.resource_profiles,
        "workload_version": trace.manifest.dataset_id,
        "source_trace": str(args.trace),
        "profile_seed": args.profile_seed,
        "environment_schema": "scheduling-env-v1",
        "observation_size": env.observation_space.shape[0],
        "action_ids": list(env.target_ids),
    }
    agent.save(args.output_dir / "checkpoint.json", metadata)
    (args.output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2) + "\n", encoding="utf-8"
    )
    print(f"trained {args.episodes} episodes; checkpoint={args.output_dir / 'checkpoint.json'}")


if __name__ == "__main__":
    main()
