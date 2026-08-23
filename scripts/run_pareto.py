"""Scan objective weights and generate a reproducible Pareto frontier."""

from __future__ import annotations

import argparse
from pathlib import Path

from edge_agent_workflow_scheduling.profiler import (
    REFERENCE_POLICIES,
    load_resource_profile_set,
    load_weight_set,
    run_pareto_experiment,
)
from edge_agent_workflow_scheduling.scheduler import ObjectiveNormalization


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_path", type=Path, help="Fixed TraceBundle JSON workload")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/milestone_3_7"),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("configs/pareto_weights_v1.json"),
    )
    parser.add_argument(
        "--resource-profile",
        type=Path,
        default=Path("configs/pareto_resource_profiles_v1.json"),
    )
    parser.add_argument("--latency-ref-sec", type=float, default=0.2)
    parser.add_argument("--energy-ref-joules", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--profile-seed", type=int, default=37)
    parser.add_argument("--profile-jitter-ratio", type=float, default=0.0)
    parser.add_argument("--min-quality", type=float, default=0.8)
    parser.add_argument(
        "--reference-policies",
        nargs="+",
        default=list(REFERENCE_POLICIES),
    )
    parser.add_argument("--experiment-id")
    args = parser.parse_args()

    weight_set = load_weight_set(args.weights)
    profile_version, resource_profiles = load_resource_profile_set(args.resource_profile)
    result = run_pareto_experiment(
        args.trace_path,
        output_dir=args.output_dir,
        weight_set=weight_set,
        objective_normalization=ObjectiveNormalization(
            latency_ref_sec=args.latency_ref_sec,
            energy_ref_joules=args.energy_ref_joules,
        ),
        resource_profiles=resource_profiles,
        resource_profile_version=profile_version,
        reference_policies=args.reference_policies,
        random_seeds=args.random_seeds,
        seed=args.seed,
        profile_seed=args.profile_seed,
        profile_jitter_ratio=args.profile_jitter_ratio,
        min_quality=args.min_quality,
        experiment_id=args.experiment_id,
    )
    print(
        f"generated {len(result.points)} points and {len(result.pareto_points)} "
        f"Pareto points -> {result.output_dir}"
    )
    print(result.weight_scan_note)


if __name__ == "__main__":
    main()
