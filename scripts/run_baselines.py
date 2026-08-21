"""Run reproducible baseline scheduler comparisons on a fixed replay trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from edge_agent_workflow_scheduling.profiler import (
    DEFAULT_BASELINE_POLICIES,
    run_baseline_experiment,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_path", type=Path, help="Fixed TraceBundle JSON input")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/milestone_3_6"),
        help="Root directory for versioned experiment artifacts",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=list(DEFAULT_BASELINE_POLICIES),
        help="Registered policy names to compare",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0],
        help="Scheduler seeds; use multiple values for random policies",
    )
    parser.add_argument("--profile-seed", type=int, default=0)
    parser.add_argument("--profile-jitter-ratio", type=float, default=0.0)
    parser.add_argument("--profile-failure-rate", type=float, default=0.0)
    parser.add_argument(
        "--resource-profile",
        type=Path,
        help="Optional JSON file containing resource_profiles or the resource profile mapping",
    )
    parser.add_argument(
        "--objective-weights",
        type=Path,
        help=(
            "Optional JSON object/file with latency, energy, deadline_miss, "
            "load_imbalance, quality"
        ),
    )
    parser.add_argument(
        "--objective-normalization",
        type=Path,
        help="Optional JSON object/file with latency_ref_sec and energy_ref_joules",
    )
    parser.add_argument("--experiment-id")
    args = parser.parse_args()

    resource_profiles = _read_optional_object(args.resource_profile)
    weights = _read_optional_object(args.objective_weights)
    normalization = _read_optional_object(args.objective_normalization)
    result = run_baseline_experiment(
        args.trace_path,
        output_dir=args.output_dir,
        policies=args.policies,
        seeds=args.seeds,
        profile_seed=args.profile_seed,
        profile_jitter_ratio=args.profile_jitter_ratio,
        profile_failure_rate=args.profile_failure_rate,
        resource_profiles=(
            resource_profiles.get("resource_profiles", resource_profiles)
            if resource_profiles is not None
            else None
        ),
        objective_weights=weights,
        objective_normalization=normalization,
        experiment_id=args.experiment_id,
    )
    print(
        f"ran {len(result.runs)} baseline run(s) for {len(result.policies)} policy(s) "
        f"-> {result.output_dir}"
    )


def _read_optional_object(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


if __name__ == "__main__":
    main()
