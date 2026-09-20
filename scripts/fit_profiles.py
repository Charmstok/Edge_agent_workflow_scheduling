"""Fit versioned latency and optional energy profiles from saved observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from edge_agent_workflow_scheduling.profiler.profile_fitting import (
    SyntheticEnergyConfig,
    fit_profile_catalog,
    write_profile_catalog,
)


def _profile_version(value: str) -> str:
    if not value or Path(value).name != value or value in {".", ".."}:
        raise argparse.ArgumentTypeError("profile version must be one path-safe name")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tool-sampling-run",
        type=Path,
        action="append",
        help="Repeat for additional physical devices; profile IDs include the host digest",
    )
    parser.add_argument("--llm-benchmark", type=Path)
    parser.add_argument(
        "--profile-version",
        type=_profile_version,
        default="calibrated-profile-v1",
    )
    parser.add_argument(
        "--fallback-policy",
        choices=("error", "aggregate"),
        default="error",
        help="Behavior when a call has no exact task/size/concurrency bucket",
    )
    parser.add_argument("--synthetic-tool-latency-multiplier", type=float, default=1.75)
    parser.add_argument(
        "--synthetic-energy-config",
        type=Path,
        help="Optional, explicitly synthetic energy assumptions for logical replicas",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Generated catalog path (default: "
            "data/profile_calibration/<profile-version>/profiles.json)"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace the output when reproducing the same profile version",
    )
    args = parser.parse_args()

    energy = None
    if args.synthetic_energy_config:
        value = json.loads(args.synthetic_energy_config.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("synthetic energy config must contain a JSON object")
        energy = SyntheticEnergyConfig.from_mapping(value)

    catalog = fit_profile_catalog(
        profile_version=args.profile_version,
        tool_sampling_run=args.tool_sampling_run,
        llm_benchmark=args.llm_benchmark,
        fallback_policy=args.fallback_policy,
        synthetic_tool_latency_multiplier=args.synthetic_tool_latency_multiplier,
        synthetic_energy=energy,
    )
    output_path = args.output or (
        Path("data/profile_calibration") / args.profile_version / "profiles.json"
    )
    output = write_profile_catalog(catalog, output_path, overwrite=args.overwrite)
    resources = catalog["resource_profiles"]
    print(
        f"wrote {len(resources['tool_replicas'])} Tool and "
        f"{len(resources['llm_instances'])} LLM profiles -> {output}"
    )


if __name__ == "__main__":
    main()
