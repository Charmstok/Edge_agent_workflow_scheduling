#!/usr/bin/env python3
"""Run the preregistered offline RL comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from edge_agent_workflow_scheduling.profiler.pareto import load_resource_profile_set
from edge_agent_workflow_scheduling.rl.comparison import run_rl_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/rl_comparison_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/rl_comparison"))
    parser.add_argument(
        "--checkpoint-dir", type=Path, help="Evaluate saved checkpoints without training"
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    version, profiles = load_resource_profile_set(config["resource_profile"])
    if version != config["resource_profile_version"]:
        raise ValueError("resource profile version mismatch")
    # Capture uncommitted source as well as committed code; stable across output directories.
    digest = hashlib.sha256()
    for path in sorted(Path("src").rglob("*.py")) + [Path(__file__)]:
        digest.update(str(path.relative_to(Path.cwd()) if path.is_absolute() else path).encode())
        digest.update(path.read_bytes())
    result = run_rl_comparison(
        config["trace"],
        output_dir=args.output_dir,
        config=config,
        resource_profiles=profiles,
        code_version=f"source-sha256:{digest.hexdigest()}",
        checkpoint_dir=args.checkpoint_dir,
    )
    print(
        f"evaluated {result['run_count']} runs; failures={result['failed_run_count']}; "
        f"summary={args.output_dir / 'summary.json'}"
    )
    if not result["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
