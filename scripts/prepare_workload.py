"""Validate a versioned workload and save a deterministic arrival plan (no execution)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from edge_agent_workflow_scheduling.common import WorkloadConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--scenario", default="low_load")
    parser.add_argument("--split", choices=("calibration", "validation"), default="calibration")
    parser.add_argument("--output", type=Path, default=Path("data/workload_v1/plan.json"))
    args = parser.parse_args()
    workload = WorkloadConfig.from_json(args.config)
    plan = workload.generate(args.scenario, split=args.split, artifact_root=args.config.parent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(f"Prepared {len(plan['requests'])} Agent requests (not executed) -> {args.output}")


if __name__ == "__main__":
    main()
