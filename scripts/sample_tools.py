"""Collect reproducible local Tool timing and resource samples for profile calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from edge_agent_workflow_scheduling.profiler.tool_sampling import (
    ToolSamplingConfig,
    run_tool_sampling,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/tool_sampling_v1.json"),
        help="Versioned sampling matrix",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/tool_sampling"),
        help="Root for immutable sampling runs",
    )
    parser.add_argument(
        "--experiment-id",
        help="Optional stable run ID; the output directory must not already exist",
    )
    args = parser.parse_args()
    config = ToolSamplingConfig.from_json(args.config)
    output = run_tool_sampling(
        config,
        config_path=args.config,
        output_root=args.output_dir,
        experiment_id=args.experiment_id,
    )
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    print(
        f"Collected {summary['measurement_call_count']} measured calls across "
        f"{summary['completed_combination_count']} configurations -> {output}"
    )
    if summary["skipped_combination_count"]:
        print(f"Skipped configurations: {summary['skipped_combination_count']}")


if __name__ == "__main__":
    main()
