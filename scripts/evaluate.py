"""Evaluate one or more comparable experiment trace bundles."""

from __future__ import annotations

import argparse
from pathlib import Path

from edge_agent_workflow_scheduling.profiler import (
    evaluate_traces,
    load_trace_bundle,
    write_evaluation_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trace_paths",
        nargs="+",
        type=Path,
        help="TraceBundle JSON files from one comparable experiment",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/evaluation"),
        help="Directory for summary.json and CSV outputs",
    )
    args = parser.parse_args()

    evaluation = evaluate_traces([load_trace_bundle(path) for path in args.trace_paths])
    write_evaluation_artifacts(evaluation, args.output_dir)
    print(
        f"evaluated {len(evaluation.agent_runs)} run(s), "
        f"{len(evaluation.call_evaluations)} call(s) -> {args.output_dir}"
    )


if __name__ == "__main__":
    main()
