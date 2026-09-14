"""Score Agent traces and build task-specific quality profiles."""

from __future__ import annotations

import argparse
from pathlib import Path

from edge_agent_workflow_scheduling.common import WorkloadConfig
from edge_agent_workflow_scheduling.config import load_llm_profiles
from edge_agent_workflow_scheduling.profiler import (
    QualityCalibrationConfig,
    apply_quality_report,
    build_quality_report,
    evaluate_traces,
    load_trace_bundle,
    score_trace_bundles,
    write_evaluation_artifacts,
    write_quality_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_paths", nargs="*", type=Path)
    parser.add_argument(
        "--trace-root",
        action="append",
        default=[],
        type=Path,
        help="Recursively include files named trace.json below this directory",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/quality_calibration_v1.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/quality_scoring"))
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = QualityCalibrationConfig.from_json(config_path)
    workload_path = _resolve(config_path.parent, config.workload_config)
    catalog_path = _resolve(config_path.parent, config.llm_catalog)
    workload = WorkloadConfig.from_json(workload_path)

    paths = {path.resolve() for path in args.trace_paths}
    for root in args.trace_root:
        paths.update(path.resolve() for path in root.rglob("trace.json"))
    ordered_paths = sorted(paths)
    if not ordered_paths:
        parser.error("provide trace_paths or --trace-root containing trace.json files")

    traces = [load_trace_bundle(path) for path in ordered_paths]
    scores = score_trace_bundles(
        traces,
        workload,
        trace_refs=[str(path) for path in ordered_paths],
    )
    report = build_quality_report(scores, workload, config)
    profiles = apply_quality_report(load_llm_profiles(catalog_path), report)
    write_quality_artifacts(args.output_dir, scores, report, profiles)
    evaluation = evaluate_traces(traces, task_scores=scores)
    write_evaluation_artifacts(evaluation, args.output_dir / "evaluation")
    print(
        f"scored {len(scores)} AgentRun(s), calibrated "
        f"{len(report['profiles'])} model profile(s) -> {args.output_dir}"
    )


def _resolve(parent: Path, configured: str) -> Path:
    path = Path(configured)
    return path if path.is_absolute() else parent / path


if __name__ == "__main__":
    main()
