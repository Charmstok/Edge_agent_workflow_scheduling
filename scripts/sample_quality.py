"""Run comparable live Agent tasks and calibrate task-specific LLM quality."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from edge_agent_workflow_scheduling.agents import AgentRunner
from edge_agent_workflow_scheduling.common import WorkloadConfig
from edge_agent_workflow_scheduling.config import load_llm_profiles
from edge_agent_workflow_scheduling.executors import (
    ExecutorFactoryRegistry,
    ExecutorPool,
    LocalToolExecutor,
    create_openai_chat_executor,
    create_openai_responses_executor,
)
from edge_agent_workflow_scheduling.profiler import (
    QualityCalibrationConfig,
    TraceBundle,
    TraceBundleStore,
    apply_quality_report,
    build_experiment_manifest,
    build_quality_report,
    build_trace_bundle,
    evaluate_traces,
    score_trace_bundles,
    write_evaluation_artifacts,
    write_quality_artifacts,
)
from edge_agent_workflow_scheduling.resources import (
    ResourceRegistry,
    ToolReplicaProfile,
)
from edge_agent_workflow_scheduling.scheduler import BaselineScheduler
from edge_agent_workflow_scheduling.tools import (
    DocumentToolConfig,
    ImagePreprocessConfig,
    ImagePreprocessTool,
    OCRConfig,
    OCRTool,
    PDFParseTool,
    PDFRenderConfig,
    PDFRenderTool,
    ToolRegistry,
)
from edge_agent_workflow_scheduling.workers import LocalWorker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/quality_calibration_v1.json"),
    )
    parser.add_argument("--llm-id", action="append", dest="llm_ids")
    parser.add_argument(
        "--split",
        action="append",
        choices=("calibration", "validation"),
        dest="splits",
    )
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("data/quality_sampling"))
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = QualityCalibrationConfig.from_json(config_path)
    workload_path = _resolve(config_path.parent, config.workload_config)
    catalog_path = _resolve(config_path.parent, config.llm_catalog)
    workload = WorkloadConfig.from_json(workload_path)
    profiles = {profile.llm_id: profile for profile in load_llm_profiles(catalog_path)}
    selected_ids = tuple(args.llm_ids or config.llm_ids)
    unknown = sorted(set(selected_ids) - set(profiles))
    if unknown:
        raise ValueError(f"unknown llm_ids: {unknown}")
    repeats = args.repeats or config.repeats
    if repeats < 1:
        raise ValueError("repeats must be positive")
    splits = tuple(args.splits or ("calibration", "validation"))
    run_dir = args.output_dir / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=False)

    traces: list[TraceBundle] = []
    trace_refs: list[str] = []
    skipped: list[dict[str, str]] = []
    tool_reason = _tool_runtime_unavailable_reason(workload)
    if tool_reason is not None:
        summary = {
            "schema_version": 1,
            "quality_profile_version": config.quality_profile_version,
            "requested_llm_ids": list(selected_ids),
            "completed_trace_count": 0,
            "skipped_models": [],
            "skipped_runtime": tool_reason,
            "splits": list(splits),
            "repeats": repeats,
        }
        _write_json(run_dir / "sampling_summary.json", summary)
        print(f"quality sampling skipped: {tool_reason} -> {run_dir}")
        return
    for llm_id in selected_ids:
        profile = profiles[llm_id]
        reason = _unavailable_reason(profile)
        if reason is not None:
            skipped.append({"llm_id": llm_id, "reason": reason})
            continue
        parameters = dict(profile.deployment_config.get("model_parameters", {}))
        parameters.update({"max_tokens": config.max_output_tokens, "tool_choice": "auto"})
        measured_profile = replace(
            profile,
            deployment_config={**profile.deployment_config, "model_parameters": parameters},
        )
        for split in splits:
            samples = sorted(
                (sample for sample in workload.tasks if sample.split == split),
                key=lambda sample: sample.task_id,
            )
            for sample in samples:
                for repeat in range(repeats):
                    trace_path, trace = _run_sample(
                        workload=workload,
                        workload_path=workload_path,
                        profile=measured_profile,
                        sample=sample,
                        repeat=repeat,
                        run_dir=run_dir,
                        config=config,
                    )
                    traces.append(trace)
                    trace_refs.append(str(trace_path.resolve()))
                    print(
                        f"{llm_id} {sample.task_id} repeat={repeat + 1}: "
                        f"{trace.run.status} ({trace.run.end_to_end_latency_sec:.3f}s)"
                    )

    summary = {
        "schema_version": 1,
        "quality_profile_version": config.quality_profile_version,
        "requested_llm_ids": list(selected_ids),
        "completed_trace_count": len(traces),
        "skipped_models": skipped,
        "splits": list(splits),
        "repeats": repeats,
    }
    _write_json(run_dir / "sampling_summary.json", summary)
    if not traces:
        print(f"no available models; wrote skip summary -> {run_dir}")
        return

    scores = score_trace_bundles(traces, workload, trace_refs=trace_refs)
    report = build_quality_report(scores, workload, config)
    calibrated_profiles = apply_quality_report(list(profiles.values()), report)
    quality_dir = run_dir / "quality"
    write_quality_artifacts(quality_dir, scores, report, calibrated_profiles)
    write_evaluation_artifacts(
        evaluate_traces(traces, task_scores=scores),
        quality_dir / "evaluation",
    )
    print(f"quality calibration -> {quality_dir}")


def _run_sample(*, workload, workload_path, profile, sample, repeat, run_dir, config):
    sample_dir = (
        run_dir
        / profile.llm_id
        / sample.split
        / sample.task_id
        / f"repeat-{repeat + 1:02d}"
    )
    registry = _tool_registry(
        sample.allowed_tools,
        local_root=workload_path.parent,
        output_dir=sample_dir / "tool_outputs",
    )
    resources = ResourceRegistry()
    resources.register_llm(profile)
    for tool_name in registry.supported_tools():
        resources.register_tool_replica(
            ToolReplicaProfile(
                replica_id=f"local-{tool_name}",
                tool_name=tool_name,
                node_id="local-coordinator",
                platform="linux",
                implementation_version="quality-calibration-v1",
                executor_type="local",
                max_concurrency=1,
                metadata={"source_kind": "real_local_tool"},
            )
        )
    factories = ExecutorFactoryRegistry()
    factories.register_llm("openai_chat", create_openai_chat_executor)
    factories.register_llm("openai_responses", create_openai_responses_executor)
    factories.register_tool(
        "local",
        lambda tool_profile: LocalToolExecutor(
            LocalWorker(profile=tool_profile, tool_registry=registry)
        ),
    )
    runner = AgentRunner(
        agent_id="quality-calibration-agent",
        system_instruction=(
            f"{workload.agent.system_prompt} {config.system_prompt_suffix}"
        ),
        tool_registry=registry,
        resources=resources,
        scheduler=BaselineScheduler("least_queue"),
        executor_pool=ExecutorPool(factories),
        max_rounds=workload.agent.max_rounds,
        max_tool_calls=workload.agent.max_tool_calls,
        timeout_sec=workload.agent.timeout_sec,
        model_name=profile.model,
    )
    manifest = build_experiment_manifest(
        experiment_id=f"{config.quality_profile_version}-{profile.llm_id}-{sample.task_id}",
        dataset_id=workload.dataset_id,
        sample_ids=[sample.task_id],
        runner=runner,
        system_prompt_version=config.system_prompt_version,
        user_template=workload.agent.user_template,
        user_template_version=workload.agent.user_template_version,
        llm_profile_version=config.quality_profile_version,
        tool_profile_version="quality-calibration-tools-v1",
        code_version="working-tree",
        mode="live",
        sampling_parameters=profile.deployment_config.get("model_parameters", {}),
        workload_parameters={
            "workload_id": workload.workload_id,
            "workload_version": workload.workload_version,
            "task_type": sample.task_type,
            "split": sample.split,
            "input_size": sample.input_size,
            "scoring_rule": sample.scoring_rule,
            "repeat": repeat + 1,
        },
    )
    execution = runner.run(
        workload.agent.render_user_task(sample),
        task_id=sample.task_id,
        run_id=f"{profile.llm_id}-{sample.task_id}-r{repeat + 1:02d}",
        call_metadata={
            "task_type": sample.task_type,
            "input_size": sample.input_size,
            "data_version": sample.data_version,
        },
    )
    trace = build_trace_bundle(execution, manifest)
    trace_path = sample_dir / "trace.json"
    TraceBundleStore(trace_path).write(trace)
    return trace_path, trace


def _tool_registry(tool_names, *, local_root: Path, output_dir: Path) -> ToolRegistry:
    available = {
        "image_preprocess": ImagePreprocessTool(
            ImagePreprocessConfig(local_root=local_root, output_dir=output_dir / "image")
        ),
        "ocr": OCRTool(
            OCRConfig(
                local_root=local_root,
                output_dir=output_dir / "ocr",
                executable=os.getenv("TESSERACT_EXECUTABLE", "tesseract"),
            )
        ),
        "pdf_parse": PDFParseTool(
            DocumentToolConfig(local_root=local_root, output_dir=output_dir / "pdf_parse")
        ),
        "pdf_render": PDFRenderTool(
            PDFRenderConfig(local_root=local_root, output_dir=output_dir / "pdf_render")
        ),
    }
    registry = ToolRegistry()
    for tool_name in tool_names:
        try:
            registry.register(available[tool_name])
        except KeyError as exc:
            raise ValueError(f"unsupported workload Tool {tool_name!r}") from exc
    return registry


def _unavailable_reason(profile) -> str | None:
    if profile.deployment_config.get("enabled") is False:
        return "deployment_config.enabled is false"
    if "function_calling" not in profile.capabilities:
        return "function_calling capability is not declared"
    missing = [name for name in profile.secret_env_vars if not os.getenv(name)]
    if missing:
        return f"missing environment variables: {', '.join(missing)}"
    if profile.model.startswith("SET_"):
        return "model identifier is unresolved"
    return None


def _tool_runtime_unavailable_reason(workload: WorkloadConfig) -> str | None:
    required_tools = {tool for sample in workload.tasks for tool in sample.allowed_tools}
    if "ocr" in required_tools:
        executable = os.getenv("TESSERACT_EXECUTABLE", "tesseract")
        if shutil.which(executable) is None and not Path(executable).is_file():
            return f"OCR dependency unavailable: {executable}"
    if "pdf_render" in required_tools and shutil.which("pdftoppm") is None:
        return "PDF render dependency unavailable: pdftoppm"
    return None


def _resolve(parent: Path, configured: str) -> Path:
    path = Path(configured)
    return path if path.is_absolute() else parent / path


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
