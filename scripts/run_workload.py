"""Run the Milestone 4.8 workload in scripted, replay, or live mode.

The scripted path is an offline regression fixture.  It intentionally uses a
deterministic backend and profile executors, so its scores are not live LLM
quality measurements.  Replay schedules the saved call stream without asking
an LLM for new Tool decisions.  Live only reports validated runs when the
selected deployment advertises verified Function Calling and has credentials.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from edge_agent_workflow_scheduling.agents import (
    AgentRunner,
    ScriptedFunctionCall,
    ScriptedLLMBackend,
)
from edge_agent_workflow_scheduling.common import WorkloadConfig
from edge_agent_workflow_scheduling.config import load_llm_profile
from edge_agent_workflow_scheduling.executors import (
    BackendLLMExecutor,
    ExecutorFactoryRegistry,
    ExecutorPool,
    LocalToolExecutor,
    ProfileToolExecutor,
    create_openai_chat_executor,
)
from edge_agent_workflow_scheduling.profiler import (
    TraceBundleStore,
    build_experiment_manifest,
    build_trace_bundle,
    content_digest,
    evaluate_trace_bundle,
    load_trace_bundle,
    run_baseline_experiment,
    score_task_output,
    score_trace_bundle,
    write_evaluation_artifacts,
)
from edge_agent_workflow_scheduling.resources import (
    LLMInstanceProfile,
    LLMInstanceState,
    ResourceRegistry,
    ToolReplicaProfile,
    ToolReplicaState,
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
    ToolExecution,
    ToolRegistry,
)
from edge_agent_workflow_scheduling.workers import LocalWorker

DEFAULT_WORKLOAD = Path("configs/workload_milestone_4_1_v1.json")
DEFAULT_OUTPUT = Path("data/milestone_4_8")
DEFAULT_LLM_CONFIG = Path("configs/llm_profiles.toml")
DEFAULT_REPLAY_POLICIES = ("round_robin", "least_queue")


class _SchemaTool:
    """Register a workload schema without invoking a local Tool implementation."""

    def __init__(self, spec: dict[str, Any]) -> None:
        self.spec = spec
        self.tool_name = spec["name"]

    def execute(self, arguments: dict[str, Any], *, invocation_id: str) -> ToolExecution:
        del arguments, invocation_id
        return ToolExecution(
            success=False,
            error_code="scripted_registry_only",
            error_message="scripted mode executes Tools through a profile executor",
        )


@dataclass(frozen=True, slots=True)
class ScriptedRun:
    trace_path: Path
    task_id: str
    run_id: str
    score: float


def run_scripted(
    workload_path: Path,
    *,
    scenario: str,
    split: str,
    output_dir: Path,
    profile_seed: int = 0,
) -> dict[str, Any]:
    """Materialize one deterministic trace per generated workload request."""

    workload = WorkloadConfig.from_json(workload_path)
    plan = workload.generate(scenario, split=split, artifact_root=workload_path.parent)
    plan_path = output_dir / "workload_plans" / f"{scenario}-{split}.json"
    _write_json(plan_path, plan)
    sample_by_id = {sample.task_id: sample for sample in workload.tasks}
    scripted_dir = output_dir / "scripted"
    runs: list[ScriptedRun] = []
    for request in plan["requests"]:
        request = {**request, "scenario": scenario}
        sample = sample_by_id[request["task_id"]]
        trace, score = _run_scripted_request(
            workload,
            sample=sample,
            request=request,
            artifact_root=workload_path.parent,
            profile_seed=profile_seed,
        )
        run_dir = scripted_dir / request["run_id"]
        TraceBundleStore(run_dir / "trace.json").write(trace)
        _write_json(run_dir / "manifest.json", trace.manifest.to_dict())
        _write_json(run_dir / "score.json", score.to_dict())
        write_evaluation_artifacts(
            evaluate_trace_bundle(trace, task_score=score), run_dir / "evaluation"
        )
        _write_json(
            run_dir / "summary.json",
            {
                "mode": "scripted_offline",
                "run_id": trace.run.run_id,
                "task_id": trace.run.task_id,
                "status": trace.run.status,
                "final_task_score": score.normalized_score,
                "call_ids": [call.call_id for call in trace.calls],
                "call_input_records": [
                    {
                        "call_id": call.call_id,
                        "call_kind": call.call_kind,
                        "parameter_summary": call.parameter_summary,
                    }
                    for call in trace.calls
                ],
                "wall_clock_recorded": True,
                "wall_clock_semantics": "local scripted execution time; not a model benchmark",
            },
        )
        runs.append(
            ScriptedRun(
                trace_path=run_dir / "trace.json",
                task_id=sample.task_id,
                run_id=trace.run.run_id,
                score=score.normalized_score,
            )
        )
    summary = {
        "mode": "scripted_offline",
        "status": "completed",
        "scenario": scenario,
        "split": split,
        "workload_plan": str(plan_path),
        "input_fingerprint": content_digest(
            {"plan": plan, "calls": [_trace_call_fingerprint(run.trace_path) for run in runs]}
        ),
        "run_count": len(runs),
        "task_types": sorted({sample_by_id[run.task_id].task_type for run in runs}),
        "final_task_score": _distribution([run.score for run in runs]),
        "quality_interpretation": "scripted regression score; not live LLM quality",
        "traces": [str(run.trace_path) for run in runs],
    }
    _write_json(scripted_dir / "summary.json", summary)
    _write_json(output_dir / "summary.json", summary)
    return summary


def run_replay(
    trace_paths: Sequence[Path],
    *,
    workload_path: Path | None = None,
    output_dir: Path,
    policies: Sequence[str] = DEFAULT_REPLAY_POLICIES,
    seeds: Sequence[int] = (0,),
    profile_seed: int = 0,
    profile_jitter_ratio: float = 0.0,
    profile_failure_rate: float = 0.0,
) -> dict[str, Any]:
    """Compare baseline policies over immutable saved call streams."""

    if len(policies) < 2:
        raise ValueError("replay requires at least two policies")
    if not trace_paths:
        raise ValueError("replay requires at least one trace")
    replay_dir = output_dir / "replay"
    replay_workload = (
        WorkloadConfig.from_json(workload_path) if workload_path is not None else None
    )
    entries: list[dict[str, Any]] = []
    for trace_path in sorted(trace_paths):
        trace = load_trace_bundle(trace_path)
        result = run_baseline_experiment(
            trace,
            output_dir=replay_dir / trace.run.run_id,
            policies=policies,
            seeds=seeds,
            profile_seed=profile_seed,
            profile_jitter_ratio=profile_jitter_ratio,
            profile_failure_rate=profile_failure_rate,
            experiment_id=f"milestone-4-8-{trace.run.run_id}",
        )
        run_records = []
        for run in result.runs:
            run_dir = Path(result.output_dir) / f"{run.policy_name}-seed-{run.seed}"
            generated_trace_path = run_dir / "trace.json"
            run_record = run.to_dict()
            if replay_workload is not None and generated_trace_path.is_file():
                generated_trace = load_trace_bundle(generated_trace_path)
                task_score = score_trace_bundle(
                    generated_trace,
                    replay_workload,
                    trace_ref=str(generated_trace_path),
                )
                evaluation = evaluate_trace_bundle(generated_trace, task_score=task_score)
                run_summary_path = run_dir / "summary.json"
                run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
                write_evaluation_artifacts(evaluation, run_dir)
                run_summary["task_score"] = task_score.to_dict()
                run_summary["evaluation"] = evaluation.to_dict()
                _write_json(run_summary_path, run_summary)
                run_record["task_score"] = task_score.to_dict()
                run_record["evaluation"] = evaluation.to_dict()
            run_records.append(run_record)
        _write_json(
            Path(result.output_dir) / "summary.json",
            {**result.to_dict(), "runs": run_records},
        )
        entries.append(
            {
                "source_trace": str(trace_path),
                "source_run_id": trace.run.run_id,
                "source_input_fingerprint": content_digest(
                    [call.call_digest for call in trace.calls]
                ),
                "policies": list(result.policies),
                "seeds": list(result.seeds),
                "runs": run_records,
            }
        )
    summary = {
        "mode": "replay_profile",
        "status": "completed",
        "policy_count": len(policies),
        "trace_count": len(entries),
        "policies": list(policies),
        "entries": entries,
        "llm_interpretation": "replay reuses recorded call decisions and does not request an LLM",
    }
    _write_json(replay_dir / "summary.json", summary)
    _write_json(output_dir / "summary.json", summary)
    return summary


def run_live(
    workload_path: Path,
    *,
    config_path: Path,
    llm_id: str,
    repeats: int,
    output_dir: Path,
    scenario: str,
    split: str,
) -> dict[str, Any]:
    """Verify the provider with real Tool calls, then run repeated live workload tasks."""

    if repeats < 1:
        raise ValueError("repeats must be at least one")
    live_dir = output_dir / "live"
    try:
        profile = load_llm_profile(config_path, llm_id)
    except (KeyError, ValueError, OSError) as exc:
        summary = {
            "mode": "live",
            "status": "not_validated",
            "llm_id": llm_id,
            "repeats": repeats,
            "reason": f"cannot load LLM profile: {exc}",
        }
        _write_json(live_dir / "summary.json", summary)
        _write_json(output_dir / "summary.json", summary)
        return summary

    missing_keys = [name for name in profile.secret_env_vars if not os.getenv(name)]
    if missing_keys:
        return _live_not_validated(
            output_dir,
            {
                "llm_id": profile.llm_id,
                "model": profile.model,
                "repeat_count": repeats,
                "reason": f"missing credentials: {', '.join(missing_keys)}",
            },
        )

    workload = WorkloadConfig.from_json(workload_path)
    plan = workload.generate(scenario, split=split, artifact_root=workload_path.parent)
    verification = _verify_live_function_calling(
        profile,
        workload,
        artifact_root=workload_path.parent,
        output_dir=live_dir / "verification",
    )
    if verification["status"] != "passed":
        return _live_not_validated(
            output_dir,
            {
                "llm_id": profile.llm_id,
                "model": profile.model,
                "scenario": scenario,
                "split": split,
                "repeat_count": repeats,
                "sampling_parameters": profile.deployment_config.get("model_parameters", {}),
                "reason": verification["reason"],
                "verification": verification,
            },
        )

    live_profile = replace(
        profile,
        capabilities=sorted(set(profile.capabilities) | {"function_calling"}),
        deployment_config={
            **profile.deployment_config,
            "model_parameters": {
                **profile.deployment_config.get("model_parameters", {}),
                "tool_choice": "auto",
            },
        },
    )
    samples = {sample.task_id: sample for sample in workload.tasks}
    records: list[dict[str, Any]] = []
    for request in plan["requests"]:
        sample = samples[request["task_id"]]
        for repeat_index in range(1, repeats + 1):
            record = _run_live_request(
                workload,
                live_profile,
                sample=sample,
                request=request,
                repeat_index=repeat_index,
                artifact_root=workload_path.parent,
                output_dir=live_dir,
            )
            records.append(record)
    summary = _live_summary(
        profile=live_profile,
        scenario=scenario,
        split=split,
        repeats=repeats,
        verification=verification,
        records=records,
        plan=plan,
    )
    _write_json(live_dir / "summary.json", summary)
    _write_json(output_dir / "summary.json", summary)
    return summary


def _live_not_validated(output_dir: Path, fields: dict[str, Any]) -> dict[str, Any]:
    summary = {"mode": "live", "status": "not_validated", "metrics": None, **fields}
    _write_json(output_dir / "live" / "summary.json", summary)
    _write_json(output_dir / "summary.json", summary)
    return summary


def _verify_live_function_calling(
    profile: LLMInstanceProfile,
    workload: WorkloadConfig,
    *,
    artifact_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run a real tool-needed and no-tool-needed probe against the configured endpoint."""

    probe_dir = output_dir / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    input_dir = probe_dir / "inputs"
    tool_output_dir = probe_dir / "tool_outputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    source = next(
        (artifact_root / reference for reference in _image_artifacts(workload)), None
    )
    if source is None or not source.is_file():
        return {"status": "not_validated", "reason": "no image fixture available for live probe"}
    registry = _live_tool_registry(artifact_root, tool_output_dir)
    verified_profile = replace(
        profile,
        capabilities=sorted(set(profile.capabilities) | {"function_calling"}),
        deployment_config={
            **profile.deployment_config,
            "model_parameters": {
                **profile.deployment_config.get("model_parameters", {}),
                "tool_choice": "auto",
            },
        },
    )
    resources = _live_resources(verified_profile, registry)
    factories = _live_factories(registry)
    runner = AgentRunner(
        agent_id="milestone-4-8-live-verifier",
        system_instruction=(
            "Use image_preprocess when the user asks to transform an image. "
            "After the Tool result, answer briefly. For arithmetic questions, do not use a Tool."
        ),
        tool_registry=registry,
        resources=resources,
        scheduler=BaselineScheduler("least_queue"),
        executor_pool=ExecutorPool(factories),
        max_rounds=4,
        max_tool_calls=2,
        timeout_sec=180.0,
        model_name=verified_profile.model,
    )
    manifest = build_experiment_manifest(
        experiment_id="milestone-4-8-live-function-calling-verification",
        dataset_id="milestone-4-8-live-verification-v1",
        sample_ids=["tool-needed", "no-tool-needed"],
        runner=runner,
        system_prompt_version="milestone-4-8-live-verification-v1",
        user_template="{task}",
        user_template_version="milestone-4-8-live-verification-v1",
        llm_profile_version=verified_profile.metadata.get("profile_version", "deployment"),
        tool_profile_version="real-local-tools",
        code_version="working-tree",
        mode="live",
        sampling_parameters=verified_profile.deployment_config.get("model_parameters", {}),
    )
    scenarios = (
        (
            "tool-needed",
            (
                "Convert this local image to grayscale and resize it, then report the "
                f"output URI: {source.resolve().as_uri()}"
            ),
            ["image_preprocess"],
        ),
        ("no-tool-needed", "What is 17 plus 25? Answer with only the number.", []),
    )
    results: list[dict[str, Any]] = []
    for scenario, task, expected_tools in scenarios:
        try:
            execution = runner.run(
                task,
                task_id=scenario,
                run_id=f"live-verification-{scenario}",
            )
            trace = build_trace_bundle(execution, manifest)
            trace_path = probe_dir / scenario / "trace.json"
            TraceBundleStore(trace_path).write(trace)
            selected_tools = [record.call.tool_name for record in execution.tool_records]
            result = {
                "scenario": scenario,
                "status": execution.agent_run.status.value,
                "expected_tools": expected_tools,
                "selected_tools": selected_tools,
                "trace": str(trace_path),
                "error_code": execution.agent_run.error_code,
                "error_message": execution.agent_run.error_message,
            }
            results.append(result)
            if result["status"] != "completed" or selected_tools != expected_tools:
                return {
                    "status": "not_validated",
                    "reason": f"Function Calling probe failed for {scenario}",
                    "scenarios": results,
                }
        except Exception as exc:
            return {
                "status": "not_validated",
                "reason": f"Function Calling probe error: {type(exc).__name__}",
                "scenarios": results,
            }
    summary = {"status": "passed", "scenarios": results, "endpoint": profile.base_url}
    _write_json(probe_dir / "summary.json", summary)
    return summary


def _live_tool_registry(input_dir: Path, output_dir: Path) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ImagePreprocessTool(
            ImagePreprocessConfig(
                local_root=input_dir, output_dir=output_dir / "image_preprocess"
            )
        )
    )
    registry.register(OCRTool(OCRConfig(local_root=input_dir, output_dir=output_dir / "ocr")))
    registry.register(
        PDFParseTool(
            DocumentToolConfig(local_root=input_dir, output_dir=output_dir / "pdf_parse")
        )
    )
    registry.register(
        PDFRenderTool(
            PDFRenderConfig(local_root=input_dir, output_dir=output_dir / "pdf_render")
        )
    )
    return registry


def _live_resources(profile: LLMInstanceProfile, registry: ToolRegistry) -> ResourceRegistry:
    resources = ResourceRegistry()
    resources.register_llm(profile)
    for tool_name in registry.supported_tools():
        resources.register_tool_replica(
            ToolReplicaProfile(
                replica_id=f"live-{tool_name}",
                tool_name=tool_name,
                node_id="local-coordinator",
                platform="linux",
                implementation_version="live-local-v1",
                executor_type="local",
                max_concurrency=1,
            ),
            ToolReplicaState(replica_id=f"live-{tool_name}"),
        )
    return resources


def _live_factories(registry: ToolRegistry) -> ExecutorFactoryRegistry:
    factories = ExecutorFactoryRegistry()
    factories.register_llm("openai_chat", create_openai_chat_executor)
    factories.register_tool(
        "local",
        lambda profile: LocalToolExecutor(LocalWorker(profile=profile, tool_registry=registry)),
    )
    return factories


def _image_artifacts(workload: WorkloadConfig) -> list[str]:
    return [
        reference
        for task in workload.tasks
        for reference in task.artifact_refs
        if reference.lower().endswith((".png", ".jpg", ".jpeg"))
    ]


def _run_live_request(
    workload: WorkloadConfig,
    profile: LLMInstanceProfile,
    *,
    sample: Any,
    request: dict[str, Any],
    repeat_index: int,
    artifact_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    run_id = f"{request['run_id']}-repeat-{repeat_index:04d}"
    run_dir = output_dir / "runs" / run_id
    registry = _live_tool_registry(artifact_root, run_dir / "tool_outputs")
    resources = _live_resources(profile, registry)
    runner = AgentRunner(
        agent_id=request["agent_id"],
        system_instruction=workload.agent.system_prompt,
        tool_registry=registry,
        resources=resources,
        scheduler=BaselineScheduler("least_queue"),
        executor_pool=ExecutorPool(_live_factories(registry)),
        max_rounds=workload.agent.max_rounds,
        max_tool_calls=workload.agent.max_tool_calls,
        timeout_sec=workload.agent.timeout_sec,
        model_name=profile.model,
    )
    manifest = build_experiment_manifest(
        experiment_id=f"milestone-4-8-live-{run_id}",
        dataset_id=workload.dataset_id,
        sample_ids=[sample.task_id],
        runner=runner,
        system_prompt_version=workload.agent.system_prompt_version,
        user_template=workload.agent.user_template,
        user_template_version=workload.agent.user_template_version,
        llm_profile_version=profile.metadata.get("profile_version", "deployment"),
        tool_profile_version="real-local-tools",
        code_version="working-tree",
        mode="live",
        sampling_parameters=profile.deployment_config.get("model_parameters", {}),
        workload_parameters={
            "scenario": request.get("scenario"),
            "split": sample.split,
            "repeat_index": repeat_index,
            "arrival_offset_sec": request["arrival_offset_sec"],
            "input_digest": request["input_digest"],
        },
    )
    execution = runner.run(
        request["user_task"],
        task_id=sample.task_id,
        run_id=run_id,
        call_metadata={"task_type": sample.task_type, "input_size": sample.input_size},
    )
    trace = build_trace_bundle(execution, manifest)
    score = score_task_output(
        sample,
        trace.run.final_output,
        run_id=trace.run.run_id,
        status=trace.run.status,
        model_ids=(profile.llm_id,),
        trace_digest=content_digest(trace.to_dict()),
    )
    TraceBundleStore(run_dir / "trace.json").write(trace)
    _write_json(run_dir / "score.json", score.to_dict())
    write_evaluation_artifacts(
        evaluate_trace_bundle(trace, task_score=score), run_dir / "evaluation"
    )
    return {
        "run_id": run_id,
        "task_id": sample.task_id,
        "repeat_index": repeat_index,
        "status": trace.run.status,
        "success": trace.run.status == "completed",
        "end_to_end_latency_sec": trace.run.end_to_end_latency_sec,
        "tool_call_count": trace.run.tool_call_count,
        "final_task_score": score.normalized_score,
        "trace": str(run_dir / "trace.json"),
        "sampling_parameters": profile.deployment_config.get("model_parameters", {}),
        "error_code": trace.run.error_code,
    }


def _live_summary(
    *,
    profile: LLMInstanceProfile,
    scenario: str,
    split: str,
    repeats: int,
    verification: dict[str, Any],
    records: list[dict[str, Any]],
    plan: dict[str, Any],
) -> dict[str, Any]:
    latencies = [record["end_to_end_latency_sec"] for record in records]
    tool_counts = [record["tool_call_count"] for record in records]
    scores = [record["final_task_score"] for record in records]
    return {
        "mode": "live",
        "status": (
            "validated"
            if records and all(record["success"] for record in records)
            else "completed_with_failures"
        ),
        "llm_id": profile.llm_id,
        "model": profile.model,
        "scenario": scenario,
        "split": split,
        "repeat_count": repeats,
        "request_count": len(plan["requests"]),
        "sampling_parameters": profile.deployment_config.get("model_parameters", {}),
        "verification": verification,
        "metrics": {
            "success_rate": (
                sum(record["success"] for record in records) / len(records)
                if records
                else 0.0
            ),
            "end_to_end_latency_sec": _distribution(latencies),
            "tool_call_count": _distribution(tool_counts),
            "final_task_score": _distribution(scores),
        },
        "runs": records,
    }


def _run_scripted_request(
    workload: WorkloadConfig,
    *,
    sample: Any,
    request: dict[str, Any],
    artifact_root: Path,
    profile_seed: int,
) -> tuple[Any, Any]:
    tool_names = _tool_names_for_sample(sample)
    function_calls = [
        ScriptedFunctionCall(
            call_id=f"{request['run_id']}-call-{index:02d}",
            name=tool_name,
            arguments=_arguments_for_tool(tool_name, sample, artifact_root),
        )
        for index, tool_name in enumerate(tool_names)
    ]
    backend = ScriptedLLMBackend.multiple_tools(
        function_calls,
        final_text=json.dumps(sample.reference_answer, ensure_ascii=False, sort_keys=True),
    )
    resources = _scripted_resources(tool_names)
    registry = ToolRegistry()
    for schema in workload.agent.tool_schemas:
        if schema["name"] in tool_names:
            registry.register(_SchemaTool(schema))
    factories = ExecutorFactoryRegistry()
    factories.register_llm(
        "scripted_backend",
        lambda profile: BackendLLMExecutor(profile=profile, backend=backend),
    )
    factories.register_tool(
        "profile",
        lambda profile: ProfileToolExecutor(
            profile=profile,
            output={"mode": "scripted_offline", "tool_name": profile.tool_name},
            seed=profile_seed,
        ),
    )
    runner = AgentRunner(
        agent_id=request["agent_id"],
        system_instruction=workload.agent.system_prompt,
        tool_registry=registry,
        resources=resources,
        scheduler=BaselineScheduler("round_robin"),
        executor_pool=ExecutorPool(factories),
        max_rounds=workload.agent.max_rounds,
        max_tool_calls=workload.agent.max_tool_calls,
        timeout_sec=workload.agent.timeout_sec,
        model_name="scripted-model",
    )
    manifest = build_experiment_manifest(
        experiment_id=f"milestone-4-8-scripted-{request['run_id']}",
        dataset_id=workload.dataset_id,
        sample_ids=[sample.task_id],
        runner=runner,
        system_prompt_version=workload.agent.system_prompt_version,
        user_template=workload.agent.user_template,
        user_template_version=workload.agent.user_template_version,
        llm_profile_version="scripted-offline-v1",
        tool_profile_version="scripted-offline-v1",
        code_version="milestone-4-8",
        mode="live",
        sampling_parameters=workload.agent.sampling_parameters,
        profile_seed=profile_seed,
        workload_parameters={
            "execution_mode": "scripted_offline",
            "scenario": request.get("scenario", "unknown"),
            "arrival_offset_sec": request["arrival_offset_sec"],
            "input_digest": request["input_digest"],
            "seed_semantics": "input/profile determinism only; wall clock is recorded separately",
        },
    )
    execution = runner.run(
        request["user_task"],
        task_id=sample.task_id,
        run_id=request["run_id"],
        call_metadata={
            "task_type": sample.task_type,
            "input_size": sample.input_size,
            "input_digest": request["input_digest"],
            "execution_mode": "scripted_offline",
        },
    )
    trace = build_trace_bundle(execution, manifest)
    score = score_task_output(
        sample,
        trace.run.final_output,
        run_id=trace.run.run_id,
        status=trace.run.status,
        model_ids=("scripted-model",),
        trace_digest=content_digest(trace.to_dict()),
    )
    return trace, score


def _scripted_resources(tool_names: Sequence[str]) -> ResourceRegistry:
    resources = ResourceRegistry()
    resources.register_llm(
        LLMInstanceProfile(
            llm_id="scripted-llm",
            provider="scripted",
            model="scripted-model",
            node_id="offline-scripted",
            platform="linux",
            executor_type="scripted_backend",
            capabilities=["function_calling"],
            context_window_tokens=32768,
            token_profile={"tokens_per_sec": 100.0},
            quality_profile={"default": 1.0},
            max_concurrency=1,
            metadata={"profile_version": "scripted-offline-v1", "source_kind": "scripted"},
        ),
        LLMInstanceState(llm_id="scripted-llm"),
    )
    for tool_name in sorted(set(tool_names)):
        for suffix, latency in (("fast", 0.02), ("slow", 0.08)):
            replica_id = f"{tool_name}-{suffix}"
            resources.register_tool_replica(
                ToolReplicaProfile(
                    replica_id=replica_id,
                    tool_name=tool_name,
                    node_id=f"offline-{suffix}",
                    platform="linux",
                    implementation_version="scripted-offline-v1",
                    executor_type="profile",
                    latency_profile={"execution_time_sec": latency},
                    quality_profile={"default": 1.0},
                    max_concurrency=1,
                    metadata={"profile_version": "scripted-offline-v1", "source_kind": "scripted"},
                ),
                ToolReplicaState(replica_id=replica_id),
            )
    return resources


def _tool_names_for_sample(sample: Any) -> list[str]:
    if sample.task_type == "image_ocr":
        return ["image_preprocess"] if "image_preprocess" in sample.allowed_tools else ["ocr"]
    if sample.task_type == "pdf_extract":
        return ["pdf_parse"]
    if sample.task_type == "document_reconcile":
        names = [name for name in ("image_preprocess", "pdf_parse") if name in sample.allowed_tools]
        return names or [sample.allowed_tools[0]]
    return [sample.allowed_tools[0]]


def _arguments_for_tool(tool_name: str, sample: Any, artifact_root: Path) -> dict[str, Any]:
    suffix = ".png" if tool_name in {"image_preprocess", "ocr"} else ".pdf"
    artifact = next(
        (reference for reference in sample.artifact_refs if reference.lower().endswith(suffix)),
        sample.artifact_refs[0],
    )
    input_uri = (artifact_root / artifact).resolve().as_uri()
    if tool_name == "image_preprocess":
        return {
            "input_uri": input_uri,
            "operations": ["grayscale", "resize"],
            "operation_repeat": 1,
        }
    return {"input_uri": input_uri}


def _trace_call_fingerprint(path: Path) -> list[dict[str, str]]:
    trace = load_trace_bundle(path)
    return [
        {
            "call_id": call.call_id,
            "call_kind": call.call_kind,
            "tool_name": call.tool_name or "",
            "parameter_summary": json.dumps(
                call.parameter_summary, ensure_ascii=False, sort_keys=True
            ),
        }
        for call in trace.calls
    ]


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "p95": 0.0, "stddev": 0.0}
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, max(0, int((len(ordered) - 1) * 0.95)))]
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p95": p95,
        "stddev": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_trace_paths(args: argparse.Namespace, output_dir: Path) -> list[Path]:
    paths = {path.resolve() for path in args.trace}
    for root in args.trace_root:
        paths.update(path.resolve() for path in root.rglob("trace.json"))
    if not paths:
        paths.update(path.resolve() for path in (output_dir / "scripted").glob("*/trace.json"))
    return sorted(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("scripted", "replay", "live"), required=True)
    parser.add_argument("--workload", type=Path, default=DEFAULT_WORKLOAD)
    parser.add_argument("--scenario", default="low_load")
    parser.add_argument("--split", choices=("calibration", "validation"), default="validation")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--profile-seed", type=int, default=0)
    parser.add_argument("--profile-jitter-ratio", type=float, default=0.0)
    parser.add_argument("--profile-failure-rate", type=float, default=0.0)
    parser.add_argument("--policies", nargs="+", default=list(DEFAULT_REPLAY_POLICIES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--trace", action="append", type=Path, default=[])
    parser.add_argument("--trace-root", action="append", type=Path, default=[])
    parser.add_argument("--llm-config", type=Path, default=DEFAULT_LLM_CONFIG)
    parser.add_argument("--llm-id", default="local-qwen35-9b")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.mode == "scripted":
        result = run_scripted(
            args.workload.resolve(),
            scenario=args.scenario,
            split=args.split,
            output_dir=args.output_dir,
            profile_seed=args.profile_seed,
        )
    elif args.mode == "replay":
        result = run_replay(
            _read_trace_paths(args, args.output_dir),
            output_dir=args.output_dir,
            workload_path=args.workload.resolve(),
            policies=args.policies,
            seeds=args.seeds,
            profile_seed=args.profile_seed,
            profile_jitter_ratio=args.profile_jitter_ratio,
            profile_failure_rate=args.profile_failure_rate,
        )
    else:
        result = run_live(
            args.workload.resolve(),
            config_path=args.llm_config.resolve(),
            llm_id=args.llm_id,
            repeats=args.repeats,
            output_dir=args.output_dir,
            scenario=args.scenario,
            split=args.split,
        )
    print(json.dumps({"mode": args.mode, "status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
