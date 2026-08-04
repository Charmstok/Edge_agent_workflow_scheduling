"""Run Milestone 2.8 live, multi-Tool, online, and replay demos."""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image

from edge_agent_workflow_scheduling.agents import (
    AgentRunner,
    ScriptedFunctionCall,
    ScriptedLLMBackend,
)
from edge_agent_workflow_scheduling.config import load_llm_profile
from edge_agent_workflow_scheduling.executors import (
    BackendLLMExecutor,
    ExecutorFactoryRegistry,
    ExecutorPool,
    LocalToolExecutor,
    ProfileLLMExecutor,
    ProfileToolExecutor,
    create_openai_responses_executor,
)
from edge_agent_workflow_scheduling.profiler import (
    ExperimentManifest,
    TraceBundle,
    TraceBundleStore,
    build_experiment_manifest,
    build_trace_bundle,
    load_trace_bundle,
    replay_calls,
    resources_from_manifest,
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
    ImagePreprocessConfig,
    ImagePreprocessTool,
    ToolRegistry,
)
from edge_agent_workflow_scheduling.workers import LocalWorker

DEFAULT_OUTPUT_DIR = Path("data/milestone_2_8")
DEFAULT_LLM_CONFIG = Path("configs/llm_profiles.toml")
DEFAULT_REPLAY_POLICIES = ("round_robin", "least_queue")
PROFILE_SEED = 28


def run_offline_demo(output_dir: Path, *, multi_tool: bool = False) -> Path:
    """Run a deterministic Agent over real and profiled Tool replicas."""

    demo_name = "multi_tool" if multi_tool else "offline"
    demo_dir = output_dir / demo_name
    input_dir = (demo_dir / "inputs").resolve()
    tool_output_dir = (demo_dir / "tool_outputs").resolve()
    image_paths = _create_demo_images(input_dir, count=2 if multi_tool else 1)
    tool_registry = _create_tool_registry(input_dir, tool_output_dir)
    function_calls = [
        ScriptedFunctionCall(
            call_id=f"{demo_name}-function-{index + 1}",
            name="image_preprocess",
            arguments={
                "input_uri": image_path.as_uri(),
                "operations": ["grayscale", "resize"],
                "operation_repeat": 1,
            },
        )
        for index, image_path in enumerate(image_paths)
    ]
    backend = ScriptedLLMBackend.from_function_calls(
        function_calls,
        final_text=f"{len(function_calls)} image preprocessing call(s) completed.",
    )
    resources = _create_offline_resources(multi_tool=multi_tool)
    factories = _create_offline_factories(
        backend=backend,
        tool_registry=tool_registry,
        multi_tool=multi_tool,
    )
    runner = AgentRunner(
        agent_id=f"{demo_name}-agent",
        system_instruction=(
            "Use image_preprocess for every requested image, then summarize the completed calls."
        ),
        tool_registry=tool_registry,
        resources=resources,
        scheduler=BaselineScheduler("least_queue"),
        executor_pool=ExecutorPool(factories),
        max_rounds=4,
        max_tool_calls=4,
        timeout_sec=30.0,
    )
    manifest = _build_manifest(
        experiment_id=f"milestone-2-8-{demo_name}",
        sample_ids=[path.name for path in image_paths],
        runner=runner,
        sampling_parameters={},
    )
    execution = runner.run(
        f"Preprocess {len(image_paths)} image(s).",
        task_id=f"{demo_name}-sample",
        run_id=f"{demo_name}-run",
    )
    trace = build_trace_bundle(execution, manifest)
    _validate_live_trace(trace, expected_tool_calls=len(function_calls))
    if multi_tool:
        _validate_multi_tool_trace(trace, function_calls)
    _write_live_artifacts(demo_dir, trace)
    print(f"{demo_name}: {trace.run.status} -> {demo_dir}")
    return demo_dir / "trace.json"


def run_online_demo(
    output_dir: Path,
    *,
    config_path: Path,
    llm_id: str,
    runs: int = 1,
) -> Path | None:
    """Run one real Responses API Function Calling loop when credentials exist."""

    if runs < 1:
        raise ValueError("online runs must be at least 1")
    demo_dir = output_dir / "online"
    input_dir = (demo_dir / "inputs").resolve()
    tool_output_dir = (demo_dir / "tool_outputs").resolve()
    image_path = _create_demo_images(input_dir, count=1)[0]
    tool_registry = _create_tool_registry(input_dir, tool_output_dir)
    llm_profile = load_llm_profile(config_path, llm_id)
    resources = ResourceRegistry()
    resources.register_llm(llm_profile)
    tool_profile = _real_tool_profile("online-image-real", node_id="coordinator")
    resources.register_tool_replica(tool_profile)

    factories = ExecutorFactoryRegistry()
    factories.register_llm("openai_responses", create_openai_responses_executor)
    factories.register_tool(
        "local",
        lambda profile: LocalToolExecutor(
            LocalWorker(profile=profile, tool_registry=tool_registry)
        ),
    )
    runner = AgentRunner(
        agent_id="online-agent",
        system_instruction=(
            "You are a Function Calling demo. You must call image_preprocess exactly once "
            "with the input URI and operation values supplied by the user. After the Tool "
            "returns, provide a short final answer."
        ),
        tool_registry=tool_registry,
        resources=resources,
        scheduler=BaselineScheduler("least_queue"),
        executor_pool=ExecutorPool(factories),
        max_rounds=4,
        max_tool_calls=2,
        timeout_sec=120.0,
        model_name=llm_profile.model,
    )
    model_parameters = llm_profile.deployment_config.get("model_parameters", {})
    manifest = _build_manifest(
        experiment_id="milestone-2-8-online",
        sample_ids=[image_path.name],
        runner=runner,
        sampling_parameters=(model_parameters if isinstance(model_parameters, dict) else {}),
    )
    _write_json(demo_dir / "manifest.json", manifest.to_dict())

    missing_env = _missing_api_key_environment(llm_profile)
    if missing_env is not None:
        summary = {
            "mode": "online",
            "status": "skipped",
            "reason": f"environment variable {missing_env!r} is not set",
            "llm_id": llm_profile.llm_id,
            "model": llm_profile.model,
        }
        _write_json(demo_dir / "summary.json", summary)
        print(f"online: skipped ({missing_env} is not set) -> {demo_dir}")
        return None

    task = (
        "Call image_preprocess once with this exact input: "
        f'{{"input_uri": "{image_path.as_uri()}", '
        '"operations": ["grayscale", "resize"], "operation_repeat": 1}}'
    )
    traces: list[TraceBundle] = []
    for run_index in range(runs):
        execution = runner.run(
            task,
            task_id="online-sample",
            run_id=f"online-run-{run_index + 1:04d}",
        )
        trace = build_trace_bundle(execution, manifest)
        _validate_live_trace(trace, expected_tool_calls=1)
        traces.append(trace)
        run_dir = demo_dir / "runs" / f"run-{run_index + 1:04d}"
        _write_live_artifacts(run_dir, trace)

    TraceBundleStore(demo_dir / "trace.json").write(traces[0])
    _write_json(demo_dir / "summary.json", _online_run_summary(traces))
    print(f"online: {len(traces)} completed run(s) -> {demo_dir}")
    return demo_dir / "trace.json"


def run_replay_demo(
    output_dir: Path,
    *,
    source_trace_path: Path,
    policies: tuple[str, ...] = DEFAULT_REPLAY_POLICIES,
) -> None:
    """Replay one immutable call stream under multiple baseline policies."""

    if len(policies) < 2:
        raise ValueError("replay demo requires at least two policies")
    source_trace = load_trace_bundle(source_trace_path)
    replay_dir = output_dir / "replay"
    fingerprints: dict[str, str] = {}
    policy_summaries: dict[str, dict[str, Any]] = {}

    for policy_name in policies:
        replay_result = replay_calls(
            source_trace,
            scheduler=BaselineScheduler(policy_name),
            resources=resources_from_manifest(source_trace),
        )
        replay_manifest = replace(
            source_trace.manifest,
            experiment_id=f"{source_trace.manifest.experiment_id}-replay-{policy_name}",
            scheduler_name=policy_name,
            mode="replay",
            run_started_at=datetime.now(UTC).isoformat(),
        )
        policy_dir = replay_dir / policy_name
        _write_json(policy_dir / "manifest.json", replay_manifest.to_dict())
        TraceBundleStore(policy_dir / "trace.json").write(source_trace)
        summary = {
            "mode": "replay",
            "source_trace": str(source_trace_path),
            "source_agent_run": _agent_run_summary(source_trace),
            "replay": replay_result.to_dict(),
        }
        _write_json(policy_dir / "summary.json", summary)
        fingerprints[policy_name] = replay_result.input_fingerprint
        policy_summaries[policy_name] = summary

    identical_inputs = len(set(fingerprints.values())) == 1
    if not identical_inputs:
        raise RuntimeError("replay policies received different input fingerprints")
    _write_json(
        replay_dir / "summary.json",
        {
            "mode": "replay-comparison",
            "source_trace": str(source_trace_path),
            "policies": list(policies),
            "input_fingerprints": fingerprints,
            "identical_inputs": identical_inputs,
            "selected_targets": {
                policy: [decision["selected_target"] for decision in summary["replay"]["decisions"]]
                for policy, summary in policy_summaries.items()
            },
        },
    )
    print(f"replay: {', '.join(policies)} -> {replay_dir}")


def _create_offline_resources(*, multi_tool: bool) -> ResourceRegistry:
    resources = ResourceRegistry()
    resources.register_llm(
        _llm_profile("offline-llm-scripted", "scripted-function-calling", "scripted"),
        LLMInstanceState(llm_id="offline-llm-scripted", queue_len=0),
    )
    resources.register_llm(
        _llm_profile("offline-llm-profile", "profiled-model", "profile"),
        LLMInstanceState(llm_id="offline-llm-profile", queue_len=5),
    )
    real_replica_ids = ["image-real-a", "image-real-b"] if multi_tool else ["image-real"]
    for replica_id in real_replica_ids:
        resources.register_tool_replica(
            _real_tool_profile(replica_id, node_id="macbook"),
            ToolReplicaState(replica_id=replica_id, queue_len=0),
        )
    profile = ToolReplicaProfile(
        replica_id="image-profile",
        tool_name="image_preprocess",
        node_id="simulated-edge",
        platform="profile",
        implementation_version="profile-v1",
        executor_type="profile",
        latency_profile={"execution_time_sec": 0.05},
        energy_profile={"joules_per_call": 0.2},
        max_concurrency=2,
        metadata={"profile_version": "demo-v1"},
    )
    resources.register_tool_replica(
        profile,
        ToolReplicaState(replica_id=profile.replica_id, queue_len=10),
    )
    return resources


def _create_offline_factories(
    *,
    backend: ScriptedLLMBackend,
    tool_registry: ToolRegistry,
    multi_tool: bool,
) -> ExecutorFactoryRegistry:
    factories = ExecutorFactoryRegistry()
    factories.register_llm(
        "scripted",
        lambda profile: BackendLLMExecutor(profile=profile, backend=backend),
    )
    factories.register_llm(
        "profile",
        lambda profile: ProfileLLMExecutor(profile=profile, seed=PROFILE_SEED),
    )
    factories.register_tool(
        "local",
        lambda profile: LocalToolExecutor(
            LocalWorker(
                profile=profile,
                tool_registry=tool_registry,
                artificial_delay_sec=0.05 if multi_tool else 0.0,
            )
        ),
    )
    factories.register_tool(
        "profile",
        lambda profile: ProfileToolExecutor(
            profile=profile,
            output={"output_uri": "profile://image-output"},
            seed=PROFILE_SEED,
        ),
    )
    return factories


def _create_tool_registry(input_dir: Path, output_dir: Path) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ImagePreprocessTool(
            ImagePreprocessConfig(
                output_dir=output_dir,
                local_root=input_dir,
                operations=("grayscale", "resize"),
            )
        )
    )
    return registry


def _llm_profile(llm_id: str, model: str, executor_type: str) -> LLMInstanceProfile:
    return LLMInstanceProfile(
        llm_id=llm_id,
        provider="offline",
        model=model,
        node_id="coordinator",
        platform="macos",
        executor_type=executor_type,
        capabilities=["function_calling"],
        context_window_tokens=4096,
        token_profile={"tokens_per_sec": 100.0},
        energy_profile={"joules_per_token": 0.001},
        max_concurrency=2,
        metadata={"profile_version": "demo-v1"},
    )


def _real_tool_profile(replica_id: str, *, node_id: str) -> ToolReplicaProfile:
    return ToolReplicaProfile(
        replica_id=replica_id,
        tool_name="image_preprocess",
        node_id=node_id,
        platform="macos",
        implementation_version="pillow-v1",
        executor_type="local",
        latency_profile={"execution_time_sec": 0.05},
        max_concurrency=1,
        metadata={"profile_version": "demo-v1"},
    )


def _build_manifest(
    *,
    experiment_id: str,
    sample_ids: list[str],
    runner: AgentRunner,
    sampling_parameters: dict[str, Any],
) -> ExperimentManifest:
    return build_experiment_manifest(
        experiment_id=experiment_id,
        dataset_id="milestone-2-8-demo",
        sample_ids=sample_ids,
        runner=runner,
        system_prompt_version="demo-v1",
        user_template="{task}",
        user_template_version="demo-v1",
        llm_profile_version="demo-v1",
        tool_profile_version="demo-v1",
        code_version=os.getenv("GIT_COMMIT", "working-tree"),
        mode="live",
        sampling_parameters=deepcopy(sampling_parameters),
        scheduler_parameters={},
        profile_seed=PROFILE_SEED,
    )


def _create_demo_images(input_dir: Path, *, count: int) -> list[Path]:
    input_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index in range(count):
        path = input_dir / f"sample-{index + 1}.png"
        image = Image.new(
            "RGB",
            (256, 192),
            color=(40 + index * 50, 110, 180 - index * 40),
        )
        image.save(path)
        paths.append(path.resolve())
    return paths


def _validate_live_trace(trace: TraceBundle, *, expected_tool_calls: int) -> None:
    if trace.run.status != "completed":
        raise RuntimeError(f"AgentRun failed: {trace.run.error_code}: {trace.run.error_message}")
    if trace.run.tool_call_count != expected_tool_calls:
        raise RuntimeError(
            f"expected {expected_tool_calls} ToolCall(s), got {trace.run.tool_call_count}"
        )
    if trace.run.llm_call_count < 2:
        raise RuntimeError("demo did not complete LLM -> Tool -> LLM")


def _validate_multi_tool_trace(
    trace: TraceBundle,
    function_calls: list[ScriptedFunctionCall],
) -> None:
    tool_calls = [call for call in trace.calls if call.call_kind == "tool"]
    expected_ids = [function_call.call_id for function_call in function_calls]
    actual_ids = [call.function_call_id for call in tool_calls]
    output_ids = [item["call_id"] for item in trace.run.function_call_outputs]
    if actual_ids != expected_ids or output_ids != expected_ids:
        raise RuntimeError("multi-Tool results did not preserve function call order")
    if len({call.selected_target for call in tool_calls}) != len(tool_calls):
        raise RuntimeError("multi-Tool demo did not use independent replicas")


def _write_live_artifacts(output_dir: Path, trace: TraceBundle) -> None:
    _write_json(output_dir / "manifest.json", trace.manifest.to_dict())
    TraceBundleStore(output_dir / "trace.json").write(trace)
    _write_json(output_dir / "summary.json", _agent_run_summary(trace))


def _agent_run_summary(trace: TraceBundle) -> dict[str, Any]:
    return {
        "mode": trace.manifest.mode,
        "run_id": trace.run.run_id,
        "status": trace.run.status,
        "final_output": trace.run.final_output,
        "total_rounds": trace.run.total_rounds,
        "llm_call_count": trace.run.llm_call_count,
        "tool_call_count": trace.run.tool_call_count,
        "end_to_end_latency_sec": trace.run.end_to_end_latency_sec,
        "calls": [
            {
                "sequence_id": call.sequence_id,
                "call_id": call.call_id,
                "call_kind": call.call_kind,
                "selected_target": call.selected_target,
                "total_latency_sec": call.total_latency_sec,
                "success": call.success,
            }
            for call in trace.calls
        ],
    }


def _online_run_summary(traces: list[TraceBundle]) -> dict[str, Any]:
    latencies = [trace.run.end_to_end_latency_sec for trace in traces]
    tool_call_counts = [trace.run.tool_call_count for trace in traces]
    round_counts = [trace.run.total_rounds for trace in traces]
    return {
        "mode": "online-live",
        "run_count": len(traces),
        "all_completed": all(trace.run.status == "completed" for trace in traces),
        "runs": [_agent_run_summary(trace) for trace in traces],
        "distribution": {
            "end_to_end_latency_sec": {
                "values": latencies,
                "mean": sum(latencies) / len(latencies),
                "min": min(latencies),
                "max": max(latencies),
            },
            "tool_call_count": tool_call_counts,
            "round_count": round_counts,
        },
    }


def _missing_api_key_environment(profile: LLMInstanceProfile) -> str | None:
    requires_api_key = profile.deployment_config.get("requires_api_key", True)
    if not isinstance(requires_api_key, bool):
        raise ValueError("deployment_config.requires_api_key must be a boolean")
    if not requires_api_key:
        return None
    if len(profile.secret_env_vars) != 1:
        raise ValueError("online profile must declare exactly one API key environment variable")
    env_name = profile.secret_env_vars[0]
    return None if os.getenv(env_name) else env_name


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("all", "offline", "multi-tool", "online", "replay"),
        default="all",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--llm-config", type=Path, default=DEFAULT_LLM_CONFIG)
    parser.add_argument("--online-llm-id", default="online-doubao")
    parser.add_argument("--online-runs", type=int, default=1)
    parser.add_argument("--replay-trace", type=Path)
    parser.add_argument(
        "--replay-policies",
        nargs="+",
        default=list(DEFAULT_REPLAY_POLICIES),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    offline_trace: Path | None = None
    if args.mode in {"all", "offline"}:
        offline_trace = run_offline_demo(args.output_dir)
    if args.mode in {"all", "multi-tool"}:
        run_offline_demo(args.output_dir, multi_tool=True)
    if args.mode in {"all", "online"}:
        run_online_demo(
            args.output_dir,
            config_path=args.llm_config,
            llm_id=args.online_llm_id,
            runs=args.online_runs,
        )
    if args.mode in {"all", "replay"}:
        source_trace = (
            args.replay_trace or offline_trace or (args.output_dir / "offline" / "trace.json")
        )
        if not source_trace.exists():
            raise FileNotFoundError(
                f"replay trace {source_trace} does not exist; run the offline demo first"
            )
        run_replay_demo(
            args.output_dir,
            source_trace_path=source_trace,
            policies=tuple(args.replay_policies),
        )


if __name__ == "__main__":
    main()
