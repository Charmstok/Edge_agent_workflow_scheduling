"""Verify real automatic Function Calling through an OpenAI-compatible LLM endpoint."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from edge_agent_workflow_scheduling.agents import AgentRunner
from edge_agent_workflow_scheduling.common import AgentRunStatus
from edge_agent_workflow_scheduling.config import load_llm_profile
from edge_agent_workflow_scheduling.executors import (
    ExecutorFactoryRegistry,
    ExecutorPool,
    LocalToolExecutor,
    create_openai_chat_executor,
    create_openai_responses_executor,
)
from edge_agent_workflow_scheduling.profiler import (
    TraceBundleStore,
    build_experiment_manifest,
    build_trace_bundle,
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

TOOL_NAMES = ("image_preprocess", "ocr", "pdf_parse", "pdf_render")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _tool_registry(input_dir: Path, output_dir: Path) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ImagePreprocessTool(
            ImagePreprocessConfig(
                local_root=input_dir,
                output_dir=output_dir / "image_preprocess",
            )
        )
    )
    registry.register(
        OCRTool(
            OCRConfig(
                local_root=input_dir,
                output_dir=output_dir / "ocr",
            )
        )
    )
    registry.register(
        PDFParseTool(
            DocumentToolConfig(
                local_root=input_dir,
                output_dir=output_dir / "pdf_parse",
            )
        )
    )
    registry.register(
        PDFRenderTool(
            PDFRenderConfig(
                local_root=input_dir,
                output_dir=output_dir / "pdf_render",
            )
        )
    )
    if tuple(registry.supported_tools()) != TOOL_NAMES:
        raise RuntimeError("verification registry did not expose all expected Tools")
    return registry


def _resources(profile, registry: ToolRegistry) -> ResourceRegistry:
    resources = ResourceRegistry()
    resources.register_llm(profile)
    for tool_name in registry.supported_tools():
        resources.register_tool_replica(
            ToolReplicaProfile(
                replica_id=f"local-{tool_name}",
                tool_name=tool_name,
                node_id="local-coordinator",
                platform="linux",
                implementation_version="verification-v1",
                executor_type="local",
                max_concurrency=1,
                metadata={"source_kind": "real_local_tool"},
            )
        )
    return resources


def _factories(registry: ToolRegistry) -> ExecutorFactoryRegistry:
    factories = ExecutorFactoryRegistry()
    factories.register_llm("openai_chat", create_openai_chat_executor)
    factories.register_llm("openai_responses", create_openai_responses_executor)
    factories.register_tool(
        "local",
        lambda profile: LocalToolExecutor(LocalWorker(profile=profile, tool_registry=registry)),
    )
    return factories


def _run_scenario(
    *,
    scenario: str,
    task: str,
    expected_tools: list[str],
    profile,
    registry: ToolRegistry,
    output_dir: Path,
) -> dict[str, object]:
    resources = _resources(profile, registry)
    runner = AgentRunner(
        agent_id="qwen38-function-calling-verifier",
        system_instruction=(
            "Decide whether the user request needs one of the provided tools. "
            "Choose tools from their descriptions and schemas. If a tool is needed, "
            "call it and use its returned result before answering. If no tool is needed, "
            "answer directly. Never invent a tool result."
        ),
        tool_registry=registry,
        resources=resources,
        scheduler=BaselineScheduler("least_queue"),
        executor_pool=ExecutorPool(_factories(registry)),
        max_rounds=3,
        max_tool_calls=2,
        timeout_sec=180.0,
        model_name=profile.model,
    )
    manifest = build_experiment_manifest(
        experiment_id=f"qwen38-function-calling-{scenario}",
        dataset_id="function-calling-verification-v1",
        sample_ids=[scenario],
        runner=runner,
        system_prompt_version="function-calling-verification-v1",
        user_template="{task}",
        user_template_version="function-calling-verification-v1",
        llm_profile_version="function-calling-verification-v1",
        tool_profile_version="function-calling-verification-v1",
        code_version="working-tree",
        mode="live",
        sampling_parameters=profile.deployment_config.get("model_parameters", {}),
    )
    execution = runner.run(
        task,
        task_id=scenario,
        run_id=f"{scenario}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}",
    )
    trace = build_trace_bundle(execution, manifest)
    scenario_dir = output_dir / scenario
    TraceBundleStore(scenario_dir / "trace.json").write(trace)
    selected_tools = [record.call.tool_name for record in execution.tool_records]
    result = {
        "scenario": scenario,
        "status": execution.agent_run.status.value,
        "expected_tools": expected_tools,
        "selected_tools": selected_tools,
        "llm_call_count": len(execution.llm_records),
        "tool_call_count": len(execution.tool_records),
        "final_output": execution.agent_run.final_output,
        "error_code": execution.agent_run.error_code,
        "error_message": execution.agent_run.error_message,
        "trace": str(scenario_dir / "trace.json"),
    }
    _write_json(scenario_dir / "summary.json", result)
    if execution.agent_run.status is not AgentRunStatus.COMPLETED:
        raise RuntimeError(
            f"{scenario} failed: {execution.agent_run.error_code}: "
            f"{execution.agent_run.error_message}"
        )
    if selected_tools != expected_tools:
        raise RuntimeError(f"{scenario} selected {selected_tools}, expected {expected_tools}")
    if expected_tools and len(execution.llm_records) < 2:
        raise RuntimeError("tool result was not followed by another LLM call")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profiles",
        type=Path,
        default=Path("configs/llm_profiles.toml"),
    )
    parser.add_argument("--llm-id", default="local-qwen38-27b")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/function_calling_verification"),
    )
    args = parser.parse_args()

    run_dir = args.output_dir / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    input_dir = (run_dir / "inputs").resolve()
    output_dir = (run_dir / "tool_outputs").resolve()
    input_dir.mkdir(parents=True, exist_ok=False)
    image_path = input_dir / "sample.png"
    Image.new("RGB", (256, 192), color=(45, 110, 180)).save(image_path)

    configured = load_llm_profile(args.profiles, args.llm_id)
    parameters = dict(configured.deployment_config.get("model_parameters", {}))
    parameters["tool_choice"] = "auto"
    profile = replace(
        configured,
        capabilities=sorted(set(configured.capabilities) | {"function_calling"}),
        deployment_config={
            **configured.deployment_config,
            "model_parameters": parameters,
        },
    )
    registry = _tool_registry(input_dir, output_dir)
    results = [
        _run_scenario(
            scenario="tool-needed",
            task=(
                "Convert this local image to grayscale and resize it, then report the "
                f"output URI: {image_path.as_uri()}"
            ),
            expected_tools=["image_preprocess"],
            profile=profile,
            registry=registry,
            output_dir=run_dir,
        ),
        _run_scenario(
            scenario="no-tool-needed",
            task="What is 17 plus 25? Answer with only the number.",
            expected_tools=[],
            profile=profile,
            registry=registry,
            output_dir=run_dir,
        ),
    ]
    summary = {
        "status": "passed",
        "llm_id": profile.llm_id,
        "model": profile.model,
        "endpoint": profile.base_url,
        "tool_choice": "auto",
        "offered_tools": list(TOOL_NAMES),
        "scenarios": results,
    }
    _write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
