"""Credential-free LLM catalog acceptance and optional real Function Calling."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from edge_agent_workflow_scheduling.agents import AgentRunner
from edge_agent_workflow_scheduling.common import LLMCall, ScheduleDecision
from edge_agent_workflow_scheduling.executors import ExecutorFactoryRegistry, ExecutorPool
from edge_agent_workflow_scheduling.executors.llm_deployment import LLMDeployment
from edge_agent_workflow_scheduling.executors.local_deployment import LocalToolDeployment
from edge_agent_workflow_scheduling.profiler.local_deployment import (
    load_deployment_samples,
    self_check_deployment,
)
from edge_agent_workflow_scheduling.profiler.trace import (
    JsonlTraceLogger,
    TraceBundleStore,
    build_experiment_manifest,
    build_llm_trace_record,
    build_trace_bundle,
)
from edge_agent_workflow_scheduling.scheduler import BaselineScheduler
from edge_agent_workflow_scheduling.tools import ToolRegistry
from edge_agent_workflow_scheduling.tools.deployment import create_local_tool


def write_report(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def catalog_report(profiles, config):
    expected = set(config["required_local_ids"] + config["required_cloud_ids"])
    if {p.llm_id for p in profiles} != expected:
        raise ValueError("LLM catalog must contain exactly the two local and two cloud IDs")
    reports = {}
    for profile in profiles:
        cloud = profile.llm_id in config["required_cloud_ids"]
        if cloud and (
            profile.provider != "volcengine"
            or profile.platform != "cloud"
            or profile.secret_env_vars != ["ARK_API_KEY"]
            or profile.deployment_config.get("requires_api_key") is not True
        ):
            raise ValueError("cloud profiles must use Volcengine and ARK_API_KEY")
        if not cloud and (profile.provider != "vllm" or profile.platform == "cloud"):
            raise ValueError("local profiles must use the vLLM adapter")
        if "function_calling" not in profile.capabilities:
            raise ValueError("all four profiles must declare Function Calling capability")
        for field in ("enabled", "requires_api_key"):
            if not isinstance(profile.deployment_config.get(field), bool):
                raise ValueError(f"LLM deployment configuration requires a boolean {field}")
        reports[profile.llm_id] = {
            "profile": profile.to_dict(),
            "capability_source": profile.metadata.get("capability_source", "declared_unverified"),
            "context_source": profile.metadata.get("context_limit_source", "declared_unverified"),
            "calibration": {
                name: {
                    "available": bool(values),
                    "values": values,
                    "source": profile.metadata.get(f"{name}_status", "unavailable"),
                }
                for name, values in (
                    ("quality", profile.quality_profile),
                    ("throughput", profile.token_profile),
                    ("energy", profile.energy_profile),
                )
            },
            "credential_check": "environment_reference_only; no client or network required",
        }
    return reports


def offline_contract(deployment, output_dir):
    scheduler = BaselineScheduler("round_robin")
    logger = JsonlTraceLogger(Path(output_dir) / "trace.jsonl")
    logger.clear()
    rows = []
    call = LLMCall(
        llm_call_id="offline-mask-probe",
        run_id="llm-profile-contract-v1",
        agent_id="llm-deployment-validation",
        required_capabilities=["function_calling"],
    )
    for index in range(len(deployment.executors)):
        call = LLMCall(
            llm_call_id=f"offline-llm-{index}",
            run_id="llm-profile-contract-v1",
            agent_id="llm-deployment-validation",
            input_items=[{"role": "user", "content": "17 plus 25?"}],
            input_tokens=8,
            estimated_output_tokens=8,
            required_capabilities=["function_calling"],
        )
        decision = scheduler.schedule(call, resources=deployment.resources)
        result = deployment.executors[decision.selected_target].execute(call, timeout_sec=10)
        logger.write(build_llm_trace_record(llm_call=call, decision=decision, result=result))
        rows.append(
            {"call": call.to_dict(), "decision": decision.to_dict(), "result": result.to_dict()}
        )
    write_report(Path(output_dir) / "execution_records.json", rows)
    mask = deployment.resources.action_mask_details(call)
    return {
        "passed": bool(rows) and all(row["result"]["success"] for row in rows),
        "selected_targets": [row["decision"]["selected_target"] for row in rows],
        "action_mask": mask.as_dict(),
        "rejection_reasons": mask.reasons_by_target(),
        "states": {
            s.profile.llm_id: s.state.to_dict() for s in deployment.resources.llm_snapshots()
        },
        "activation": deployment.reports,
        "network_requests": 0,
        "performance_scope": "synthetic_contract_fixture_not_hardware_measurement",
    }


def function_calling_check(profile, llms, tools, *, samples, output_dir, timeout_sec):
    profile = replace(
        profile,
        deployment_config={
            **profile.deployment_config,
            "model_parameters": {
                **profile.deployment_config.get("model_parameters", {}),
                "tool_choice": "auto",
            },
        },
    )
    # The executor was created with the same profile; only the request's tool_choice changes.
    llms.executors[profile.llm_id].model_parameters = profile.deployment_config["model_parameters"]
    registry = ToolRegistry()
    for name in samples:
        registry.register(
            create_local_tool(
                name,
                input_root=tools.input_root,
                output_dir=Path(output_dir) / "schema_artifacts" / name,
                timeout_sec=timeout_sec,
                options={},
            )
        )
    resources = tools.resources
    for snapshot in llms.resources.llm_snapshots():
        resources.register_llm(snapshot.profile, snapshot.state, replace=True)
    factories = ExecutorFactoryRegistry()
    llms.register_factories(factories)
    tools.register_factories(factories)
    runner = AgentRunner(
        agent_id="local-llm-function-calling-verifier",
        system_instruction=(
            "Use image_preprocess when asked to transform an image. "
            "Execute the Tool, then report its returned output URI. "
            "For arithmetic, answer directly without a Tool."
        ),
        tool_registry=registry,
        resources=resources,
        scheduler=BaselineScheduler("least_queue"),
        executor_pool=ExecutorPool(factories),
        model_name=profile.model,
        max_rounds=3,
        max_tool_calls=2,
        timeout_sec=timeout_sec,
    )
    manifest = build_experiment_manifest(
        experiment_id=f"local-function-calling-{profile.llm_id}",
        dataset_id="llm-deployment-function-calling-v1",
        sample_ids=["tool-needed", "no-tool-needed"],
        runner=runner,
        system_prompt_version="llm-deployment-function-calling-v1",
        user_template="{task}",
        user_template_version="llm-deployment-function-calling-v1",
        llm_profile_version="llm-deployment-v1",
        tool_profile_version="real-local-tools",
        code_version="working-tree",
        mode="live",
        sampling_parameters=profile.deployment_config["model_parameters"],
    )
    fixture = samples["image_preprocess"].arguments["input_uri"]
    scenarios = []
    for name, task, expected in (
        (
            "tool-needed",
            "Convert this local image to grayscale and resize it, "
            f"then report the output URI: {fixture}",
            ["image_preprocess"],
        ),
        ("no-tool-needed", "What is 17 plus 25? Answer with only the number.", []),
    ):
        execution = runner.run(task, task_id=name, run_id=f"{profile.llm_id}-{name}")
        path = Path(output_dir) / name / "trace.json"
        TraceBundleStore(path).write(build_trace_bundle(execution, manifest))
        selected = [r.call.tool_name for r in execution.tool_records]
        passed = (
            execution.agent_run.status.value == "completed"
            and selected == expected
            and (not expected or len(execution.llm_records) >= 2)
        )
        scenarios.append(
            {
                "scenario": name,
                "passed": passed,
                "selected_tools": selected,
                "llm_call_count": len(execution.llm_records),
                "trace": str(path),
                "error_code": execution.agent_run.error_code,
            }
        )
    return {
        "status": "passed" if all(s["passed"] for s in scenarios) else "failed",
        "scenarios": scenarios,
        "real_model_inference": True,
    }


def live_checks(
    profiles, config, *, output_dir, live_local=False, cloud_smoke=False, client_factory=None
):
    checks = {}
    with LLMDeployment(
        profiles,
        mode="live",
        allow_cloud=cloud_smoke,
        health_timeout_sec=config["health_timeout_sec"],
        client_factory=client_factory,
    ) as llms:
        if live_local and any(target in llms.executors for target in config["required_local_ids"]):
            samples = load_deployment_samples(config["tool_consistency_samples"])
            with LocalToolDeployment.from_catalog(
                config["tool_catalog"],
                input_root=Path(config["tool_consistency_samples"]).parent,
                output_dir=Path(output_dir) / "tool_deployment",
                manage_load=False,
                timeout_sec=config["request_timeout_sec"],
            ) as tools:
                tool_checks, _ = self_check_deployment(tools, samples)
                write_report(Path(output_dir) / "tool_deployment/startup.json", tools.startup)
                write_report(Path(output_dir) / "tool_deployment/self_check.json", tool_checks)
                for target in config["required_local_ids"]:
                    if target not in llms.executors:
                        continue
                    if tool_checks["passed"]:
                        check = function_calling_check(
                            llms.resources.llm_snapshot(target).profile,
                            llms,
                            tools,
                            samples=samples,
                            output_dir=Path(output_dir) / target,
                            timeout_sec=config["request_timeout_sec"],
                        )
                    else:
                        check = {"status": "failed", "reason": "tool_self_check_failed"}
                    checks[target] = check
                    llms.reports[target]["function_calling_currently_verified"] = (
                        check["status"] == "passed"
                    )
                    if check["status"] != "passed":
                        llms.reports[target]["online"] = False
                        llms.reports[target]["status"] = "offline"
                        state = llms.resources.llm_snapshot(target).state
                        llms.resources.update_llm_state(replace(state, is_online=False))
        cloud_logger = JsonlTraceLogger(Path(output_dir) / "cloud_trace.jsonl")
        cloud_logger.clear()
        if cloud_smoke:
            for target in config["required_cloud_ids"]:
                if target not in llms.executors:
                    continue
                executor = llms.executors[target]
                executor.model_parameters = {
                    **executor.model_parameters,
                    "max_tokens": config["max_output_tokens"],
                }
                call = LLMCall(
                    llm_call_id=f"cloud-smoke-{target}",
                    run_id="cloud-llm-smoke-v1",
                    agent_id="llm-deployment-validation",
                    input_items=[
                        {
                            "role": "user",
                            "content": "What is 17 plus 25? Answer with only the number.",
                        }
                    ],
                )
                result = executor.execute(call, timeout_sec=config["request_timeout_sec"])
                decision = ScheduleDecision(
                    call_id=call.llm_call_id,
                    call_kind="llm",
                    selected_target=target,
                    policy_name="direct_cloud_smoke",
                    reason="explicit bounded cloud smoke; not a scheduler decision",
                )
                cloud_logger.write(
                    build_llm_trace_record(
                        llm_call=call,
                        decision=decision,
                        result=result,
                        timeout=result.error_code == "timeout",
                    )
                )
                checks[target] = {
                    "status": "passed" if result.success else "failed",
                    "error_code": result.error_code,
                    "function_calling_verified": False,
                }
                if not result.success:
                    llms.reports[target]["online"] = False
                    llms.reports[target]["status"] = "offline"
                    state = llms.resources.llm_snapshot(target).state
                    llms.resources.update_llm_state(replace(state, is_online=False))
        for profile in profiles:
            target = profile.llm_id
            if target not in checks:
                checks[target] = {
                    "status": "skipped",
                    "reason": llms.reports[target].get("reason")
                    if target not in llms.executors
                    else "smoke_not_requested",
                }
        return {
            "activation": llms.reports,
            "checks": checks,
            "states": {s.profile.llm_id: s.state.to_dict() for s in llms.resources.llm_snapshots()},
            "local_function_calling_verified": any(
                checks[target]["status"] == "passed" for target in config["required_local_ids"]
            ),
        }
