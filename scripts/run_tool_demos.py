"""Run real local OCR/PDF replicas, record measured work and verify output consistency."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from edge_agent_workflow_scheduling.common import CallStatus, ToolCall
from edge_agent_workflow_scheduling.executors import LocalToolExecutor
from edge_agent_workflow_scheduling.profiler import JsonlTraceLogger, build_tool_trace_record
from edge_agent_workflow_scheduling.profiler.tool_consistency import compare_tool_results
from edge_agent_workflow_scheduling.resources import (
    ResourceRegistry,
    ToolConsistencySample,
    ToolReplicaProfile,
)
from edge_agent_workflow_scheduling.scheduler import BaselineScheduler
from edge_agent_workflow_scheduling.tools import (
    DocumentToolConfig,
    OCRConfig,
    OCRTool,
    PDFParseTool,
    ToolRegistry,
)
from edge_agent_workflow_scheduling.workers import LocalWorker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/tool_demos"))
    parser.add_argument("--samples", type=Path, default=Path("configs/tool_consistency_v1.json"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--scale", choices=("small", "medium", "large"), default="small")
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument(
        "--tools", nargs="+", choices=("ocr", "pdf_parse"), default=["ocr", "pdf_parse"]
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    run_id = f"tools-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid4().hex[:8]}"
    directory = args.output_dir / run_id
    directory.mkdir(parents=True)
    root = args.samples.parent.resolve()
    samples = json.loads(args.samples.read_text(encoding="utf-8"))
    registry = ResourceRegistry()
    scheduler = BaselineScheduler("round_robin")
    report = {
        "experiment_id": run_id,
        "mode": "local_real_tools",
        "host": platform.platform(),
        "batch_size": args.batch_size,
        "scale": args.scale,
        "tools": {},
    }
    traces = JsonlTraceLogger(directory / "trace.jsonl")
    for data in samples["samples"]:
        sample = ToolConsistencySample.from_dict(data)
        if sample.tool_name not in args.tools:
            continue
        source = root / sample.arguments["input_uri"].replace("-small.", f"-{args.scale}.")
        arguments = {"input_uri": source.as_uri()}
        expected = dict(sample.expected)
        expected["input_count"] = args.batch_size
        if sample.tool_name == "ocr":
            factor = {"small": 1, "medium": 4, "large": 16}[args.scale]
            expected.update(
                image_count=args.batch_size, image_pixels=307200 * factor * args.batch_size
            )
        else:
            expected["page_count"] = {"small": 1, "medium": 3, "large": 6}[
                args.scale
            ] * args.batch_size
        if args.batch_size > 1:
            batch = directory / f"{sample.tool_name}-batch.json"
            batch.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "input_uris": [source.as_uri()] * args.batch_size,
                    }
                ),
                encoding="utf-8",
            )
            arguments = {"input_uri": batch.resolve().as_uri()}
        sample = replace(
            sample,
            sample_id=sample.sample_id.replace("-small-", f"-{args.scale}-")
            + f"-batch-{args.batch_size}",
            arguments=arguments,
            expected=expected,
        )
        registry.register_consistency_sample(sample)
        executors = {}
        try:
            for replica_index in range(2):
                replica_id = f"{sample.tool_name}-local-{replica_index}"
                output_dir = directory / replica_id
                tool = (
                    OCRTool(
                        OCRConfig(
                            output_dir=output_dir, local_root=root, timeout_sec=args.timeout_sec
                        )
                    )
                    if sample.tool_name == "ocr"
                    else PDFParseTool(
                        DocumentToolConfig(
                            output_dir=output_dir, local_root=root, timeout_sec=args.timeout_sec
                        )
                    )
                )
                tool.check_available()
                tools = ToolRegistry()
                tools.register(tool)
                profile = ToolReplicaProfile(
                    replica_id=replica_id,
                    tool_name=tool.tool_name,
                    node_id="local",
                    platform=platform.system(),
                    implementation_version=tool.implementation_version,
                    executor_type="local",
                    metadata={"source": "local_real", "quality": "uncalibrated"},
                )
                registry.register_tool_replica(profile)
                executors[replica_id] = LocalToolExecutor(LocalWorker(profile, tools))
        except ModuleNotFoundError as exc:
            report["tools"][sample.tool_name] = {"status": "skipped", "reason": str(exc)}
            continue
        results = []
        for repeat in range(2):
            call = ToolCall(
                tool_call_id=f"{run_id}-{sample.tool_name}-{repeat}",
                run_id=run_id,
                call_id=f"function-{sample.tool_name}-{repeat}",
                agent_id="tool-validation",
                tool_name=sample.tool_name,
                arguments=arguments,
                metadata={"task_type": "tool_consistency"},
            )
            call.transition_to(CallStatus.QUEUED)
            decision = scheduler.schedule(call, resources=registry)
            call.transition_to(CallStatus.RUNNING)
            result = executors[decision.selected_target].execute(call, timeout_sec=args.timeout_sec)
            call.transition_to(CallStatus.SUCCEEDED if result.success else CallStatus.FAILED)
            results.append(result)
            traces.write(
                build_tool_trace_record(
                    tool_call=call,
                    decision=decision,
                    result=result,
                    timeout=result.error_code == "timeout",
                )
            )
            with (directory / "results.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "call": call.to_dict(),
                            "decision": decision.to_dict(),
                            "result": result.to_dict(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        consistency = compare_tool_results(sample, results)
        report["tools"][sample.tool_name] = {
            "status": "passed" if consistency["passed"] else "failed",
            "sample": sample.to_dict(),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "timeout_sec": args.timeout_sec,
            "inline_text_chars": tool.config.inline_text_chars,
            "tool_schema": tool.spec,
            "resource_profiles": [executor.profile.to_dict() for executor in executors.values()],
            "consistency": consistency,
            "execution_time_sec": [result.execution_time_sec for result in results],
            "over_10_sec": [result.execution_time_sec >= 10 for result in results],
            "energy_source": "unavailable",
            "quality_profile": "uncalibrated",
        }
    (directory / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Artifacts: {directory}")
    if any(value["status"] == "failed" for value in report["tools"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
