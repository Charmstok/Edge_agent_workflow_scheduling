"""Real Tool deployment self-checks, consistency, scheduling and failure probes."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier

from edge_agent_workflow_scheduling.agents.runner import CallExecutionRecord
from edge_agent_workflow_scheduling.common import CallStatus, ScheduleDecision, ToolCall
from edge_agent_workflow_scheduling.profiler.tool_consistency import compare_tool_results
from edge_agent_workflow_scheduling.resources import ToolConsistencySample
from edge_agent_workflow_scheduling.tools.deployment import canonical_digest, canonical_tool_output


def load_deployment_samples(path):
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    samples = {}
    for item in raw["samples"]:
        sample = ToolConsistencySample.from_dict(item)
        arguments = dict(sample.arguments)
        fixture = (path.parent / arguments["input_uri"]).resolve()
        arguments["input_uri"] = fixture.as_uri()
        samples[sample.tool_name] = replace(sample, arguments=arguments)
    return samples


def sample_call(sample, *, run_id, suffix):
    call = ToolCall(
        tool_call_id=f"{run_id}-{sample.tool_name}-{suffix}",
        run_id=run_id,
        agent_id="local-tool-deployment-validation",
        call_id=f"function-{sample.tool_name}-{suffix}",
        tool_name=sample.tool_name,
        arguments=dict(sample.arguments),
        metadata={"task_type": "default", "input_size": "small"},
    )
    call.transition_to(CallStatus.QUEUED)
    return call


def direct_record(deployment, sample, replica_id, suffix, *, timeout_sec=None, barrier=None):
    call = sample_call(sample, run_id="local-self-check-v1", suffix=f"{replica_id}-{suffix}")
    decision = ScheduleDecision(
        call_id=call.tool_call_id,
        call_kind="tool",
        selected_target=replica_id,
        policy_name="local_self_check",
        reason="direct startup/fault validation; not a scheduler decision",
    )
    if barrier is not None:
        barrier.wait(timeout=10)
    result = deployment.executors[replica_id].execute(
        call, timeout_sec=timeout_sec or deployment.timeout_sec
    )
    if call.status == CallStatus.QUEUED and result.success:
        call.transition_to(CallStatus.RUNNING)
    call.transition_to(CallStatus.SUCCEEDED if result.success else CallStatus.FAILED)
    return CallExecutionRecord(call, decision, result)


def self_check_deployment(deployment, samples):
    """Execute the same fixture on every configured slot, then compare replicas."""
    records = []
    per_replica = {}
    for replica_id, worker in deployment.workers.items():
        sample = samples[worker.profile.tool_name]
        count = worker.profile.max_concurrency
        barrier = Barrier(count)
        with ThreadPoolExecutor(max_workers=count) as pool:
            futures = [
                pool.submit(
                    direct_record, deployment, sample, replica_id, str(index), barrier=barrier
                )
                for index in range(count)
            ]
            group = [future.result() for future in futures]
        records.extend(group)
        results = [record.result for record in group]
        digests = []
        for result in results:
            if result.success:
                digests.append(
                    canonical_digest(
                        canonical_tool_output(
                            sample.tool_name, result.output, artifact_root=worker.output_dir
                        )
                    )
                )
        per_replica[replica_id] = {
            "passed": all(result.success for result in results) and len(set(digests)) == 1,
            "slot_pids": [result.metadata.get("worker_pid") for result in results],
            "distinct_slot_count": len({result.metadata.get("worker_pid") for result in results}),
            "configured_slots": count,
            "peak_running": worker.peak_running,
            "output_schema_checked": all(result.success for result in results),
            "timeout_contract": (
                "queue_and_worker_total_deadline; process_group_killed_on_rpc_timeout"
            ),
            "results": [result.to_dict() for result in results],
        }
        if per_replica[replica_id]["distinct_slot_count"] != count:
            per_replica[replica_id]["passed"] = False
    consistency = {}
    for name, sample in samples.items():
        representative = [
            record.result
            for record in records
            if record.call.tool_name == name and record.call.tool_call_id.endswith("-0")
        ]
        consistency[name] = compare_tool_results(sample, representative)
        for result in representative:
            if not consistency[name]["passed"]:
                per_replica[result.replica_id]["passed"] = False
    for replica_id, report in per_replica.items():
        results = [record.result for record in records if record.result.replica_id == replica_id]
        deployment.measure_profile(replica_id, results if report["passed"] else [])
    return {
        "passed": all(report["passed"] for report in per_replica.values()),
        "replicas": per_replica,
        "consistency": consistency,
    }, records


def run_scheduled_batch(deployment, samples, scheduler, *, run_id):
    calls = [
        sample_call(sample, run_id=run_id, suffix=str(index))
        for sample in samples.values()
        for index in range(2)
    ]
    submissions = deployment.submit_batch(calls, scheduler)
    records = []
    for call, decision, future in submissions:
        result = future.result()
        call.transition_to(CallStatus.SUCCEEDED if result.success else CallStatus.FAILED)
        records.append(CallExecutionRecord(call, decision, result))
    return records


def failure_probes(deployment, samples):
    """Exercise invalid arguments, total timeout and owned worker exit, then recover."""
    image_ids = [
        replica_id
        for replica_id, worker in deployment.workers.items()
        if worker.profile.tool_name == "image_preprocess"
    ]
    sample = samples["image_preprocess"]
    records = []
    reports = {}
    from edge_agent_workflow_scheduling.executors.local_deployment import LocalToolDeployment

    ocr = next(
        worker.profile
        for worker in deployment.workers.values()
        if worker.profile.tool_name == "ocr"
    )
    unavailable = replace(
        ocr,
        replica_id=f"{ocr.replica_id}-missing-dependency",
        deployment_config={
            "tool_options": {"executable": "__missing_tesseract_dependency_probe__"}
        },
    )
    with LocalToolDeployment(
        [unavailable],
        input_root=deployment.input_root,
        output_dir=deployment.output_dir / "fault-probes",
        timeout_sec=deployment.timeout_sec,
    ) as isolated:
        record = direct_record(isolated, samples["ocr"], unavailable.replica_id, "dependency")
        records.append(record)
        mask = isolated.resources.action_mask_details(record.call)
        reports["missing_dependency"] = {
            "passed": not record.result.success
            and record.result.error_code == "dependency_unavailable"
            and not any(mask.values),
            "startup": isolated.startup,
            "result": record.result.to_dict(),
            "action_mask": mask.as_dict(),
        }
    invalid = replace(sample, arguments={"input_uri": sample.arguments["input_uri"]})
    record = direct_record(deployment, invalid, image_ids[0], "invalid-arguments")
    records.append(record)
    reports["invalid_arguments"] = {
        "passed": not record.result.success and record.result.error_code == "invalid_arguments",
        "result": record.result.to_dict(),
    }
    slow = replace(sample, arguments={**sample.arguments, "operation_repeat": 100000})
    record = direct_record(deployment, slow, image_ids[0], "timeout", timeout_sec=0.01)
    records.append(record)
    reports["timeout"] = {
        "passed": not record.result.success and record.result.error_code == "timeout",
        "result": record.result.to_dict(),
    }
    # A worker exit is injected only into an owned process created by this deployment.
    deployment.workers[image_ids[1]].slots[0].close()
    record = direct_record(deployment, sample, image_ids[1], "worker-exit")
    records.append(record)
    mask = deployment.resources.action_mask_details(record.call)
    reports["worker_exit"] = {
        "passed": not record.result.success
        and record.result.error_code == "worker_exited"
        and not mask.as_dict()[image_ids[1]],
        "result": record.result.to_dict(),
        "action_mask": mask.as_dict(),
        "rejection_reasons": mask.reasons_by_target(),
    }
    for replica_id in image_ids:
        deployment.workers[replica_id].start()
        recovered = direct_record(deployment, sample, replica_id, "recovered")
        records.append(recovered)
        reports[f"recovered-{replica_id}"] = {
            "passed": recovered.result.success,
            "result": recovered.result.to_dict(),
        }
    return reports, records
