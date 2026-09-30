from __future__ import annotations

import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from time import monotonic, sleep

import pytest

from edge_agent_workflow_scheduling.agents import (
    AgentRunner,
    ScriptedFunctionCall,
    ScriptedLLMBackend,
)
from edge_agent_workflow_scheduling.common import ToolCall
from edge_agent_workflow_scheduling.config import load_tool_profiles
from edge_agent_workflow_scheduling.executors import (
    BackendLLMExecutor,
    ExecutorFactoryRegistry,
    ExecutorPool,
)
from edge_agent_workflow_scheduling.executors.local_deployment import LocalToolDeployment
from edge_agent_workflow_scheduling.profiler.local_deployment import (
    load_deployment_samples,
    run_scheduled_batch,
    self_check_deployment,
)
from edge_agent_workflow_scheduling.profiler.trace import build_tool_trace_record
from edge_agent_workflow_scheduling.resources import LLMInstanceProfile
from edge_agent_workflow_scheduling.scheduler import BaselineScheduler
from edge_agent_workflow_scheduling.tools import ToolRegistry
from edge_agent_workflow_scheduling.tools.deployment import canonical_tool_output, create_local_tool

ROOT = Path(__file__).resolve().parents[1]


def image_profiles():
    return load_tool_profiles(ROOT / "configs/tool_profiles.toml")[:2]


def samples():
    return load_deployment_samples(ROOT / "configs/tool_deployment_samples_v1.json")


def image_call(identifier, **updates):
    return ToolCall(
        tool_call_id=identifier,
        run_id="test",
        call_id=identifier,
        agent_id="test",
        tool_name="image_preprocess",
        arguments={**samples()["image_preprocess"].arguments, **updates},
    )


def test_all_catalog_replicas_have_separate_deployment_paths():
    profiles = load_tool_profiles(ROOT / "configs/tool_profiles.toml")
    assert len(profiles) == 8
    assert {
        name: sum(p.tool_name == name for p in profiles)
        for name in ("image_preprocess", "ocr", "pdf_parse", "pdf_render")
    } == {"image_preprocess": 2, "ocr": 2, "pdf_parse": 2, "pdf_render": 2}
    assert len({p.deployment_config["work_dir"] for p in profiles}) == 8
    assert all(p.max_concurrency == 2 for p in profiles)


def test_real_replicas_self_check_concurrency_consistency_and_scheduling(tmp_path):
    profiles = image_profiles()
    local_samples = {"image_preprocess": samples()["image_preprocess"]}
    with LocalToolDeployment(
        profiles, input_root=ROOT / "configs", output_dir=tmp_path
    ) as deployment:
        checks, records = self_check_deployment(deployment, local_samples)
        assert checks["passed"] and checks["consistency"]["image_preprocess"]["matching_content"]
        assert all(item["peak_running"] == 2 for item in checks["replicas"].values())
        pids = [result.metadata["worker_pid"] for result in (r.result for r in records)]
        assert len(set(pids)) == 4
        assert len({r.result.metadata["worker_cwd"] for r in records}) == 4
        for record in records:
            result = record.result
            assert result.input_transfer_time_sec == result.output_transfer_time_sec == 0
            assert result.metadata["resource_sampling"]["max_rss_kib"] > 0
            assert result.metadata["resource_sampling"]["cpu_time_sec"] > 0
            trace = build_tool_trace_record(
                tool_call=record.call, decision=record.decision, result=result
            )
            assert trace.result_metadata["deployment_kind"] == "same_host_logical_replica"
        for policy in ("round_robin", "least_queue"):
            group = run_scheduled_batch(
                deployment, local_samples, BaselineScheduler(policy), run_id=policy
            )
            assert {r.result.replica_id for r in group} == {p.replica_id for p in profiles}
            assert all(r.result.success for r in group)
        assert all(
            s.state.queue_len == s.state.running_tasks == 0
            for s in deployment.resources.tool_snapshots()
        )
    assert all(
        slot.process.poll() is not None
        for worker in deployment.workers.values()
        for slot in worker.slots
    )


def test_missing_dependency_is_offline_and_returns_structured_failure(tmp_path):
    original = next(
        p for p in load_tool_profiles(ROOT / "configs/tool_profiles.toml") if p.tool_name == "ocr"
    )
    profile = replace(
        original, deployment_config={"tool_options": {"executable": "__missing_tesseract__"}}
    )
    with LocalToolDeployment(
        [profile], input_root=ROOT / "configs", output_dir=tmp_path
    ) as deployment:
        assert (
            deployment.startup[profile.replica_id]["failure"]["error_code"]
            == "dependency_unavailable"
        )
        call = ToolCall(
            tool_call_id="missing",
            run_id="test",
            call_id="missing",
            agent_id="test",
            tool_name="ocr",
            arguments=samples()["ocr"].arguments,
        )
        assert not any(deployment.resources.action_mask_details(call).values)
        result = deployment.executors[profile.replica_id].execute(call)
        assert not result.success and result.error_code == "dependency_unavailable"
        assert result.output is None


def test_queue_timeout_is_measured_and_total_deadline_kills_only_owned_worker(tmp_path):
    profile = replace(image_profiles()[0], max_concurrency=1)
    with LocalToolDeployment(
        [profile], input_root=ROOT / "configs", output_dir=tmp_path
    ) as deployment:
        executor = deployment.executors[profile.replica_id]
        worker = deployment.workers[profile.replica_id]
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(
                executor.execute, image_call("slow", operation_repeat=100000), timeout_sec=0.3
            )
            end = monotonic() + 1
            while worker.running != 1 and monotonic() < end:
                sleep(0.005)
            assert worker.running == 1
            queued = executor.execute(image_call("queued"), timeout_sec=0.01)
            assert not queued.success and queued.error_code == "timeout"
            assert queued.queue_wait_time_sec >= 0.009
            assert queued.execution_time_sec == 0
            timed_out = first.result(timeout=2)
        assert not timed_out.success and timed_out.error_code == "timeout"
        assert worker.slots[0].process.poll() is not None
        assert not any(deployment.resources.action_mask_details(image_call("next")).values)
        assert worker.queue_len == worker.running == 0
        worker.start()
        assert executor.execute(image_call("recovered")).success


def test_worker_exit_immediately_masks_replica_and_invalid_input_does_not_fake_success(tmp_path):
    with LocalToolDeployment(
        image_profiles(), input_root=ROOT / "configs", output_dir=tmp_path
    ) as deployment:
        worker = next(iter(deployment.workers.values()))
        worker.slots[0].close()
        mask = deployment.resources.action_mask_details(image_call("exit"))
        assert not mask.as_dict()[worker.profile.replica_id]
        result = deployment.executors[worker.profile.replica_id].execute(image_call("exit"))
        assert not result.success and result.error_code == "worker_exited"
        other = list(deployment.executors.values())[1]
        bad = image_call("invalid-file", input_uri=str(tmp_path / "missing.png"))
        result = other.execute(bad)
        assert not result.success and result.error_code and result.output is None
        trace = build_tool_trace_record(
            tool_call=bad,
            result=result,
            decision=replace(
                BaselineScheduler("round_robin").schedule(bad, resources=deployment.resources),
                selected_target=other.profile.replica_id,
            ),
        )
        assert trace.error_code == result.error_code


def test_missing_fixture_self_check_masks_failed_replicas(tmp_path):
    sample = replace(
        samples()["image_preprocess"],
        arguments={
            **samples()["image_preprocess"].arguments,
            "input_uri": str(tmp_path / "missing.png"),
        },
    )
    with LocalToolDeployment(
        image_profiles(), input_root=ROOT / "configs", output_dir=tmp_path
    ) as deployment:
        checks, records = self_check_deployment(deployment, {"image_preprocess": sample})
        assert not checks["passed"] and all(not r.result.success for r in records)
        assert not any(
            deployment.resources.action_mask_details(image_call("missing-fixture")).values
        )


def test_artifacts_must_stay_inside_replica_root(tmp_path):
    with pytest.raises(ValueError, match="escaped"):
        canonical_tool_output(
            "image_preprocess",
            {"output_uri": str(ROOT / "configs/workload_fixtures_v1/alpha-small.png")},
            artifact_root=tmp_path,
        )


def test_repeated_pdf_render_call_preserves_prior_artifacts_and_same_content(tmp_path):
    if shutil.which("pdftoppm") is None:
        pytest.skip("Poppler is not available")
    profile = next(
        p
        for p in load_tool_profiles(ROOT / "configs/tool_profiles.toml")
        if p.tool_name == "pdf_render"
    )
    profile = replace(profile, max_concurrency=1)
    with LocalToolDeployment(
        [profile], input_root=ROOT / "configs", output_dir=tmp_path
    ) as deployment:
        executor = deployment.executors[profile.replica_id]

        def invoke():
            return executor.execute(
                ToolCall(
                    tool_call_id="repeated",
                    run_id="test",
                    call_id="repeated",
                    agent_id="test",
                    tool_name="pdf_render",
                    arguments=samples()["pdf_render"].arguments,
                )
            )

        first, second = invoke(), invoke()
        assert first.success and second.success
        assert (
            first.output["documents"][0]["pages"][0]["image_uri"]
            != (second.output["documents"][0]["pages"][0]["image_uri"])
        )
        assert canonical_tool_output("pdf_render", first.output) == canonical_tool_output(
            "pdf_render", second.output
        )


def test_agent_runner_uses_isolated_replicas_without_double_counting_load(tmp_path):
    profiles = (
        replace(image_profiles()[0], max_concurrency=1),
        replace(image_profiles()[1], max_concurrency=1),
    )
    with LocalToolDeployment(
        list(profiles), input_root=ROOT / "configs", output_dir=tmp_path, manage_load=False
    ) as deployment:
        registry = ToolRegistry()
        registry.register(
            create_local_tool(
                "image_preprocess",
                input_root=ROOT / "configs",
                output_dir=tmp_path / "unused-definition",
                timeout_sec=30,
            )
        )
        llm = LLMInstanceProfile(
            llm_id="scripted",
            provider="scripted",
            model="scripted",
            node_id="local",
            platform="linux",
            executor_type="scripted",
            capabilities=["function_calling"],
            token_profile={"tokens_per_sec": 100.0},
        )
        deployment.resources.register_llm(llm)
        backend = ScriptedLLMBackend.multiple_tools(
            [
                ScriptedFunctionCall(
                    call_id=f"function-{i}",
                    name="image_preprocess",
                    arguments=samples()["image_preprocess"].arguments,
                )
                for i in range(2)
            ],
            final_text="Done",
        )
        factories = ExecutorFactoryRegistry()
        factories.register_llm("scripted", lambda profile: BackendLLMExecutor(profile, backend))
        deployment.register_factories(factories)
        runner = AgentRunner(
            agent_id="test",
            system_instruction="Use tools",
            tool_registry=registry,
            resources=deployment.resources,
            scheduler=BaselineScheduler("round_robin"),
            executor_pool=ExecutorPool(factories),
        )
        execution = runner.run("Preprocess both", run_id="agent-smoke", task_id="agent-smoke")
        assert execution.agent_run.status.value == "completed"
        assert {r.result.replica_id for r in execution.tool_records} == {
            p.replica_id for p in profiles
        }
        assert all(
            s.state.queue_len == s.state.running_tasks == 0
            for s in deployment.resources.tool_snapshots()
        )


def test_timeout_kills_nested_backend_process_group(tmp_path):
    # A deterministic test backend spawns a descendant; the production Tool wrapper
    # and worker deadline must terminate the whole owned group, not just its parent.
    pid_file = tmp_path / "descendant.pid"
    backend = tmp_path / "fake_tesseract"
    backend.write_text(f"""#!{sys.executable}
import subprocess, sys, time
from pathlib import Path
if '--version' in sys.argv:
    print('test-tesseract 1')
elif '--list-langs' in sys.argv:
    print('List of available languages:\\neng')
else:
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    Path({str(pid_file)!r}).write_text(str(child.pid))
    time.sleep(60)
""")
    backend.chmod(0o755)
    profile = next(
        p for p in load_tool_profiles(ROOT / "configs/tool_profiles.toml") if p.tool_name == "ocr"
    )
    profile = replace(
        profile, max_concurrency=1, deployment_config={"tool_options": {"executable": str(backend)}}
    )
    with LocalToolDeployment(
        [profile], input_root=ROOT / "configs", output_dir=tmp_path / "deployment"
    ) as deployment:
        call = ToolCall(
            tool_call_id="nested",
            run_id="test",
            call_id="nested",
            agent_id="test",
            tool_name="ocr",
            arguments=samples()["ocr"].arguments,
        )
        result = deployment.executors[profile.replica_id].execute(call, timeout_sec=0.3)
        assert not result.success and result.error_code == "timeout"
        assert pid_file.exists()
        stat = Path(f"/proc/{pid_file.read_text()}/stat")
        deadline = monotonic() + 1
        while (
            stat.exists()
            and stat.read_text().split(") ")[1].split()[0] != "Z"
            and monotonic() < deadline
        ):
            sleep(0.01)
        assert not stat.exists() or stat.read_text().split(") ")[1].split()[0] == "Z"
