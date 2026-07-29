"""Run the local mixed LLM/real-Tool scheduling demo."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from edge_agent_workflow_scheduling.agents import (
    LLMCallTemplate,
    SimulatedAgent,
    ToolCallTemplate,
)
from edge_agent_workflow_scheduling.common import CallStatus, LLMCall, ToolCall, TraceRecord
from edge_agent_workflow_scheduling.executors import LocalToolExecutor, MockLLMExecutor
from edge_agent_workflow_scheduling.llm import MockLLMRuntime
from edge_agent_workflow_scheduling.profiler import (
    JsonlTraceLogger,
    build_llm_trace_record,
    build_tool_trace_record,
)
from edge_agent_workflow_scheduling.queue import InMemoryCallQueue
from edge_agent_workflow_scheduling.resources import (
    LLMInstanceProfile,
    ResourceRegistry,
    ToolReplicaProfile,
)
from edge_agent_workflow_scheduling.scheduler import (
    DEFAULT_SCHEDULER_POLICY_REGISTRY,
    BaselineScheduler,
)
from edge_agent_workflow_scheduling.tools import (
    ImagePreprocessConfig,
    ImagePreprocessTool,
    ToolRegistry,
)
from edge_agent_workflow_scheduling.workers import LocalWorker


@dataclass(frozen=True, slots=True)
class DemoSummary:
    total_records: int
    average_latency_sec: float
    success_rate: float
    worker_counts: dict[str, int]
    llm_counts: dict[str, int]
    trace_path: Path


def run_demo(
    *,
    policy: str = "round_robin",
    trace_path: Path = Path("data/traces/first_demo.jsonl"),
    data_dir: Path = Path("data/first_demo"),
    runs_per_agent: int = 10,
) -> DemoSummary:
    if runs_per_agent < 1:
        raise ValueError("runs_per_agent must be at least 1")

    input_dir = (data_dir / "inputs").resolve()
    output_dir = (data_dir / "outputs").resolve()
    _create_images(input_dir, runs_per_agent)

    queue = InMemoryCallQueue()
    for index, agent in enumerate(_create_agents(input_dir)):
        queue.push_many(
            agent.generate_calls(
                runs_per_agent,
                start_sequence_id=index * runs_per_agent,
            )
        )

    runtimes = _create_runtimes()
    workers = _create_workers(output_dir)
    llm_executors = {llm_id: MockLLMExecutor(runtime) for llm_id, runtime in runtimes.items()}
    tool_executors = {
        replica_id: LocalToolExecutor(worker) for replica_id, worker in workers.items()
    }
    resources = _create_resource_registry(runtimes, workers)
    scheduler = BaselineScheduler(policy)
    logger = JsonlTraceLogger(trace_path)
    logger.clear()
    records: list[TraceRecord] = []
    worker_counts: Counter[str] = Counter()
    llm_counts: Counter[str] = Counter()

    while call := queue.pop():
        call.transition_to(CallStatus.RUNNING)
        if isinstance(call, LLMCall):
            decision = scheduler.schedule(
                call,
                resources=resources,
            )
            result = llm_executors[decision.selected_target].execute(call)
            record = build_llm_trace_record(llm_call=call, decision=decision, result=result)
            llm_counts[decision.selected_target] += 1
        elif isinstance(call, ToolCall):
            decision = scheduler.schedule(
                call,
                resources=resources,
            )
            result = tool_executors[decision.selected_target].execute(call)
            record = build_tool_trace_record(tool_call=call, decision=decision, result=result)
            worker_counts[decision.selected_target] += 1
        else:
            raise TypeError(f"unsupported call type: {type(call)!r}")
        call.transition_to(CallStatus.SUCCEEDED if result.success else CallStatus.FAILED)
        records.append(record)
        logger.write(record)

    return DemoSummary(
        total_records=len(records),
        average_latency_sec=sum(record.total_latency_sec for record in records) / len(records),
        success_rate=sum(record.success for record in records) / len(records),
        worker_counts=dict(sorted(worker_counts.items())),
        llm_counts=dict(sorted(llm_counts.items())),
        trace_path=trace_path,
    )


def _create_agents(input_dir: Path) -> list[SimulatedAgent]:
    agents = []
    for model_name, agent_id, input_tokens, priority in (
        ("qwen-7b", "agent_7b", 512, 1),
        ("qwen-27b", "agent_27b", 1024, 2),
    ):
        agents.append(
            SimulatedAgent(
                agent_id=agent_id,
                template=ToolCallTemplate(
                    tool_name="image_preprocess",
                    input_uri_prefix=input_dir.as_uri(),
                    input_size_mb=0.02,
                    image_count=1,
                    deadline_sec=30,
                    priority=priority,
                    file_extension="png",
                ),
                llm_template=LLMCallTemplate(
                    model_name=model_name,
                    input_tokens=input_tokens,
                    estimated_output_tokens=input_tokens // 4,
                    context_length=input_tokens * 4,
                    deadline_sec=20,
                    priority=priority,
                ),
            )
        )
    return agents


def _create_runtimes() -> dict[str, MockLLMRuntime]:
    runtimes = (
        MockLLMRuntime(
            _create_llm_profile("llm_qwen_7b_mock", "qwen-7b", 7, 160, 0.72),
            160,
        ),
        MockLLMRuntime(
            _create_llm_profile("llm_qwen_27b_mock", "qwen-27b", 27, 80, 0.86),
            80,
        ),
    )
    return {runtime.llm_id: runtime for runtime in runtimes}


def _create_llm_profile(
    llm_id: str,
    model: str,
    model_size_b: float,
    tokens_per_sec: float,
    quality: float,
) -> LLMInstanceProfile:
    return LLMInstanceProfile(
        llm_id=llm_id,
        provider="mock",
        model=model,
        node_id="laptop",
        platform="macos",
        executor_type="mock",
        base_url=f"mock://{llm_id}",
        model_size_b=model_size_b,
        capabilities=["function_calling"],
        context_window_tokens=32768,
        quality_profile={"image_preprocess": quality},
        token_profile={"tokens_per_sec": tokens_per_sec},
        energy_profile={"joules_per_token": model_size_b / 1000},
    )


def _create_workers(output_dir: Path) -> dict[str, LocalWorker]:
    workers = (
        LocalWorker(
            _create_tool_replica_profile("worker_local_1", "macbook_local", 0.004),
            _create_tool_registry(output_dir),
        ),
        LocalWorker(
            _create_tool_replica_profile("worker_local_2", "ubuntu_logical", 0.006),
            _create_tool_registry(output_dir),
            artificial_delay_sec=0.001,
        ),
    )
    return {worker.replica_id: worker for worker in workers}


def _create_tool_replica_profile(
    replica_id: str,
    node_id: str,
    execution_time_sec: float,
) -> ToolReplicaProfile:
    return ToolReplicaProfile(
        replica_id=replica_id,
        tool_name="image_preprocess",
        node_id=node_id,
        platform="macos" if node_id == "macbook_local" else "ubuntu",
        implementation_version="pillow-12.3.0",
        executor_type="local",
        capabilities=["grayscale", "resize", "blur", "threshold", "edge_detect"],
        latency_profile={"execution_time_sec": execution_time_sec},
        energy_profile={"joules_per_call": 0.02},
        quality_profile={"image_fidelity": 1.0},
        deployment_config={
            "requirements_file": "requirements-edge.txt",
            "system_packages": ["libjpeg", "zlib", "freetype"],
        },
    )


def _create_resource_registry(
    runtimes: dict[str, MockLLMRuntime],
    workers: dict[str, LocalWorker],
) -> ResourceRegistry:
    resources = ResourceRegistry()
    for runtime in runtimes.values():
        resources.register_llm(runtime.to_profile(), runtime.get_state())
    for worker in workers.values():
        resources.register_tool_replica(worker.to_profile(), worker.get_state())
    return resources


def _create_tool_registry(output_dir: Path) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ImagePreprocessTool(
            ImagePreprocessConfig(
                output_dir=output_dir,
                operations=("grayscale", "blur", "threshold"),
                operation_repeat=2,
            )
        )
    )
    return registry


def _create_images(input_dir: Path, runs_per_agent: int) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    for agent_id, offset, base_size in (
        ("agent_7b", 0, 32),
        ("agent_27b", runs_per_agent, 48),
    ):
        for sequence_id in range(offset, offset + runs_per_agent):
            size = base_size + sequence_id % 4 * 8
            color = (sequence_id * 31 % 255, 90, 170)
            Image.new("RGB", (size, size), color).save(
                input_dir / f"{agent_id}_{sequence_id:04d}.png"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        default="round_robin",
        choices=DEFAULT_SCHEDULER_POLICY_REGISTRY.available_policies(),
    )
    parser.add_argument("--runs-per-agent", type=int, default=10)
    parser.add_argument("--trace-path", type=Path, default=Path("data/traces/first_demo.jsonl"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/first_demo"))
    args = parser.parse_args()
    summary = run_demo(
        policy=args.policy,
        trace_path=args.trace_path,
        data_dir=args.data_dir,
        runs_per_agent=args.runs_per_agent,
    )
    print(f"trace_path: {summary.trace_path}")
    print(f"total_records: {summary.total_records}")
    print(f"average_latency_sec: {summary.average_latency_sec:.6f}")
    print(f"success_rate: {summary.success_rate:.2%}")
    print(f"worker_counts: {summary.worker_counts}")
    print(f"llm_counts: {summary.llm_counts}")


if __name__ == "__main__":
    main()
