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
from edge_agent_workflow_scheduling.common import LLMCall, ToolCall, TraceRecord
from edge_agent_workflow_scheduling.llm import MockLLMRuntime
from edge_agent_workflow_scheduling.profiler import (
    JsonlTraceLogger,
    build_llm_trace_record,
    build_tool_trace_record,
)
from edge_agent_workflow_scheduling.queue import InMemoryWorkflowQueue
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
    steps_per_agent: int = 10,
) -> DemoSummary:
    if steps_per_agent < 1:
        raise ValueError("steps_per_agent must be at least 1")

    input_dir = (data_dir / "inputs").resolve()
    output_dir = (data_dir / "outputs").resolve()
    _create_images(input_dir, steps_per_agent)

    queue = InMemoryWorkflowQueue()
    for index, agent in enumerate(_create_agents(input_dir)):
        queue.push_many(
            agent.generate_workflow_steps(
                steps_per_agent,
                start_sequence_id=index * steps_per_agent,
            )
        )

    runtimes = _create_runtimes()
    workers = _create_workers(output_dir)
    scheduler = BaselineScheduler(policy)
    logger = JsonlTraceLogger(trace_path)
    logger.clear()
    records: list[TraceRecord] = []
    worker_counts: Counter[str] = Counter()
    llm_counts: Counter[str] = Counter()

    while step := queue.pop():
        if isinstance(step, LLMCall):
            decision = scheduler.schedule(
                step,
                llm_states=[runtime.get_state() for runtime in runtimes.values()],
                worker_states=[],
            )
            result = runtimes[decision.selected_target].generate(step)
            record = build_llm_trace_record(llm_call=step, decision=decision, result=result)
            llm_counts[decision.selected_target] += 1
        elif isinstance(step, ToolCall):
            decision = scheduler.schedule(
                step,
                llm_states=[],
                worker_states=[worker.get_state() for worker in workers.values()],
            )
            result = workers[decision.selected_target].run_tool(step)
            record = build_tool_trace_record(tool_call=step, decision=decision, result=result)
            worker_counts[decision.selected_target] += 1
        else:
            raise TypeError(f"unsupported step type: {type(step)!r}")
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
                    tool_type="image_preprocess",
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
        MockLLMRuntime("llm_qwen_7b_mock", "qwen-7b", "laptop", 160),
        MockLLMRuntime("llm_qwen_27b_mock", "qwen-27b", "laptop", 80),
    )
    return {runtime.llm_id: runtime for runtime in runtimes}


def _create_workers(output_dir: Path) -> dict[str, LocalWorker]:
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
    workers = (
        LocalWorker(
            "worker_local_1",
            registry.supported_tools(),
            tool_executors=registry.as_executor_mapping(),
        ),
        LocalWorker(
            "worker_local_2",
            registry.supported_tools(),
            artificial_delay_sec=0.001,
            tool_executors=registry.as_executor_mapping(),
        ),
    )
    return {worker.worker_id: worker for worker in workers}


def _create_images(input_dir: Path, steps_per_agent: int) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    for agent_id, offset, base_size in (
        ("agent_7b", 0, 32),
        ("agent_27b", steps_per_agent, 48),
    ):
        for sequence_id in range(offset, offset + steps_per_agent):
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
    parser.add_argument("--steps-per-agent", type=int, default=10)
    parser.add_argument("--trace-path", type=Path, default=Path("data/traces/first_demo.jsonl"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/first_demo"))
    args = parser.parse_args()
    summary = run_demo(
        policy=args.policy,
        trace_path=args.trace_path,
        data_dir=args.data_dir,
        steps_per_agent=args.steps_per_agent,
    )
    print(f"trace_path: {summary.trace_path}")
    print(f"total_records: {summary.total_records}")
    print(f"average_latency_sec: {summary.average_latency_sec:.6f}")
    print(f"success_rate: {summary.success_rate:.2%}")
    print(f"worker_counts: {summary.worker_counts}")
    print(f"llm_counts: {summary.llm_counts}")


if __name__ == "__main__":
    main()
