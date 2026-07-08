"""Run the first local end-to-end workflow scheduling demo."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from edge_agent_workflow_scheduling.agents import (
    LLMCallTemplate,
    SimulatedAgent,
    ToolCallTemplate,
)
from edge_agent_workflow_scheduling.common import (
    LLMCall,
    LLMInstanceState,
    ToolCall,
    TraceRecord,
    WorkerState,
)
from edge_agent_workflow_scheduling.llm import MockLLMRuntime
from edge_agent_workflow_scheduling.profiler import (
    JsonlTraceLogger,
    build_llm_trace_record,
    build_tool_trace_record,
)
from edge_agent_workflow_scheduling.queue import InMemoryWorkflowQueue
from edge_agent_workflow_scheduling.scheduler import BaselineScheduler
from edge_agent_workflow_scheduling.scheduler.policies import (
    DEFAULT_SCHEDULER_POLICY_REGISTRY,
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
    average_total_latency_sec: float
    success_rate: float
    worker_counts: dict[str, int]
    llm_counts: dict[str, int]
    trace_path: Path


def main() -> None:
    args = parse_args()
    summary = run_demo(
        policy=args.policy,
        trace_path=args.trace_path,
        data_dir=args.data_dir,
        steps_per_agent=args.steps_per_agent,
        overwrite_trace=not args.append_trace,
    )
    print_summary(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        default="round_robin",
        choices=DEFAULT_SCHEDULER_POLICY_REGISTRY.available_policies(),
        help="Scheduler policy to use.",
    )
    parser.add_argument(
        "--trace-path",
        type=Path,
        default=Path("data/traces/first_demo.jsonl"),
        help="Path to the JSONL trace file.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/inputs/first_demo"),
        help="Directory used for generated demo inputs and outputs.",
    )
    parser.add_argument(
        "--steps-per-agent",
        type=int,
        default=10,
        help="Number of paired LLMCall and ToolCall steps generated per simulated agent.",
    )
    parser.add_argument(
        "--append-trace",
        action="store_true",
        help="Append to an existing trace instead of replacing it.",
    )
    return parser.parse_args()


def run_demo(
    *,
    policy: str = "round_robin",
    trace_path: Path = Path("data/traces/first_demo.jsonl"),
    data_dir: Path = Path("data/inputs/first_demo"),
    steps_per_agent: int = 10,
    overwrite_trace: bool = True,
) -> DemoSummary:
    if steps_per_agent < 1:
        msg = "steps_per_agent must be at least 1"
        raise ValueError(msg)

    input_dir = (data_dir / "images").resolve()
    output_dir = (data_dir / "outputs").resolve()
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    create_demo_images(input_dir=input_dir, steps_per_agent=steps_per_agent)

    agents = create_agents(input_dir=input_dir)
    queue = InMemoryWorkflowQueue(ordering="fifo")
    for agent_index, agent in enumerate(agents):
        queue.push_many(
            agent.generate_workflow_steps(
                steps_per_agent,
                start_sequence_id=agent_index * steps_per_agent,
            )
        )

    runtimes = create_llm_runtimes()
    workers = create_workers(output_dir=output_dir)
    scheduler = BaselineScheduler(policy_name=policy)
    logger = JsonlTraceLogger(trace_path)
    if overwrite_trace:
        logger.clear()

    records: list[TraceRecord] = []
    worker_counts: Counter[str] = Counter()
    llm_counts: Counter[str] = Counter()

    while not queue.is_empty():
        step = queue.pop()
        if step is None:
            break

        if isinstance(step, LLMCall):
            decision = scheduler.schedule(
                step,
                llm_states=build_llm_states(runtimes),
                worker_states=build_worker_states(workers),
            )
            runtime = runtimes[decision.selected_target]
            result = runtime.generate(step)
            record = build_llm_trace_record(
                llm_call=step,
                decision=decision,
                result=result,
            )
            llm_counts[decision.selected_target] += 1
        elif isinstance(step, ToolCall):
            decision = scheduler.schedule(
                step,
                llm_states=build_llm_states(runtimes),
                worker_states=build_worker_states(workers),
            )
            worker = workers[decision.selected_target]
            result = worker.run_tool(step)
            record = build_tool_trace_record(
                tool_call=step,
                decision=decision,
                result=result,
            )
            worker_counts[decision.selected_target] += 1
        else:
            msg = f"unsupported workflow step type: {type(step)!r}"
            raise TypeError(msg)

        records.append(record)
        logger.write(record)

    return build_summary(
        records=records,
        worker_counts=worker_counts,
        llm_counts=llm_counts,
        trace_path=trace_path,
    )


def create_agents(input_dir: Path) -> list[SimulatedAgent]:
    return [
        SimulatedAgent(
            agent_id="agent_image_7b",
            template=ToolCallTemplate(
                tool_type="image_preprocess",
                input_uri_prefix=input_dir.as_uri(),
                input_size_mb=0.01,
                image_count=1,
                deadline_sec=30,
                priority=1,
                file_extension="ppm",
            ),
            llm_template=LLMCallTemplate(
                model_name="qwen-7b",
                prompt_uri_prefix="local://prompts/agent_image_7b",
                input_tokens=512,
                estimated_output_tokens=128,
                context_length=2048,
                deadline_sec=15,
                priority=1,
            ),
        ),
        SimulatedAgent(
            agent_id="agent_image_27b",
            template=ToolCallTemplate(
                tool_type="image_preprocess",
                input_uri_prefix=input_dir.as_uri(),
                input_size_mb=0.02,
                image_count=1,
                deadline_sec=30,
                priority=2,
                file_extension="ppm",
            ),
            llm_template=LLMCallTemplate(
                model_name="qwen-27b",
                prompt_uri_prefix="local://prompts/agent_image_27b",
                input_tokens=1024,
                estimated_output_tokens=256,
                context_length=4096,
                deadline_sec=20,
                priority=2,
            ),
        ),
    ]


def create_llm_runtimes() -> dict[str, MockLLMRuntime]:
    runtimes = [
        MockLLMRuntime(
            llm_id="llm_edge_qwen_7b_mock",
            model_name="qwen-7b",
            device_id="edge_board_1",
            tokens_per_sec=160,
            queue_wait_time_sec=0.02,
            fixed_inference_overhead_sec=0.05,
        ),
        MockLLMRuntime(
            llm_id="llm_ubuntu_qwen_27b_mock",
            model_name="qwen-27b",
            device_id="ubuntu_server",
            tokens_per_sec=80,
            queue_wait_time_sec=0.04,
            fixed_inference_overhead_sec=0.08,
        ),
    ]
    return {runtime.llm_id: runtime for runtime in runtimes}


def create_workers(output_dir: Path) -> dict[str, LocalWorker]:
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
    workers = [
        LocalWorker(
            worker_id="worker_local_1",
            supported_tools=registry.supported_tools(),
            tool_executors=registry.as_executor_mapping(),
        ),
        LocalWorker(
            worker_id="worker_local_2",
            supported_tools=registry.supported_tools(),
            artificial_delay_sec=0.001,
            tool_executors=registry.as_executor_mapping(),
        ),
    ]
    return {worker.worker_id: worker for worker in workers}


def build_llm_states(runtimes: dict[str, MockLLMRuntime]) -> list[LLMInstanceState]:
    return [
        runtime.get_state(
            queue_len=0,
            avg_latency_sec=runtime.fixed_inference_overhead_sec,
        )
        for runtime in runtimes.values()
    ]


def build_worker_states(workers: dict[str, LocalWorker]) -> list[WorkerState]:
    return [
        worker.get_state(
            queue_len=0,
            cpu_util=0.1 if worker.worker_id == "worker_local_1" else 0.2,
            memory_util=0.1,
            network_latency_ms=0.0,
        )
        for worker in workers.values()
    ]


def create_demo_images(*, input_dir: Path, steps_per_agent: int) -> None:
    for agent_id, sequence_offset, base_size in (
        ("agent_image_7b", 0, 16),
        ("agent_image_27b", steps_per_agent, 24),
    ):
        for sequence_id in range(sequence_offset, sequence_offset + steps_per_agent):
            size = base_size + (sequence_id % 4) * 4
            image_path = input_dir / f"{agent_id}_{sequence_id:04d}.ppm"
            write_test_ppm(image_path, width=size, height=size)


def write_test_ppm(path: Path, *, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    pixels = bytearray()
    for y_pos in range(height):
        for x_pos in range(width):
            pixels.extend(
                (
                    (x_pos * 17) % 256,
                    (y_pos * 23) % 256,
                    ((x_pos + y_pos) * 13) % 256,
                )
            )
    path.write_bytes(header + bytes(pixels))


def build_summary(
    *,
    records: list[TraceRecord],
    worker_counts: Counter[str],
    llm_counts: Counter[str],
    trace_path: Path,
) -> DemoSummary:
    if not records:
        return DemoSummary(
            total_records=0,
            average_total_latency_sec=0.0,
            success_rate=0.0,
            worker_counts=dict(worker_counts),
            llm_counts=dict(llm_counts),
            trace_path=trace_path,
        )

    average_total_latency_sec = sum(record.total_latency_sec for record in records) / len(records)
    success_rate = sum(1 for record in records if record.success) / len(records)
    return DemoSummary(
        total_records=len(records),
        average_total_latency_sec=average_total_latency_sec,
        success_rate=success_rate,
        worker_counts=dict(sorted(worker_counts.items())),
        llm_counts=dict(sorted(llm_counts.items())),
        trace_path=trace_path,
    )


def print_summary(summary: DemoSummary) -> None:
    print(f"trace_path: {summary.trace_path}")
    print(f"total_records: {summary.total_records}")
    print(f"average_total_latency_sec: {summary.average_total_latency_sec:.6f}")
    print(f"success_rate: {summary.success_rate:.2%}")
    print("worker_counts:")
    for worker_id, count in summary.worker_counts.items():
        print(f"  {worker_id}: {count}")
    print("llm_counts:")
    for llm_id, count in summary.llm_counts.items():
        print(f"  {llm_id}: {count}")


if __name__ == "__main__":
    main()
