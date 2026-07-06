import json
from pathlib import Path

import pytest

from edge_agent_workflow_scheduling.common import (
    LLMCall,
    LLMResult,
    ScheduleDecision,
    ToolCall,
    ToolResult,
)
from edge_agent_workflow_scheduling.profiler import (
    JsonlTraceLogger,
    build_llm_trace_record,
    build_tool_trace_record,
    calculate_step_reward,
)


def test_build_llm_trace_record_contains_required_llm_fields() -> None:
    llm_call = LLMCall(
        llm_call_id="lc_001",
        agent_id="agent_1",
        model_name="qwen-7b",
        input_tokens=512,
        estimated_output_tokens=128,
    )
    decision = ScheduleDecision(
        task_id="lc_001",
        task_kind="llm",
        selected_target="llm_edge_qwen_7b_mock",
        policy_name="round_robin",
    )
    result = LLMResult(
        llm_call_id="lc_001",
        llm_id="llm_edge_qwen_7b_mock",
        success=True,
        output_tokens=96,
        queue_wait_time_sec=0.2,
        inference_time_sec=1.5,
    )

    trace = build_llm_trace_record(
        llm_call=llm_call,
        decision=decision,
        result=result,
        input_transfer_time_sec=0.1,
        output_transfer_time_sec=0.05,
    )

    assert trace.task_id == "lc_001"
    assert trace.task_kind == "llm"
    assert trace.agent_id == "agent_1"
    assert trace.model_name == "qwen-7b"
    assert trace.selected_target == "llm_edge_qwen_7b_mock"
    assert trace.policy_name == "round_robin"
    assert trace.queue_wait_time_sec == 0.2
    assert trace.execution_time_sec == 1.5
    assert trace.total_latency_sec == 1.85
    assert trace.reward == -1.85
    assert trace.input_tokens == 512
    assert trace.output_tokens == 96
    assert trace.success is True
    assert trace.timeout is False


def test_build_tool_trace_record_contains_required_tool_fields() -> None:
    tool_call = ToolCall(
        tool_call_id="tc_001",
        agent_id="agent_1",
        tool_type="image_preprocess",
        input_uri="local://inputs/image.ppm",
        input_size_mb=2.0,
        image_count=1,
    )
    decision = ScheduleDecision(
        task_id="tc_001",
        task_kind="tool",
        selected_target="worker_local_1",
        policy_name="least_queue",
    )
    result = ToolResult(
        tool_call_id="tc_001",
        worker_id="worker_local_1",
        success=True,
        output_uri="file:///tmp/output.pgm",
        execution_time_sec=2.5,
    )

    trace = build_tool_trace_record(
        tool_call=tool_call,
        decision=decision,
        result=result,
        queue_wait_time_sec=0.3,
        input_transfer_time_sec=0.1,
        output_transfer_time_sec=0.2,
    )

    assert trace.task_id == "tc_001"
    assert trace.task_kind == "tool"
    assert trace.agent_id == "agent_1"
    assert trace.tool_type == "image_preprocess"
    assert trace.selected_target == "worker_local_1"
    assert trace.policy_name == "least_queue"
    assert trace.queue_wait_time_sec == 0.3
    assert trace.execution_time_sec == 2.5
    assert trace.total_latency_sec == 3.1
    assert trace.reward == -3.1
    assert trace.success is True
    assert trace.timeout is False


def test_jsonl_trace_logger_writes_and_reads_records(tmp_path: Path) -> None:
    trace_path = tmp_path / "traces" / "first_demo.jsonl"
    logger = JsonlTraceLogger(trace_path)
    llm_trace = build_llm_trace_record(
        llm_call=LLMCall(
            llm_call_id="lc_001",
            agent_id="agent_1",
            model_name="qwen-7b",
            input_tokens=128,
            estimated_output_tokens=32,
        ),
        decision=ScheduleDecision(
            task_id="lc_001",
            task_kind="llm",
            selected_target="llm_edge_qwen_7b_mock",
            policy_name="random",
        ),
        result=LLMResult(
            llm_call_id="lc_001",
            llm_id="llm_edge_qwen_7b_mock",
            success=True,
            output_tokens=24,
            queue_wait_time_sec=0.1,
            inference_time_sec=0.4,
        ),
    )
    tool_trace = build_tool_trace_record(
        tool_call=ToolCall(
            tool_call_id="tc_001",
            agent_id="agent_1",
            tool_type="image_preprocess",
            input_uri="local://inputs/image.ppm",
            input_size_mb=1.0,
        ),
        decision=ScheduleDecision(
            task_id="tc_001",
            task_kind="tool",
            selected_target="worker_local_1",
            policy_name="random",
        ),
        result=ToolResult(
            tool_call_id="tc_001",
            worker_id="worker_local_1",
            success=True,
            execution_time_sec=0.8,
        ),
    )

    logger.write(llm_trace)
    logger.write_many([tool_trace])
    raw_records = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    restored_records = logger.read_all()

    assert trace_path.exists()
    assert logger.count() == 2
    assert len(raw_records) == 2
    assert len(restored_records) == 2
    assert restored_records == [llm_trace, tool_trace]
    assert raw_records[0]["task_kind"] == "llm"
    assert raw_records[0]["model_name"] == "qwen-7b"
    assert raw_records[0]["input_tokens"] == 128
    assert raw_records[0]["output_tokens"] == 24
    assert raw_records[1]["task_kind"] == "tool"
    assert raw_records[1]["tool_type"] == "image_preprocess"
    assert raw_records[1]["execution_time_sec"] == 0.8


def test_jsonl_trace_logger_clear_removes_trace_file(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    logger = JsonlTraceLogger(trace_path)
    trace_path.write_text("{}\n", encoding="utf-8")

    logger.clear()

    assert logger.read_all() == []
    assert logger.count() == 0
    assert not trace_path.exists()


def test_trace_builder_rejects_mismatched_decision_and_result_ids() -> None:
    with pytest.raises(ValueError, match="decision task_id"):
        build_tool_trace_record(
            tool_call=ToolCall(
                tool_call_id="tc_001",
                agent_id="agent_1",
                tool_type="image_preprocess",
                input_uri="local://inputs/image.ppm",
                input_size_mb=1.0,
            ),
            decision=ScheduleDecision(
                task_id="tc_other",
                task_kind="tool",
                selected_target="worker_local_1",
                policy_name="random",
            ),
            result=ToolResult(
                tool_call_id="tc_001",
                worker_id="worker_local_1",
                success=True,
            ),
        )

    with pytest.raises(ValueError, match="LLMResult id"):
        build_llm_trace_record(
            llm_call=LLMCall(
                llm_call_id="lc_001",
                agent_id="agent_1",
            ),
            decision=ScheduleDecision(
                task_id="lc_001",
                task_kind="llm",
                selected_target="llm_edge_qwen_7b_mock",
                policy_name="random",
            ),
            result=LLMResult(
                llm_call_id="lc_other",
                llm_id="llm_edge_qwen_7b_mock",
                success=True,
            ),
        )


def test_calculate_step_reward_applies_timeout_and_failure_penalties() -> None:
    assert calculate_step_reward(total_latency_sec=3.0, success=True, timeout=False) == -3.0
    assert calculate_step_reward(total_latency_sec=3.0, success=True, timeout=True) == -23.0
    assert calculate_step_reward(total_latency_sec=3.0, success=False, timeout=False) == -53.0
    assert calculate_step_reward(total_latency_sec=3.0, success=False, timeout=True) == -73.0
