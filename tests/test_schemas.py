import json

import pytest

from edge_tool_offloading.common import (
    LLMCall,
    LLMInstanceInfo,
    LLMInstanceState,
    LLMResult,
    ScheduleDecision,
    ToolCall,
    ToolResult,
    TraceRecord,
    WorkerInfo,
    WorkerState,
)


def test_tool_call_round_trip_json() -> None:
    tool_call = ToolCall(
        tool_call_id="tc_001",
        agent_id="agent_1",
        tool_type="image_preprocess",
        input_uri="local://data/input.png",
        input_size_mb=8.5,
        image_count=1,
    )

    restored = ToolCall.from_json(tool_call.to_json())

    assert restored == tool_call
    assert restored.page_count == 0
    assert restored.priority == 0
    assert restored.deadline_sec is None


def test_llm_call_round_trip_json() -> None:
    llm_call = LLMCall(
        llm_call_id="lc_001",
        agent_id="agent_1",
        input_tokens=2048,
        estimated_output_tokens=512,
        context_length=4096,
        model_name="qwen-30b",
        deadline_sec=20,
    )

    restored = LLMCall.from_json(llm_call.to_json())

    assert restored == llm_call
    assert restored.priority == 0


def test_tool_call_requires_required_fields() -> None:
    with pytest.raises(TypeError):
        ToolCall.from_dict({"tool_call_id": "tc_001"})


def test_worker_info_round_trip_dict() -> None:
    worker_info = WorkerInfo(
        worker_id="worker_local_1",
        supported_tools=["image_preprocess"],
        metadata={"device_type": "local"},
    )

    restored = WorkerInfo.from_dict(worker_info.to_dict())

    assert restored == worker_info
    assert restored.max_concurrency == 1


def test_llm_instance_info_round_trip_dict() -> None:
    llm_info = LLMInstanceInfo(
        llm_id="qwen_30b_1",
        model_name="qwen-30b",
        endpoint_url="http://127.0.0.1:8001",
        device_id="ubuntu_server",
        model_size_b=30,
        accelerator="gpu",
    )

    restored = LLMInstanceInfo.from_dict(llm_info.to_dict())

    assert restored == llm_info
    assert restored.max_concurrency == 1


def test_worker_state_defaults() -> None:
    worker_state = WorkerState(
        worker_id="worker_local_1",
        supported_tools=["image_preprocess"],
    )

    assert worker_state.queue_len == 0
    assert worker_state.cpu_util == 0.0
    assert worker_state.memory_util == 0.0
    assert worker_state.network_latency_ms == 0.0
    assert worker_state.is_online is True


def test_llm_instance_state_defaults() -> None:
    llm_state = LLMInstanceState(
        llm_id="qwen_30b_1",
        model_name="qwen-30b",
        device_id="ubuntu_server",
    )

    assert llm_state.queue_len == 0
    assert llm_state.running_requests == 0
    assert llm_state.gpu_util == 0.0
    assert llm_state.memory_util == 0.0
    assert llm_state.tokens_per_sec == 0.0
    assert llm_state.avg_latency_sec == 0.0
    assert llm_state.is_online is True


def test_schedule_decision_round_trip_json() -> None:
    decision = ScheduleDecision(
        task_id="tc_001",
        task_kind="tool",
        selected_target="worker_local_1",
        policy_name="round_robin",
        reason="next worker in rotation",
    )

    restored = ScheduleDecision.from_json(decision.to_json())

    assert restored == decision


def test_llm_result_round_trip_json() -> None:
    result = LLMResult(
        llm_call_id="lc_001",
        llm_id="qwen_30b_1",
        success=True,
        output_uri="local://outputs/llm_001.txt",
        output_tokens=256,
        queue_wait_time_sec=0.5,
        inference_time_sec=8.0,
    )

    restored = LLMResult.from_json(result.to_json())

    assert restored == result
    assert restored.error_message is None


def test_tool_result_round_trip_json() -> None:
    result = ToolResult(
        tool_call_id="tc_001",
        worker_id="worker_local_1",
        success=True,
        output_uri="local://data/output.png",
        execution_time_sec=2.5,
    )

    restored = ToolResult.from_json(result.to_json())

    assert restored == result
    assert restored.error_message is None


def test_trace_record_is_json_serializable() -> None:
    trace = TraceRecord(
        task_id="tc_001",
        task_kind="tool",
        agent_id="agent_1",
        selected_target="worker_local_1",
        policy_name="round_robin",
        queue_wait_time_sec=0.1,
        execution_time_sec=2.5,
        total_latency_sec=2.7,
        success=True,
        timeout=False,
        reward=-2.7,
        tool_type="image_preprocess",
    )

    raw_json = trace.to_json()
    decoded = json.loads(raw_json)
    restored = TraceRecord.from_json(raw_json)

    assert decoded["task_id"] == "tc_001"
    assert restored == trace
    assert restored.input_transfer_time_sec == 0.0
    assert restored.output_transfer_time_sec == 0.0


def test_trace_record_supports_llm_step() -> None:
    trace = TraceRecord(
        task_id="lc_001",
        task_kind="llm",
        agent_id="agent_1",
        selected_target="qwen_30b_1",
        policy_name="least_queue",
        queue_wait_time_sec=0.4,
        execution_time_sec=8.0,
        total_latency_sec=8.4,
        success=True,
        timeout=False,
        reward=-8.4,
        model_name="qwen-30b",
        input_tokens=2048,
        output_tokens=512,
    )

    restored = TraceRecord.from_json(trace.to_json())

    assert restored == trace
    assert restored.task_kind == "llm"
    assert restored.model_name == "qwen-30b"
