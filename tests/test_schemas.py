import json

import pytest

from edge_tool_offloading.common import (
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


def test_schedule_decision_round_trip_json() -> None:
    decision = ScheduleDecision(
        tool_call_id="tc_001",
        selected_worker="worker_local_1",
        policy_name="round_robin",
        reason="next worker in rotation",
    )

    restored = ScheduleDecision.from_json(decision.to_json())

    assert restored == decision


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
        tool_call_id="tc_001",
        agent_id="agent_1",
        tool_type="image_preprocess",
        selected_worker="worker_local_1",
        policy_name="round_robin",
        queue_wait_time_sec=0.1,
        execution_time_sec=2.5,
        total_latency_sec=2.7,
        success=True,
        timeout=False,
        reward=-2.7,
    )

    raw_json = trace.to_json()
    decoded = json.loads(raw_json)
    restored = TraceRecord.from_json(raw_json)

    assert decoded["tool_call_id"] == "tc_001"
    assert restored == trace
    assert restored.input_transfer_time_sec == 0.0
    assert restored.output_transfer_time_sec == 0.0
