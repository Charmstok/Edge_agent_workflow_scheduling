import random

import pytest

from edge_agent_workflow_scheduling.common import (
    LLMCall,
    LLMInstanceState,
    ToolCall,
    WorkerState,
)
from edge_agent_workflow_scheduling.scheduler import (
    BaselineScheduler,
    PolicySelection,
    RandomSchedulerPolicy,
    SchedulerPolicyRegistry,
    SchedulingCandidate,
    WorkflowStep,
)


def test_random_scheduler_selects_only_workers_that_support_tool() -> None:
    scheduler = BaselineScheduler(policy_name="random")
    tool_call = ToolCall(
        tool_call_id="tc_001",
        agent_id="agent_1",
        tool_type="image_preprocess",
        input_uri="local://inputs/image.png",
        input_size_mb=2.0,
    )
    worker_states = [
        WorkerState(
            worker_id="worker_ocr",
            supported_tools=["ocr"],
        ),
        WorkerState(
            worker_id="worker_image",
            supported_tools=["image_preprocess"],
        ),
    ]

    decision = scheduler.schedule(
        tool_call,
        llm_states=[],
        worker_states=worker_states,
    )

    assert decision.task_id == "tc_001"
    assert decision.task_kind == "tool"
    assert decision.selected_target == "worker_image"
    assert decision.policy_name == "random"


def test_random_scheduler_selects_only_online_matching_llm_runtime() -> None:
    scheduler = BaselineScheduler(policy_name="random")
    llm_call = LLMCall(
        llm_call_id="lc_001",
        agent_id="agent_1",
        model_name="qwen-7b",
        input_tokens=100,
        estimated_output_tokens=25,
    )
    llm_states = [
        LLMInstanceState(
            llm_id="llm_offline",
            model_name="qwen-7b",
            device_id="edge_board_1",
            is_online=False,
        ),
        LLMInstanceState(
            llm_id="llm_wrong_model",
            model_name="qwen-27b",
            device_id="ubuntu_server",
            is_online=True,
        ),
        LLMInstanceState(
            llm_id="llm_online",
            model_name="qwen-7b",
            device_id="edge_board_2",
            is_online=True,
        ),
    ]

    decision = scheduler.schedule(
        llm_call,
        llm_states=llm_states,
        worker_states=[],
    )

    assert decision.task_id == "lc_001"
    assert decision.task_kind == "llm"
    assert decision.selected_target == "llm_online"
    assert decision.policy_name == "random"


def test_random_scheduler_can_be_seeded_through_policy_registry() -> None:
    registry = SchedulerPolicyRegistry()
    registry.register_factory(
        "random",
        lambda: RandomSchedulerPolicy(rng=random.Random(3)),
        replace=True,
    )
    scheduler = BaselineScheduler(policy_name="random", policy_registry=registry)
    tool_call = ToolCall(
        tool_call_id="tc_001",
        agent_id="agent_1",
        tool_type="ocr",
        input_uri="local://inputs/page.png",
        input_size_mb=4.0,
    )
    worker_states = [
        WorkerState(worker_id="worker_a", supported_tools=["ocr"]),
        WorkerState(worker_id="worker_b", supported_tools=["ocr"]),
        WorkerState(worker_id="worker_c", supported_tools=["ocr"]),
    ]

    decision = scheduler.schedule(
        tool_call,
        llm_states=[],
        worker_states=worker_states,
    )

    assert decision.selected_target == "worker_a"


def test_round_robin_scheduler_rotates_between_available_workers() -> None:
    scheduler = BaselineScheduler(policy_name="round_robin")
    tool_call = ToolCall(
        tool_call_id="tc_001",
        agent_id="agent_1",
        tool_type="ocr",
        input_uri="local://inputs/page.png",
        input_size_mb=4.0,
    )
    worker_states = [
        WorkerState(worker_id="worker_b", supported_tools=["ocr"]),
        WorkerState(worker_id="worker_a", supported_tools=["ocr"]),
        WorkerState(worker_id="worker_unsupported", supported_tools=["pdf_parse"]),
        WorkerState(worker_id="worker_offline", supported_tools=["ocr"], is_online=False),
    ]

    decisions = [
        scheduler.schedule(tool_call, llm_states=[], worker_states=worker_states)
        for _ in range(3)
    ]

    assert [decision.selected_target for decision in decisions] == [
        "worker_a",
        "worker_b",
        "worker_a",
    ]
    assert all(decision.policy_name == "round_robin" for decision in decisions)


def test_round_robin_scheduler_rotates_between_available_llm_runtimes() -> None:
    scheduler = BaselineScheduler(policy_name="round_robin")
    llm_call = LLMCall(
        llm_call_id="lc_001",
        agent_id="agent_1",
        model_name="qwen-7b",
        input_tokens=100,
        estimated_output_tokens=25,
    )
    llm_states = [
        LLMInstanceState(
            llm_id="llm_b",
            model_name="qwen-7b",
            device_id="edge_board_b",
        ),
        LLMInstanceState(
            llm_id="llm_a",
            model_name="qwen-7b",
            device_id="edge_board_a",
        ),
        LLMInstanceState(
            llm_id="llm_wrong_model",
            model_name="qwen-27b",
            device_id="ubuntu_server",
        ),
        LLMInstanceState(
            llm_id="llm_offline",
            model_name="qwen-7b",
            device_id="edge_board_offline",
            is_online=False,
        ),
    ]

    decisions = [
        scheduler.schedule(llm_call, llm_states=llm_states, worker_states=[])
        for _ in range(3)
    ]

    assert [decision.selected_target for decision in decisions] == [
        "llm_a",
        "llm_b",
        "llm_a",
    ]


def test_least_queue_scheduler_selects_smallest_queue() -> None:
    scheduler = BaselineScheduler(policy_name="least_queue")
    tool_call = ToolCall(
        tool_call_id="tc_001",
        agent_id="agent_1",
        tool_type="pdf_parse",
        input_uri="local://inputs/doc.pdf",
        input_size_mb=16.0,
    )
    worker_states = [
        WorkerState(worker_id="worker_busy", supported_tools=["pdf_parse"], queue_len=5),
        WorkerState(worker_id="worker_idle", supported_tools=["pdf_parse"], queue_len=1),
    ]

    decision = scheduler.schedule(
        tool_call,
        llm_states=[],
        worker_states=worker_states,
    )

    assert decision.selected_target == "worker_idle"
    assert decision.score == 1.0


def test_earliest_finish_time_scheduler_selects_fastest_llm_estimate() -> None:
    scheduler = BaselineScheduler(policy_name="earliest_finish_time")
    llm_call = LLMCall(
        llm_call_id="lc_001",
        agent_id="agent_1",
        model_name="qwen-7b",
        input_tokens=80,
        estimated_output_tokens=20,
    )
    llm_states = [
        LLMInstanceState(
            llm_id="llm_slow",
            model_name="qwen-7b",
            device_id="edge_board_slow",
            tokens_per_sec=10,
        ),
        LLMInstanceState(
            llm_id="llm_fast",
            model_name="qwen-7b",
            device_id="edge_board_fast",
            tokens_per_sec=100,
        ),
    ]

    decision = scheduler.schedule(
        llm_call,
        llm_states=llm_states,
        worker_states=[],
    )

    assert decision.selected_target == "llm_fast"
    assert decision.score == 1.0


def test_scheduler_raises_when_no_available_targets_exist() -> None:
    scheduler = BaselineScheduler(policy_name="random")
    tool_call = ToolCall(
        tool_call_id="tc_001",
        agent_id="agent_1",
        tool_type="table_extract",
        input_uri="local://inputs/table.pdf",
        input_size_mb=8.0,
    )

    with pytest.raises(ValueError, match="no available execution targets"):
        scheduler.schedule(
            tool_call,
            llm_states=[],
            worker_states=[
                WorkerState(worker_id="worker_ocr", supported_tools=["ocr"]),
            ],
        )


def test_scheduler_policy_registry_can_register_custom_policy() -> None:
    class LastCandidatePolicy:
        name = "last"

        def select(
            self,
            step: WorkflowStep,
            candidates: list[SchedulingCandidate],
        ) -> PolicySelection:
            return PolicySelection(
                candidate=candidates[-1],
                reason="selected last candidate",
            )

    registry = SchedulerPolicyRegistry()
    registry.register_factory("last", LastCandidatePolicy)
    scheduler = BaselineScheduler(policy_name="last", policy_registry=registry)
    tool_call = ToolCall(
        tool_call_id="tc_001",
        agent_id="agent_1",
        tool_type="ocr",
        input_uri="local://inputs/page.png",
        input_size_mb=4.0,
    )

    decision = scheduler.schedule(
        tool_call,
        llm_states=[],
        worker_states=[
            WorkerState(worker_id="worker_a", supported_tools=["ocr"]),
            WorkerState(worker_id="worker_b", supported_tools=["ocr"]),
        ],
    )

    assert "last" in registry.available_policies()
    assert decision.selected_target == "worker_b"
    assert decision.policy_name == "last"
    assert decision.reason == "selected last candidate"


def test_scheduler_policy_registry_rejects_unknown_policy() -> None:
    registry = SchedulerPolicyRegistry()

    with pytest.raises(ValueError, match="scheduler policy must be one of"):
        registry.create("unknown")
