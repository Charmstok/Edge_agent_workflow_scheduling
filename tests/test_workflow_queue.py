from collections.abc import Sequence

import pytest

from edge_agent_workflow_scheduling.agents import LLMCallTemplate, SimulatedAgent, ToolCallTemplate
from edge_agent_workflow_scheduling.common import LLMCall, ToolCall
from edge_agent_workflow_scheduling.queue import InMemoryWorkflowQueue
from edge_agent_workflow_scheduling.queue.policy import QueuePolicyFactory


def test_fifo_queue_handles_mixed_steps_from_multiple_agents() -> None:
    image_agent = SimulatedAgent(
        agent_id="agent_image",
        template=ToolCallTemplate(
            tool_type="image_preprocess",
            input_uri_prefix="local://inputs/images",
            input_size_mb=3.0,
            image_count=1,
            deadline_sec=15,
            file_extension="png",
        ),
        llm_template=LLMCallTemplate(
            model_name="qwen-7b",
            prompt_uri_prefix="local://prompts/image",
            input_tokens=512,
            estimated_output_tokens=128,
            deadline_sec=10,
        ),
    )
    pdf_agent = SimulatedAgent(
        agent_id="agent_pdf",
        template=ToolCallTemplate(
            tool_type="pdf_parse",
            input_uri_prefix="local://inputs/pdfs",
            input_size_mb=24.0,
            page_count=80,
            deadline_sec=60,
            file_extension="pdf",
        ),
        llm_template=LLMCallTemplate(
            model_name="qwen-27b",
            prompt_uri_prefix="local://prompts/pdf",
            input_tokens=2048,
            estimated_output_tokens=512,
            deadline_sec=25,
        ),
    )
    steps = [
        *image_agent.generate_workflow_steps(1),
        *pdf_agent.generate_workflow_steps(1),
    ]
    queue = InMemoryWorkflowQueue(ordering="fifo")

    queue.push_many(steps)

    assert queue.size() == 4
    assert queue.peek() == steps[0]
    assert queue.size() == 4
    assert [queue.pop(), queue.pop(), queue.pop(), queue.pop()] == steps
    assert queue.pop() is None
    assert queue.peek() is None
    assert queue.is_empty() is True


def test_priority_queue_returns_highest_priority_first_with_fifo_ties() -> None:
    low_priority = ToolCall(
        tool_call_id="tc_low",
        agent_id="agent_1",
        tool_type="image_preprocess",
        input_uri="local://inputs/low.png",
        input_size_mb=2.0,
        priority=1,
    )
    high_priority_first = LLMCall(
        llm_call_id="lc_high_first",
        agent_id="agent_2",
        model_name="qwen-27b",
        prompt_uri="local://prompts/high_first.txt",
        priority=5,
    )
    medium_priority = LLMCall(
        llm_call_id="lc_medium",
        agent_id="agent_3",
        model_name="qwen-7b",
        prompt_uri="local://prompts/medium.txt",
        priority=3,
    )
    high_priority_second = ToolCall(
        tool_call_id="tc_high_second",
        agent_id="agent_4",
        tool_type="ocr",
        input_uri="local://inputs/high.png",
        input_size_mb=8.0,
        priority=5,
    )
    queue = InMemoryWorkflowQueue(
        [low_priority, high_priority_first, medium_priority, high_priority_second],
        ordering="priority",
    )

    assert queue.peek() == high_priority_first
    assert queue.size() == 4
    assert [queue.pop(), queue.pop(), queue.pop(), queue.pop()] == [
        high_priority_first,
        high_priority_second,
        medium_priority,
        low_priority,
    ]


def test_queue_allows_per_call_ordering_override() -> None:
    first = ToolCall(
        tool_call_id="tc_first",
        agent_id="agent_1",
        tool_type="image_preprocess",
        input_uri="local://inputs/first.png",
        input_size_mb=1.0,
        priority=0,
    )
    second = ToolCall(
        tool_call_id="tc_second",
        agent_id="agent_1",
        tool_type="image_preprocess",
        input_uri="local://inputs/second.png",
        input_size_mb=1.0,
        priority=10,
    )
    queue = InMemoryWorkflowQueue([first, second], ordering="fifo")

    assert queue.peek(ordering="priority") == second
    assert queue.pop(ordering="priority") == second
    assert queue.pop() == first


def test_queue_rejects_invalid_ordering() -> None:
    with pytest.raises(ValueError, match="ordering must be one of"):
        InMemoryWorkflowQueue(ordering="unknown")  # type: ignore[arg-type]


def test_queue_policy_factory_can_register_custom_policy() -> None:
    class LastItemQueuePolicy:
        name = "last"

        def select_index(self, items: Sequence[object]) -> int:
            return len(items) - 1

    factory = QueuePolicyFactory()
    factory.register(LastItemQueuePolicy())

    assert factory.available_orderings() == ["fifo", "last", "priority"]
    assert factory.create("last").select_index([object(), object(), object()]) == 2


def test_queue_can_use_injected_policy_factory() -> None:
    class LastItemQueuePolicy:
        name = "last"

        def select_index(self, items: Sequence[object]) -> int:
            return len(items) - 1

    first = ToolCall(
        tool_call_id="tc_first",
        agent_id="agent_1",
        tool_type="image_preprocess",
        input_uri="local://inputs/first.png",
        input_size_mb=1.0,
    )
    second = ToolCall(
        tool_call_id="tc_second",
        agent_id="agent_1",
        tool_type="image_preprocess",
        input_uri="local://inputs/second.png",
        input_size_mb=1.0,
    )
    factory = QueuePolicyFactory()
    factory.register(LastItemQueuePolicy())
    queue = InMemoryWorkflowQueue([first, second], ordering="last", policy_factory=factory)

    assert queue.ordering == "last"
    assert queue.pop() == second
    assert queue.pop() == first


def test_queue_rejects_unsupported_step_type() -> None:
    queue = InMemoryWorkflowQueue()

    with pytest.raises(TypeError, match="step must be an LLMCall or ToolCall"):
        queue.push(object())  # type: ignore[arg-type]
