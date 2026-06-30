import pytest

from edge_agent_workflow_scheduling.agents import LLMCallTemplate, SimulatedAgent, ToolCallTemplate
from edge_agent_workflow_scheduling.common import LLMCall, ToolCall


def test_simulated_agent_generates_one_tool_call() -> None:
    agent = SimulatedAgent(
        agent_id="agent_1",
        template=ToolCallTemplate(
            tool_type="image_preprocess",
            input_uri_prefix="local://inputs/images",
            input_size_mb=4.5,
            image_count=1,
            deadline_sec=30,
            priority=2,
            file_extension="png",
        ),
    )

    tool_call = agent.generate_tool_call(0)

    assert tool_call.tool_call_id == "agent_1-tc-0000"
    assert tool_call.agent_id == "agent_1"
    assert tool_call.tool_type == "image_preprocess"
    assert tool_call.input_uri == "local://inputs/images/agent_1_0000.png"
    assert tool_call.input_size_mb == 4.5
    assert tool_call.page_count == 0
    assert tool_call.image_count == 1
    assert tool_call.deadline_sec == 30
    assert tool_call.priority == 2


def test_simulated_agent_generates_one_llm_call() -> None:
    agent = SimulatedAgent(
        agent_id="agent_1",
        template=ToolCallTemplate(
            tool_type="image_preprocess",
            input_uri_prefix="local://inputs/images",
            input_size_mb=4.5,
        ),
        llm_template=LLMCallTemplate(
            model_name="qwen-7b",
            prompt_uri_prefix="local://prompts/tasks",
            input_tokens=1024,
            estimated_output_tokens=256,
            context_length=4096,
            deadline_sec=20,
            priority=3,
        ),
    )

    llm_call = agent.generate_llm_call(0)

    assert llm_call.llm_call_id == "agent_1-lc-0000"
    assert llm_call.agent_id == "agent_1"
    assert llm_call.prompt_uri == "local://prompts/tasks/agent_1_0000.txt"
    assert llm_call.input_tokens == 1024
    assert llm_call.estimated_output_tokens == 256
    assert llm_call.context_length == 4096
    assert llm_call.model_name == "qwen-7b"
    assert llm_call.deadline_sec == 20
    assert llm_call.priority == 3


def test_simulated_agent_generates_deterministic_batch() -> None:
    agent = SimulatedAgent(
        agent_id="agent_pdf",
        template=ToolCallTemplate(
            tool_type="pdf_parse",
            input_uri_prefix="local://inputs/pdfs",
            input_size_mb=20,
            page_count=50,
            deadline_sec=60,
            file_extension="pdf",
        ),
    )

    tool_calls = agent.generate_tool_calls(3, start_sequence_id=7)

    assert [tool_call.tool_call_id for tool_call in tool_calls] == [
        "agent_pdf-tc-0007",
        "agent_pdf-tc-0008",
        "agent_pdf-tc-0009",
    ]
    assert [tool_call.input_uri for tool_call in tool_calls] == [
        "local://inputs/pdfs/agent_pdf_0007.pdf",
        "local://inputs/pdfs/agent_pdf_0008.pdf",
        "local://inputs/pdfs/agent_pdf_0009.pdf",
    ]
    assert all(tool_call.tool_type == "pdf_parse" for tool_call in tool_calls)
    assert all(tool_call.page_count == 50 for tool_call in tool_calls)


def test_simulated_agent_generates_deterministic_llm_batch() -> None:
    agent = SimulatedAgent(
        agent_id="agent_llm",
        template=ToolCallTemplate(
            tool_type="ocr",
            input_uri_prefix="local://inputs/images",
            input_size_mb=8,
            image_count=5,
        ),
        llm_template=LLMCallTemplate(
            model_name="qwen-27b",
            prompt_uri_prefix="local://prompts/ocr",
            input_tokens=2048,
            estimated_output_tokens=512,
            context_length=8192,
            file_extension="md",
        ),
    )

    llm_calls = agent.generate_llm_calls(3, start_sequence_id=4)

    assert [llm_call.llm_call_id for llm_call in llm_calls] == [
        "agent_llm-lc-0004",
        "agent_llm-lc-0005",
        "agent_llm-lc-0006",
    ]
    assert [llm_call.prompt_uri for llm_call in llm_calls] == [
        "local://prompts/ocr/agent_llm_0004.md",
        "local://prompts/ocr/agent_llm_0005.md",
        "local://prompts/ocr/agent_llm_0006.md",
    ]
    assert all(llm_call.model_name == "qwen-27b" for llm_call in llm_calls)
    assert all(llm_call.input_tokens == 2048 for llm_call in llm_calls)


def test_simulated_agent_generates_paired_workflow_steps() -> None:
    agent = SimulatedAgent(
        agent_id="agent_flow",
        template=ToolCallTemplate(
            tool_type="pdf_parse",
            input_uri_prefix="local://inputs/pdfs",
            input_size_mb=32,
            page_count=80,
            file_extension="pdf",
        ),
        llm_template=LLMCallTemplate(
            model_name="qwen-7b",
            prompt_uri_prefix="local://prompts/pdf",
            input_tokens=1536,
            estimated_output_tokens=256,
        ),
    )

    steps = agent.generate_workflow_steps(2, start_sequence_id=9)

    assert [type(step) for step in steps] == [LLMCall, ToolCall, LLMCall, ToolCall]
    assert [step.agent_id for step in steps] == ["agent_flow"] * 4
    assert steps[0].llm_call_id == "agent_flow-lc-0009"
    assert steps[1].tool_call_id == "agent_flow-tc-0009"
    assert steps[2].llm_call_id == "agent_flow-lc-0010"
    assert steps[3].tool_call_id == "agent_flow-tc-0010"


def test_simulated_agent_rejects_negative_sequence_id() -> None:
    agent = SimulatedAgent(
        agent_id="agent_1",
        template=ToolCallTemplate(
            tool_type="image_preprocess",
            input_uri_prefix="local://inputs/images",
            input_size_mb=1,
        ),
    )

    with pytest.raises(ValueError, match="sequence_id must be non-negative"):
        agent.generate_tool_call(-1)

    with pytest.raises(ValueError, match="sequence_id must be non-negative"):
        agent.generate_llm_call(-1)


def test_simulated_agent_rejects_negative_count() -> None:
    agent = SimulatedAgent(
        agent_id="agent_1",
        template=ToolCallTemplate(
            tool_type="image_preprocess",
            input_uri_prefix="local://inputs/images",
            input_size_mb=1,
        ),
    )

    with pytest.raises(ValueError, match="count must be non-negative"):
        agent.generate_tool_calls(-1)

    with pytest.raises(ValueError, match="count must be non-negative"):
        agent.generate_llm_calls(-1)

    with pytest.raises(ValueError, match="count must be non-negative"):
        agent.generate_workflow_steps(-1)


def test_simulated_agent_rejects_negative_start_sequence_id() -> None:
    agent = SimulatedAgent(
        agent_id="agent_1",
        template=ToolCallTemplate(
            tool_type="image_preprocess",
            input_uri_prefix="local://inputs/images",
            input_size_mb=1,
        ),
    )

    with pytest.raises(ValueError, match="start_sequence_id must be non-negative"):
        agent.generate_tool_calls(1, start_sequence_id=-1)

    with pytest.raises(ValueError, match="start_sequence_id must be non-negative"):
        agent.generate_llm_calls(1, start_sequence_id=-1)

    with pytest.raises(ValueError, match="start_sequence_id must be non-negative"):
        agent.generate_workflow_steps(1, start_sequence_id=-1)
