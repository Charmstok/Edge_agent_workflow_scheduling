import pytest

from edge_agent_workflow_scheduling.common import LLMCall
from edge_agent_workflow_scheduling.llm import MockLLMRuntime


def test_mock_llm_runtime_generates_successful_result() -> None:
    runtime = MockLLMRuntime(
        llm_id="llm_ubuntu_qwen_27b_mock",
        model_name="qwen-27b",
        device_id="ubuntu_server",
        tokens_per_sec=100,
        queue_wait_time_sec=0.25,
        fixed_inference_overhead_sec=0.5,
    )
    llm_call = LLMCall(
        llm_call_id="lc_001",
        agent_id="agent_1",
        prompt_uri="local://prompts/task.txt",
        input_tokens=200,
        estimated_output_tokens=50,
        model_name="qwen-27b",
    )

    result = runtime.generate(llm_call)

    assert result.llm_call_id == "lc_001"
    assert result.llm_id == "llm_ubuntu_qwen_27b_mock"
    assert result.success is True
    assert result.output_uri == "local://outputs/llm/llm_ubuntu_qwen_27b_mock/lc_001.json"
    assert result.output_tokens == 50
    assert result.queue_wait_time_sec == 0.25
    assert result.inference_time_sec == 3.0
    assert result.error_message is None


def test_mock_llm_runtime_uses_default_output_tokens() -> None:
    runtime = MockLLMRuntime(
        llm_id="llm_edge_qwen_7b_mock",
        model_name="qwen-7b",
        device_id="edge_board_1",
        tokens_per_sec=50,
        default_output_tokens=32,
    )
    llm_call = LLMCall(
        llm_call_id="lc_002",
        agent_id="agent_1",
        input_tokens=68,
        estimated_output_tokens=0,
        model_name="qwen-7b",
    )

    result = runtime.generate(llm_call)

    assert result.success is True
    assert result.output_tokens == 32
    assert result.inference_time_sec == 2.0


def test_mock_llm_runtime_rejects_model_mismatch() -> None:
    runtime = MockLLMRuntime(
        llm_id="llm_edge_qwen_7b_mock",
        model_name="qwen-7b",
        device_id="edge_board_1",
        tokens_per_sec=50,
    )
    llm_call = LLMCall(
        llm_call_id="lc_003",
        agent_id="agent_1",
        input_tokens=128,
        estimated_output_tokens=32,
        model_name="qwen-27b",
    )

    result = runtime.generate(llm_call)

    assert result.success is False
    assert result.output_uri is None
    assert result.output_tokens == 0
    assert result.queue_wait_time_sec == 0.0
    assert result.inference_time_sec == 0.0
    assert "does not match runtime model" in result.error_message


def test_mock_llm_runtime_allows_unspecified_call_model() -> None:
    runtime = MockLLMRuntime(
        llm_id="llm_ubuntu_qwen_27b_mock",
        model_name="qwen-27b",
        device_id="ubuntu_server",
        tokens_per_sec=100,
    )
    llm_call = LLMCall(
        llm_call_id="lc_004",
        agent_id="agent_1",
        input_tokens=100,
        estimated_output_tokens=20,
        model_name=None,
    )

    result = runtime.generate(llm_call)

    assert result.success is True
    assert result.output_tokens == 20


def test_mock_llm_runtime_info_and_state_snapshots() -> None:
    runtime = MockLLMRuntime(
        llm_id="llm_ubuntu_qwen_27b_mock",
        model_name="qwen-27b",
        device_id="ubuntu_server",
        tokens_per_sec=80,
        endpoint_url="mock://custom",
        model_size_b=27,
        accelerator="gpu",
        max_concurrency=2,
        metadata={"placement": "local"},
    )

    info = runtime.to_info()
    state = runtime.get_state(queue_len=2, running_requests=1, gpu_util=0.6)

    assert info.llm_id == "llm_ubuntu_qwen_27b_mock"
    assert info.model_name == "qwen-27b"
    assert info.endpoint_url == "mock://custom"
    assert info.device_id == "ubuntu_server"
    assert info.model_size_b == 27
    assert info.accelerator == "gpu"
    assert info.max_concurrency == 2
    assert info.metadata == {"placement": "local"}
    assert state.llm_id == "llm_ubuntu_qwen_27b_mock"
    assert state.model_name == "qwen-27b"
    assert state.device_id == "ubuntu_server"
    assert state.queue_len == 2
    assert state.running_requests == 1
    assert state.gpu_util == 0.6
    assert state.tokens_per_sec == 80


def test_mock_llm_runtime_validates_configuration() -> None:
    with pytest.raises(ValueError, match="llm_id must be non-empty"):
        MockLLMRuntime(llm_id="", model_name="qwen-7b", device_id="edge", tokens_per_sec=10)

    with pytest.raises(ValueError, match="model_name must be non-empty"):
        MockLLMRuntime(llm_id="llm", model_name="", device_id="edge", tokens_per_sec=10)

    with pytest.raises(ValueError, match="device_id must be non-empty"):
        MockLLMRuntime(llm_id="llm", model_name="qwen-7b", device_id="", tokens_per_sec=10)

    with pytest.raises(ValueError, match="tokens_per_sec must be positive"):
        MockLLMRuntime(llm_id="llm", model_name="qwen-7b", device_id="edge", tokens_per_sec=0)

    with pytest.raises(ValueError, match="max_concurrency must be at least 1"):
        MockLLMRuntime(
            llm_id="llm",
            model_name="qwen-7b",
            device_id="edge",
            tokens_per_sec=10,
            max_concurrency=0,
        )
