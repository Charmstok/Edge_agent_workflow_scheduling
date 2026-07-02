import pytest

from edge_agent_workflow_scheduling.common import ToolCall
from edge_agent_workflow_scheduling.tools import ToolExecution
from edge_agent_workflow_scheduling.workers import LocalWorker


def test_local_worker_runs_supported_tool_call_with_default_executor() -> None:
    worker = LocalWorker(
        worker_id="worker_local_1",
        supported_tools=["image_preprocess", "ocr"],
    )
    tool_call = ToolCall(
        tool_call_id="tc_001",
        agent_id="agent_1",
        tool_type="image_preprocess",
        input_uri="local://inputs/image.png",
        input_size_mb=4.5,
    )

    result = worker.run_tool(tool_call)

    assert result.tool_call_id == "tc_001"
    assert result.worker_id == "worker_local_1"
    assert result.success is True
    assert result.output_uri == (
        "local://outputs/tools/worker_local_1/image_preprocess/tc_001.json"
    )
    assert result.execution_time_sec >= 0
    assert result.error_message is None


def test_local_worker_runs_custom_tool_executor() -> None:
    worker = LocalWorker(
        worker_id="worker_local_2",
        supported_tools=["pdf_parse"],
        tool_executors={
            "pdf_parse": lambda tool_call: f"local://parsed/{tool_call.tool_call_id}.json",
        },
    )
    tool_call = ToolCall(
        tool_call_id="tc_pdf_001",
        agent_id="agent_pdf",
        tool_type="pdf_parse",
        input_uri="local://inputs/doc.pdf",
        input_size_mb=20,
        page_count=50,
    )

    result = worker.run_tool(tool_call)

    assert result.success is True
    assert result.output_uri == "local://parsed/tc_pdf_001.json"


def test_local_worker_preserves_structured_tool_execution_metadata() -> None:
    worker = LocalWorker(
        worker_id="worker_local_2",
        supported_tools=["pdf_parse"],
        tool_executors={
            "pdf_parse": lambda _: ToolExecution(
                output_uri="local://parsed/tc_pdf_001.json",
                metadata={"estimated_work_units": 50},
            ),
        },
    )
    tool_call = ToolCall(
        tool_call_id="tc_pdf_001",
        agent_id="agent_pdf",
        tool_type="pdf_parse",
        input_uri="local://inputs/doc.pdf",
        input_size_mb=20,
        page_count=50,
    )

    result = worker.run_tool(tool_call)

    assert result.success is True
    assert result.output_uri == "local://parsed/tc_pdf_001.json"
    assert result.metadata == {"estimated_work_units": 50}


def test_local_worker_rejects_unsupported_tool_call() -> None:
    worker = LocalWorker(
        worker_id="worker_local_1",
        supported_tools=["image_preprocess"],
    )
    tool_call = ToolCall(
        tool_call_id="tc_ocr_001",
        agent_id="agent_1",
        tool_type="ocr",
        input_uri="local://inputs/page.png",
        input_size_mb=8.0,
        image_count=1,
    )

    result = worker.run_tool(tool_call)

    assert result.success is False
    assert result.output_uri is None
    assert result.execution_time_sec >= 0
    assert "is not supported" in result.error_message


def test_local_worker_converts_executor_exception_to_failed_result() -> None:
    def failing_executor(_: ToolCall) -> str:
        raise RuntimeError("tool failed")

    worker = LocalWorker(
        worker_id="worker_local_1",
        supported_tools=["image_preprocess"],
        tool_executors={"image_preprocess": failing_executor},
    )
    tool_call = ToolCall(
        tool_call_id="tc_001",
        agent_id="agent_1",
        tool_type="image_preprocess",
        input_uri="local://inputs/image.png",
        input_size_mb=4.5,
    )

    result = worker.run_tool(tool_call)

    assert result.success is False
    assert result.error_message == "tool failed"


def test_local_worker_info_and_state_snapshots() -> None:
    worker = LocalWorker(
        worker_id="worker_local_1",
        supported_tools=["ocr", "ocr", "pdf_parse"],
        max_concurrency=2,
        metadata={"device": "local"},
    )

    info = worker.to_info()
    state = worker.get_state(queue_len=3, cpu_util=0.7, memory_util=0.4)

    assert info.worker_id == "worker_local_1"
    assert info.supported_tools == ["ocr", "pdf_parse"]
    assert info.max_concurrency == 2
    assert info.metadata == {"device": "local"}
    assert state.worker_id == "worker_local_1"
    assert state.supported_tools == ["ocr", "pdf_parse"]
    assert state.queue_len == 3
    assert state.cpu_util == 0.7
    assert state.memory_util == 0.4


def test_local_worker_validates_configuration() -> None:
    with pytest.raises(ValueError, match="worker_id must be non-empty"):
        LocalWorker(worker_id="", supported_tools=["ocr"])

    with pytest.raises(ValueError, match="supported_tools must be non-empty"):
        LocalWorker(worker_id="worker_local_1", supported_tools=[])

    with pytest.raises(ValueError, match="max_concurrency must be at least 1"):
        LocalWorker(worker_id="worker_local_1", supported_tools=["ocr"], max_concurrency=0)

    with pytest.raises(ValueError, match="artificial_delay_sec must be non-negative"):
        LocalWorker(
            worker_id="worker_local_1",
            supported_tools=["ocr"],
            artificial_delay_sec=-1,
        )
