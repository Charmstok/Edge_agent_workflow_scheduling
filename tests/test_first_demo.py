import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from edge_agent_workflow_scheduling.profiler import JsonlTraceLogger


def _load_first_demo_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_first_demo.py"
    spec = importlib.util.spec_from_file_location("run_first_demo", script_path)
    if spec is None or spec.loader is None:
        msg = f"failed to load demo script from {script_path}"
        raise RuntimeError(msg)

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_first_demo = _load_first_demo_module()


def test_first_demo_generates_llm_and_tool_traces(tmp_path: Path) -> None:
    trace_path = tmp_path / "traces" / "first_demo.jsonl"
    data_dir = tmp_path / "inputs"

    summary = run_first_demo.run_demo(
        policy="round_robin",
        trace_path=trace_path,
        data_dir=data_dir,
        steps_per_agent=2,
    )
    records = JsonlTraceLogger(trace_path).read_all()
    llm_records = [record for record in records if record.task_kind == "llm"]
    tool_records = [record for record in records if record.task_kind == "tool"]

    assert summary.trace_path == trace_path
    assert summary.total_records == 8
    assert summary.success_rate == 1.0
    assert trace_path.exists()
    assert len(records) == 8
    assert len(llm_records) == 4
    assert len(tool_records) == 4
    assert summary.worker_counts == {"worker_local_1": 2, "worker_local_2": 2}
    assert summary.llm_counts == {
        "llm_edge_qwen_7b_mock": 2,
        "llm_ubuntu_qwen_27b_mock": 2,
    }
    assert all(record.policy_name == "round_robin" for record in records)
    assert all(record.success for record in records)
    assert all(record.total_latency_sec > 0 for record in records)
    assert all(record.model_name in {"qwen-7b", "qwen-27b"} for record in llm_records)
    assert all(record.input_tokens > 0 for record in llm_records)
    assert all(record.output_tokens > 0 for record in llm_records)
    assert all(record.tool_type == "image_preprocess" for record in tool_records)
    assert all(record.execution_time_sec > 0 for record in tool_records)
