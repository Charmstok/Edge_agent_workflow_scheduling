from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "run_workload", Path(__file__).resolve().parents[1] / "scripts/run_workload.py"
)
assert _SCRIPT_SPEC is not None and _SCRIPT_SPEC.loader is not None
_SCRIPT = importlib.util.module_from_spec(_SCRIPT_SPEC)
sys.modules["run_workload"] = _SCRIPT
_SCRIPT_SPEC.loader.exec_module(_SCRIPT)
run_live = _SCRIPT.run_live
run_replay = _SCRIPT.run_replay
run_scripted = _SCRIPT.run_scripted


ROOT = Path(__file__).resolve().parents[1]
WORKLOAD = ROOT / "configs/workload_milestone_4_1_v1.json"
LLM_CONFIG = ROOT / "configs/llm_profiles.toml"


def _trace_paths(root: Path) -> list[Path]:
    return sorted((root / "scripted").glob("*/trace.json"))


def test_scripted_workload_covers_all_task_types_and_is_reproducible(tmp_path: Path) -> None:
    first = run_scripted(
        WORKLOAD,
        scenario="low_load",
        split="validation",
        output_dir=tmp_path / "first",
        profile_seed=17,
    )
    second = run_scripted(
        WORKLOAD,
        scenario="low_load",
        split="validation",
        output_dir=tmp_path / "second",
        profile_seed=17,
    )

    assert first["input_fingerprint"] == second["input_fingerprint"]
    first_traces = _trace_paths(tmp_path / "first")
    second_traces = _trace_paths(tmp_path / "second")
    assert len(first_traces) == len(second_traces) == 9
    assert set(first["task_types"]) == {"image_ocr", "pdf_extract", "document_reconcile"}
    for left, right in zip(first_traces, second_traces, strict=True):
        left_data = json.loads(left.read_text(encoding="utf-8"))
        right_data = json.loads(right.read_text(encoding="utf-8"))
        assert left_data["run"]["run_id"] == right_data["run"]["run_id"]
        assert [call["call_id"] for call in left_data["calls"]] == [
            call["call_id"] for call in right_data["calls"]
        ]
        assert [call["parameter_summary"] for call in left_data["calls"]] == [
            call["parameter_summary"] for call in right_data["calls"]
        ]


def test_replay_compares_policies_and_keeps_failed_calls(tmp_path: Path) -> None:
    run_scripted(
        WORKLOAD,
        scenario="low_load",
        split="validation",
        output_dir=tmp_path,
        profile_seed=3,
    )
    summary = run_replay(
        _trace_paths(tmp_path),
        workload_path=WORKLOAD,
        output_dir=tmp_path,
        policies=("round_robin", "least_queue"),
        seeds=(0,),
        profile_seed=3,
        profile_failure_rate=1.0,
    )

    assert summary["policy_count"] == 2
    assert summary["trace_count"] == 9
    generated = list(
        (tmp_path / "replay").glob("*/workload-*/profile-*/experiment-*/*/trace.json")
    )
    assert generated
    failed_trace = json.loads(generated[0].read_text(encoding="utf-8"))
    assert any(not call["success"] for call in failed_trace["calls"])
    assert failed_trace["calls"][0]["call_id"].endswith("llm-0000")
    replay_summary = json.loads(
        generated[0].with_name("summary.json").read_text(encoding="utf-8")
    )
    assert replay_summary["task_score"]["normalized_score"] == 0.0


def test_live_without_verified_function_calling_is_not_validated(tmp_path: Path) -> None:
    summary = run_live(
        WORKLOAD,
        config_path=LLM_CONFIG,
        llm_id="local-qwen35-9b",
        repeats=3,
        output_dir=tmp_path,
        scenario="low_load",
        split="validation",
    )

    assert summary["status"] == "not_validated"
    assert summary["metrics"] is None
    assert "Function Calling" in summary["reason"]
