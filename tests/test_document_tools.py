from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import pytest

from edge_agent_workflow_scheduling.common import ToolCall
from edge_agent_workflow_scheduling.executors import LocalToolExecutor
from edge_agent_workflow_scheduling.profiler.tool_consistency import compare_tool_results
from edge_agent_workflow_scheduling.resources import ToolConsistencySample, ToolReplicaProfile
from edge_agent_workflow_scheduling.tools import (
    DocumentToolConfig,
    OCRConfig,
    OCRTool,
    PDFParseTool,
    ToolRegistry,
    resolve_local_path,
)
from edge_agent_workflow_scheduling.workers import LocalWorker

FIXTURES = Path("configs/workload_fixtures_v1").resolve()
HAS_PDF = importlib.util.find_spec("pypdf") is not None
HAS_OCR = shutil.which("tesseract") is not None


def _executor(tool, replica_id="replica-1", energy_profile=None):
    registry = ToolRegistry()
    registry.register(tool)
    profile = ToolReplicaProfile(
        replica_id=replica_id,
        tool_name=tool.tool_name,
        node_id="local",
        platform="test",
        implementation_version=tool.implementation_version,
        executor_type="local",
        energy_profile=energy_profile or {},
    )
    return LocalToolExecutor(LocalWorker(profile, registry))


def _call(tool_name, input_uri, call_id="tool-call"):
    return ToolCall(
        tool_call_id=call_id,
        call_id="function-1",
        run_id="run-1",
        agent_id="agent-1",
        tool_name=tool_name,
        arguments={"input_uri": str(input_uri)},
    )


@pytest.mark.parametrize("tool_name", ["ocr", "pdf_parse"])
def test_real_tools_return_measured_work_and_bounded_artifacts(tmp_path, tool_name):
    if tool_name == "ocr" and not HAS_OCR or tool_name == "pdf_parse" and not HAS_PDF:
        pytest.skip("optional real Tool dependency unavailable")
    tool = (
        OCRTool(OCRConfig(output_dir=tmp_path, inline_text_chars=30))
        if tool_name == "ocr"
        else PDFParseTool(
            DocumentToolConfig(
                output_dir=tmp_path,
                inline_text_chars=30,
            )
        )
    )
    extension = "png" if tool_name == "ocr" else "pdf"
    result = _executor(tool).execute(_call(tool_name, FIXTURES / f"alpha-small.{extension}"))
    assert result.success, result.error_message
    assert result.execution_time_sec > 0
    assert result.metadata["execution_time_source"] == "measured_wall_clock"
    assert result.metadata["energy_source"] == "unavailable"
    assert result.metadata["sec_per_work_unit"] > 0
    assert result.metadata["work_units"] == (307200 if tool_name == "ocr" else 1)
    assert result.metadata["backend_time_sec"] > 0
    output = result.output
    assert output["text_truncated"]
    assert len(output["text"]) == 30
    full_text = resolve_local_path(output["text_uri"], tmp_path).read_bytes()
    assert hashlib.sha256(full_text).hexdigest() == output["text_sha256"]
    assert b"ALPHA" in full_text
    assert result.metadata["backend_version"]


@pytest.mark.skipif(not HAS_PDF, reason="optional pypdf unavailable")
def test_pdf_batch_work_and_consistency_use_complete_artifact(tmp_path):
    batch = tmp_path / "batch.json"
    batch.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "input_uris": [
                    str(FIXTURES / "alpha-medium.pdf"),
                    str(FIXTURES / "alpha-large.pdf"),
                ],
            }
        )
    )
    results = []
    for replica_id in ("first", "second"):
        tool = PDFParseTool(
            DocumentToolConfig(output_dir=tmp_path / replica_id, inline_text_chars=0)
        )
        results.append(_executor(tool, replica_id).execute(_call("pdf_parse", batch)))
    sample = ToolConsistencySample(
        sample_id="batch",
        tool_name="pdf_parse",
        arguments={"input_uri": str(batch)},
        expected={"page_count": 9, "input_count": 2, "text_contains": ["budget: 150.00"]},
        numeric_tolerances={"page_count": 0},
    )
    report = compare_tool_results(sample, results)
    assert report["passed"]
    assert results[0].output["text"] == ""
    assert results[0].metadata["completed_inputs"] == 2
    changed = replace(
        results[1],
        metadata={
            **results[1].metadata,
            "backend_version": "different-engine-version",
        },
    )
    assert compare_tool_results(sample, [results[0], changed])["requires_separate_quality_profile"]
    different = replace(sample, expected={"page_count": 10}, numeric_tolerances={"page_count": 0})
    assert not compare_tool_results(different, results)["passed"]
    path = resolve_local_path(results[1].output["text_uri"], tmp_path)
    path.write_text("corrupted result")
    assert not compare_tool_results(sample, results)["passed"]


@pytest.mark.parametrize(
    "input_value,code",
    [
        ("missing.pdf", "input_not_found"),
        ("https://example.invalid/input.pdf", "invalid_input"),
    ],
)
def test_input_failures_are_structured(tmp_path, input_value, code):
    result = _executor(PDFParseTool(DocumentToolConfig(output_dir=tmp_path))).execute(
        _call("pdf_parse", input_value),
    )
    assert not result.success
    assert result.error_code == code


def test_missing_dependency_does_not_break_registration(tmp_path):
    tool = OCRTool(OCRConfig(output_dir=tmp_path, executable="missing-tesseract-for-test"))
    result = _executor(tool).execute(_call("ocr", FIXTURES / "alpha-small.png"))
    assert result.error_code == "dependency_unavailable"


def test_invalid_batch_and_registry_arguments(tmp_path):
    executor = _executor(PDFParseTool(DocumentToolConfig(output_dir=tmp_path)))
    call = _call("pdf_parse", "unused.pdf")
    call.arguments = {"extra": 1}
    assert executor.execute(call).error_code == "invalid_arguments"
    batch = tmp_path / "batch.json"
    batch.write_text('{"schema_version": 1, "input_uris": []}')
    assert executor.execute(_call("pdf_parse", batch)).error_code == "invalid_input"


@pytest.mark.skipif(not HAS_PDF, reason="optional pypdf unavailable")
def test_corrupt_pdf_reports_backend_failure(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"not a PDF")
    tool = PDFParseTool(DocumentToolConfig(output_dir=tmp_path / "out"))
    result = _executor(tool).execute(_call("pdf_parse", path))
    assert result.error_code == "backend_execution_failed"
    assert result.execution_time_sec > 0


def test_timeout_kills_backend_instead_of_waiting_for_completion(tmp_path):
    executable = tmp_path / "fake-tesseract"
    executable.write_text(
        f"#!{sys.executable}\nimport os, sys, time\n"
        "if '--version' in sys.argv:\n    print('test-engine')\n"
        f"else:\n    open({str(tmp_path / 'pid')!r}, 'w').write(str(os.getpid()))\n"
        "    time.sleep(30)\n",
    )
    executable.chmod(0o755)
    tool = OCRTool(OCRConfig(output_dir=tmp_path / "out", executable=str(executable)))
    started = perf_counter()
    result = _executor(tool).execute(
        _call("ocr", FIXTURES / "alpha-small.png"),
        timeout_sec=1.5,
    )
    assert result.error_code == "timeout"
    assert perf_counter() - started < 5
    pid = int((tmp_path / "pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_timeout_budget_is_shared_across_batch_inputs(tmp_path, monkeypatch):
    tool = PDFParseTool(DocumentToolConfig(output_dir=tmp_path / "out"))
    batch = tmp_path / "batch.json"
    batch.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "input_uris": [
                    str(FIXTURES / "alpha-small.pdf"),
                    str(FIXTURES / "alpha-small.pdf"),
                ],
            }
        )
    )
    deadlines = []

    def extract(self, path, output, deadline):
        deadlines.append(deadline)
        if len(deadlines) > 1:
            raise subprocess.TimeoutExpired("pdf-extract", 0.01)
        output.write_text("ALPHA")
        return {"page_count": 1, "backend_version": "test", "backend_time_sec": 0.01}

    monkeypatch.setattr(PDFParseTool, "check_available", lambda self: None)
    monkeypatch.setattr(PDFParseTool, "_extract", extract)
    result = _executor(tool).execute(_call("pdf_parse", batch), timeout_sec=1)
    assert result.error_code == "timeout"
    assert result.metadata["completed_inputs"] == 1
    assert deadlines[0] == deadlines[1]
    assert result.output is None


def test_energy_profile_is_not_reported_as_measured(tmp_path, monkeypatch):
    tool = PDFParseTool(DocumentToolConfig(output_dir=tmp_path))
    monkeypatch.setattr(PDFParseTool, "check_available", lambda self: None)

    def extract(self, path, output, deadline):
        output.write_text("test")
        return {"page_count": 1, "backend_version": "test", "backend_time_sec": 0.01}

    monkeypatch.setattr(PDFParseTool, "_extract", extract)
    result = _executor(tool, energy_profile={"joules_per_call": 2.5}).execute(
        _call("pdf_parse", FIXTURES / "alpha-small.pdf"),
    )
    assert result.energy_joules == 2.5
    assert result.metadata["energy_source"] == "profiled"


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf")])
def test_invalid_document_timeouts_rejected(tmp_path, value):
    with pytest.raises(ValueError):
        DocumentToolConfig(output_dir=tmp_path, timeout_sec=value)


def test_zero_remaining_executor_budget_does_not_execute(tmp_path, monkeypatch):
    executor = _executor(PDFParseTool(DocumentToolConfig(output_dir=tmp_path)))
    executor.queue_wait_time_sec = 1
    result = executor.execute(_call("pdf_parse", "unused.pdf"), timeout_sec=0.5)
    assert result.error_code == "timeout"
    assert result.execution_time_sec == 0
