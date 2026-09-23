from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from edge_agent_workflow_scheduling.profiler.tool_sampling import ToolSamplingConfig
from edge_agent_workflow_scheduling.tools import PDFRenderConfig, PDFRenderTool

ROOT = Path(__file__).resolve().parents[1]


def test_sampling_matrix_declares_every_implemented_tool() -> None:
    expected = {"image_preprocess", "ocr", "pdf_parse", "pdf_render"}
    for name in ("tool_profile_sampling_v1.json", "tool_profile_validation_v1.json"):
        config = ToolSamplingConfig.from_json(ROOT / "configs" / name)
        assert set(config.tools) == expected
        assert set(config.input_templates) == expected
        assert config.pdf_render["dpi"] == 150


@pytest.mark.skipif(shutil.which("pdftoppm") is None, reason="Poppler is unavailable")
def test_pdf_render_repeated_invocations_use_distinct_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "rendered"
    tool = PDFRenderTool(
        PDFRenderConfig(
            local_root=ROOT / "configs",
            output_dir=output_dir,
            dpi=72,
        )
    )
    arguments = {"input_uri": "workload_fixtures_v1/alpha-small.pdf"}

    first = tool.execute(arguments, invocation_id="repeated-invocation-0001")
    second = tool.execute(arguments, invocation_id="repeated-invocation-0002")

    assert first.success is True
    assert second.success is True
    first_uri = first.output["documents"][0]["pages"][0]["image_uri"]
    second_uri = second.output["documents"][0]["pages"][0]["image_uri"]
    assert first_uri != second_uri
    assert Path(first_uri.removeprefix("file://")).is_file()
    assert Path(second_uri.removeprefix("file://")).is_file()
