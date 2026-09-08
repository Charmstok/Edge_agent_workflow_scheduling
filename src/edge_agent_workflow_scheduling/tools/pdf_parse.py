"""pypdf-backed PDF text extraction Tool."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from edge_agent_workflow_scheduling.tools.document_common import (
    DocumentExtractionTool,
    DocumentToolConfig,
    _remaining,
)


@dataclass(frozen=True, slots=True)
class PDFParseTool(DocumentExtractionTool):
    config: DocumentToolConfig
    tool_name = "pdf_parse"
    backend_name = "pypdf"
    work_unit_name = "page_count"
    description = (
        "Extract embedded text from one or more local PDF files using pypdf. "
        "input_uri must reference a PDF file or a JSON batch manifest containing "
        "PDF file URIs. The result contains bounded inline text and a local artifact "
        "reference for the complete page-ordered output. This tool does not perform OCR "
        "and may return empty text for image-only PDF pages."
    )

    def check_available(self) -> None:
        if importlib.util.find_spec("pypdf") is None:
            raise ModuleNotFoundError("pypdf is unavailable; install requirements-tools.txt")

    def _extract(self, path: Path, output: Path, deadline: float) -> dict[str, Any]:
        started = perf_counter()
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("_pdf_extract.py")),
                str(path),
                str(output),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=_remaining(deadline),
        )
        observed = json.loads(result.stdout)
        observed["backend_configuration"] = {"extraction_mode": "plain", "implicit_ocr": False}
        observed["backend_time_sec"] = perf_counter() - started
        return observed
