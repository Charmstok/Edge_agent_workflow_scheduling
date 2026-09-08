"""Tesseract-backed OCR Tool."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image

from edge_agent_workflow_scheduling.tools.document_common import (
    DocumentExtractionTool,
    DocumentToolConfig,
    _remaining,
)


@dataclass(frozen=True, slots=True)
class OCRConfig(DocumentToolConfig):
    executable: str = "tesseract"
    language: str = "eng"
    page_segmentation_mode: int = 6
    threads: int = 1

    def __post_init__(self) -> None:
        DocumentToolConfig.__post_init__(self)
        if not self.language.strip() or not self.executable.strip():
            raise ValueError("language and executable must be non-empty")
        if (
            isinstance(self.page_segmentation_mode, bool)
            or not isinstance(self.page_segmentation_mode, int)
            or not 3 <= self.page_segmentation_mode <= 13
        ):
            raise ValueError("page_segmentation_mode must be an integer from 3 to 13")
        if isinstance(self.threads, bool) or not isinstance(self.threads, int) or self.threads < 1:
            raise ValueError("threads must be a positive integer")


@dataclass(frozen=True, slots=True)
class OCRTool(DocumentExtractionTool):
    config: OCRConfig
    tool_name = "ocr"
    backend_name = "tesseract"
    work_unit_name = "image_pixels"
    description = (
        "Extract text from one or more local image files using Tesseract OCR. "
        "input_uri must reference an image file or a JSON batch manifest containing "
        "image file URIs. The result contains bounded inline text and a local artifact "
        "reference for the complete OCR output. This tool does not process PDF files "
        "or perform structured field extraction."
    )

    def check_available(self) -> None:
        if shutil.which(self.config.executable) is None:
            raise ModuleNotFoundError("Tesseract executable is unavailable; install tesseract")

    def _extract(self, path: Path, output: Path, deadline: float) -> dict[str, Any]:
        with Image.open(path) as image:
            width, height = image.size
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("use a batch manifest for multiple image pages")
        environment = {**os.environ, "OMP_THREAD_LIMIT": str(self.config.threads)}
        version = subprocess.run(
            [self.config.executable, "--version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=_remaining(deadline),
            env=environment,
        ).stdout.splitlines()[0]
        started = perf_counter()
        result = subprocess.run(
            [
                self.config.executable,
                str(path),
                "stdout",
                "-l",
                self.config.language,
                "--psm",
                str(self.config.page_segmentation_mode),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
            timeout=_remaining(deadline),
            env=environment,
        )
        elapsed = perf_counter() - started
        output.write_text(result.stdout, encoding="utf-8")
        return {
            "image_count": 1,
            "image_pixels": width * height,
            "backend_time_sec": elapsed,
            "backend_version": version,
            "backend_configuration": {
                "language": self.config.language,
                "psm": self.config.page_segmentation_mode,
                "threads": self.config.threads,
            },
        }
