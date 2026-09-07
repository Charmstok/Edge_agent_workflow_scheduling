"""Real OCR/PDF Tools with bounded subprocesses and explicit work-unit measurements."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Any, ClassVar

from PIL import Image

from edge_agent_workflow_scheduling.tools.base import ToolExecution, ToolSpec
from edge_agent_workflow_scheduling.tools.image_preprocess import resolve_local_path


@dataclass(frozen=True, slots=True)
class DocumentToolConfig:
    """Local paths and bounded output policy shared by OCR and PDF extraction."""

    output_dir: Path
    local_root: Path = Path(".")
    timeout_sec: float = 120.0
    inline_text_chars: int = 4096

    def __post_init__(self) -> None:
        if (
            isinstance(self.timeout_sec, bool)
            or not isinstance(self.timeout_sec, int | float)
            or not isfinite(self.timeout_sec)
            or self.timeout_sec <= 0
        ):
            raise ValueError("timeout_sec must be finite and positive")
        if (
            isinstance(self.inline_text_chars, bool)
            or not isinstance(self.inline_text_chars, int)
            or self.inline_text_chars < 0
        ):
            raise ValueError("inline_text_chars must be a non-negative integer")


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


class _DocumentTool:
    config: DocumentToolConfig
    tool_name: str
    description: ClassVar[str]
    implementation_version: ClassVar[str] = "1.0.0"

    @property
    def spec(self) -> ToolSpec:
        return {
            "type": "function",
            "name": self.tool_name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {"input_uri": {"type": "string"}},
                "required": ["input_uri"],
                "additionalProperties": False,
            },
            "strict": True,
        }

    def execute(self, arguments: dict[str, Any], *, invocation_id: str) -> ToolExecution:
        return self.execute_with_timeout(arguments, invocation_id=invocation_id, timeout_sec=None)

    def execute_with_timeout(
        self,
        arguments: dict[str, Any],
        *,
        invocation_id: str,
        timeout_sec: float | None,
    ) -> ToolExecution:
        if timeout_sec is not None and (
            isinstance(timeout_sec, bool)
            or not isinstance(timeout_sec, int | float)
            or not isfinite(timeout_sec)
            or timeout_sec <= 0
        ):
            raise ValueError("timeout_sec must be finite and positive")
        started = perf_counter()
        budget = (
            self.config.timeout_sec
            if timeout_sec is None
            else min(
                self.config.timeout_sec,
                timeout_sec,
            )
        )
        deadline = started + budget
        metadata: dict[str, Any] = {
            "implementation_version": self.implementation_version,
            "backend": "tesseract" if self.tool_name == "ocr" else "pypdf",
            "execution_time_source": "measured_wall_clock",
            "energy_source": "unavailable",
            "work_features": {},
            "completed_inputs": 0,
            "backend_time_sec": 0.0,
            "timeout_scope": "invocation",
        }
        try:
            _remaining(deadline)
            paths = _input_paths(arguments, self.config.local_root)
            metadata["work_features"] = {
                "input_count": len(paths),
                "input_bytes": sum(path.stat().st_size for path in paths),
            }
            self.check_available()
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            prefix = hashlib.sha256(invocation_id.encode()).hexdigest()[:16]
            with tempfile.TemporaryDirectory(prefix="tool-") as scratch:
                texts = []
                for index, path in enumerate(paths):
                    text_path = Path(scratch) / f"{index}.txt"
                    observed = self._extract(path, text_path, deadline)
                    metadata["backend_version"] = observed.pop("backend_version")
                    metadata["backend_time_sec"] += observed.pop("backend_time_sec")
                    configuration = observed.pop("backend_configuration", {})
                    metadata["backend_configuration"] = configuration
                    for name, value in observed.items():
                        features = metadata["work_features"]
                        features[name] = features.get(name, 0) + value
                    texts.append(text_path.read_text(encoding="utf-8"))
                    metadata["completed_inputs"] += 1
                    _remaining(deadline)
                text = "\n\f\n".join(texts)
            text_bytes = text.encode("utf-8")
            digest = hashlib.sha256(text_bytes).hexdigest()
            output_path = self.config.output_dir / f"{prefix}-{digest}.txt"
            output_path.write_bytes(text_bytes)
            limit = self.config.inline_text_chars
            truncated = len(text) > limit
            preview = (
                text[:limit]
                if not truncated
                else (text[: limit // 2] + text[-(limit - limit // 2) :] if limit else "")
            )
            metadata["text_chars"] = len(text)
            metadata["work_units"] = metadata["work_features"].get(
                "image_pixels" if self.tool_name == "ocr" else "page_count",
                0,
            )
            metadata["work_unit"] = "pixel" if self.tool_name == "ocr" else "page"
            metadata["tool_wall_time_sec"] = perf_counter() - started
            metadata["sec_per_work_unit"] = (
                metadata["tool_wall_time_sec"] / metadata["work_units"]
                if metadata["work_units"]
                else None
            )
            _remaining(deadline)
            return ToolExecution(
                success=True,
                output={
                    "schema_version": 1,
                    "text": preview,
                    "text_truncated": truncated,
                    "text_uri": output_path.resolve().as_uri(),
                    "text_sha256": digest,
                    "normalized_text_sha256": hashlib.sha256(
                        " ".join(text.split()).encode()
                    ).hexdigest(),
                    "text_chars": len(text),
                    "input_count": len(paths),
                },
                metadata=metadata,
            )
        except (TimeoutError, subprocess.TimeoutExpired) as exc:
            code, message = "timeout", str(exc) or "document Tool budget exhausted"
        except ModuleNotFoundError as exc:
            code, message = "dependency_unavailable", str(exc)
        except FileNotFoundError as exc:
            code, message = "input_not_found", str(exc)
        except subprocess.CalledProcessError as exc:
            code = "backend_execution_failed"
            message = (exc.stderr or "backend failed")[-2000:]
        except (ValueError, OSError, TypeError, KeyError) as exc:
            code, message = "invalid_input", str(exc)
        metadata["tool_wall_time_sec"] = perf_counter() - started
        return ToolExecution(
            success=False,
            error_code=code,
            error_message=message or code,
            metadata=metadata,
        )


@dataclass(frozen=True, slots=True)
class OCRTool(_DocumentTool):
    config: OCRConfig
    tool_name: ClassVar[str] = "ocr"
    description: ClassVar[str] = (
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


@dataclass(frozen=True, slots=True)
class PDFParseTool(_DocumentTool):
    config: DocumentToolConfig
    tool_name: ClassVar[str] = "pdf_parse"
    description: ClassVar[str] = (
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


def _remaining(deadline: float) -> float:
    remaining = deadline - perf_counter()
    if remaining <= 0:
        raise TimeoutError("document Tool budget exhausted")
    return remaining


def _input_paths(arguments: dict[str, Any], root: Path) -> list[Path]:
    if set(arguments) != {"input_uri"} or not isinstance(arguments["input_uri"], str):
        raise ValueError("expected one input_uri string")
    path = resolve_local_path(arguments["input_uri"], root).resolve()
    paths = [path]
    if path.suffix.lower() == ".json":
        batch = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(batch, dict) or batch.get("schema_version") != 1:
            raise ValueError("batch manifest requires schema_version=1")
        uris = batch.get("input_uris")
        if (
            not isinstance(uris, list)
            or not uris
            or any(not isinstance(uri, str) or not uri.strip() for uri in uris)
        ):
            raise ValueError("batch input_uris must be a non-empty list of paths")
        paths = [resolve_local_path(uri, path.parent).resolve() for uri in uris]
    for item in paths:
        if not item.is_file():
            raise FileNotFoundError(f"input file not found: {item}")
    return paths
