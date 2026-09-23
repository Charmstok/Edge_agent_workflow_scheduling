"""PDF-to-page-image rendering Tool for OCR fallback workflows."""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from edge_agent_workflow_scheduling.tools.base import ToolExecution, ToolSpec
from edge_agent_workflow_scheduling.tools.document_common import (
    DocumentToolConfig,
    _remaining,
    _validate_timeout,
    input_paths,
)


@dataclass(frozen=True, slots=True)
class PDFRenderConfig(DocumentToolConfig):
    """Configuration for Poppler PDF page rendering."""

    executable: str = "pdftoppm"
    dpi: int = 150

    def __post_init__(self) -> None:
        DocumentToolConfig.__post_init__(self)
        if not self.executable.strip():
            raise ValueError("executable must be non-empty")
        if isinstance(self.dpi, bool) or not isinstance(self.dpi, int) or self.dpi < 36:
            raise ValueError("dpi must be an integer >= 36")


@dataclass(frozen=True, slots=True)
class PDFRenderTool:
    """Render each PDF page to a PNG artifact using Poppler."""

    config: PDFRenderConfig
    tool_name: str = "pdf_render"
    implementation_version: str = "1.0.0"

    @property
    def spec(self) -> ToolSpec:
        return {
            "type": "function",
            "name": self.tool_name,
            "description": (
                "Render one or more local PDF files into page PNG artifacts. "
                "Use the returned page image URIs as inputs to OCR; this tool does "
                "not extract text or perform OCR itself."
            ),
            "parameters": {
                "type": "object",
                "properties": {"input_uri": {"type": "string", "minLength": 1}},
                "required": ["input_uri"],
                "additionalProperties": False,
            },
            "strict": True,
        }

    def check_available(self) -> None:
        if shutil.which(self.config.executable) is None:
            raise ModuleNotFoundError("pdftoppm executable is unavailable; install Poppler")

    def execute(self, arguments: dict[str, Any], *, invocation_id: str) -> ToolExecution:
        return self.execute_with_timeout(arguments, invocation_id=invocation_id, timeout_sec=None)

    def execute_with_timeout(
        self,
        arguments: dict[str, Any],
        *,
        invocation_id: str,
        timeout_sec: float | None,
    ) -> ToolExecution:
        _validate_timeout(timeout_sec)
        started = perf_counter()
        budget = (
            self.config.timeout_sec
            if timeout_sec is None
            else min(self.config.timeout_sec, timeout_sec)
        )
        deadline = started + budget
        metadata: dict[str, Any] = {
            "implementation_version": self.implementation_version,
            "backend": "pdftoppm",
            "execution_time_source": "measured_wall_clock",
            "energy_source": "unavailable",
            "work_features": {},
            "completed_inputs": 0,
            "backend_time_sec": 0.0,
            "timeout_scope": "invocation",
            "backend_configuration": {"dpi": self.config.dpi, "format": "png"},
        }
        try:
            _remaining(deadline)
            paths = input_paths(arguments, self.config.local_root, suffix=".pdf")
            metadata["work_features"] = {
                "input_count": len(paths),
                "input_bytes": sum(path.stat().st_size for path in paths),
            }
            self.check_available()
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            safe_prefix = (
                re.sub(r"[^A-Za-z0-9_.-]", "_", invocation_id)[:48]
                + "-"
                + hashlib.sha256(invocation_id.encode()).hexdigest()[:16]
            )
            documents: list[dict[str, Any]] = []
            for index, path in enumerate(paths):
                prefix = self.config.output_dir / f"{safe_prefix}-{index}"
                before = set(self.config.output_dir.glob(f"{prefix.name}-*.png"))
                render_started = perf_counter()
                subprocess.run(
                    [
                        self.config.executable,
                        "-png",
                        "-r",
                        str(self.config.dpi),
                        str(path),
                        str(prefix),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=_remaining(deadline),
                )
                metadata["backend_time_sec"] += perf_counter() - render_started
                rendered = sorted(
                    set(self.config.output_dir.glob(f"{prefix.name}-*.png")) - before,
                    key=lambda item: _page_number(item),
                )
                if not rendered:
                    raise RuntimeError(f"pdftoppm produced no pages for {path}")
                documents.append(
                    {
                        "input_uri": path.as_uri(),
                        "pages": [
                            {
                                "page_number": _page_number(image),
                                "image_uri": image.resolve().as_uri(),
                            }
                            for image in rendered
                        ],
                    }
                )
                metadata["completed_inputs"] += 1
                metadata["work_features"]["page_count"] = (
                    metadata["work_features"].get("page_count", 0) + len(rendered)
                )
                metadata["work_features"]["image_count"] = (
                    metadata["work_features"].get("image_count", 0) + len(rendered)
                )
                _remaining(deadline)
            page_count = metadata["work_features"].get("page_count", 0)
            metadata["work_units"] = page_count
            metadata["work_unit"] = "page"
            metadata["tool_wall_time_sec"] = perf_counter() - started
            metadata["sec_per_work_unit"] = (
                metadata["tool_wall_time_sec"] / page_count if page_count else None
            )
            return ToolExecution(
                success=True,
                output={
                    "schema_version": 1,
                    "documents": documents,
                    "input_count": len(paths),
                    "page_count": page_count,
                    "image_count": metadata["work_features"]["image_count"],
                },
                metadata=metadata,
            )
        except (TimeoutError, subprocess.TimeoutExpired) as exc:
            code, message = "timeout", str(exc) or "PDF render budget exhausted"
        except ModuleNotFoundError as exc:
            code, message = "dependency_unavailable", str(exc)
        except FileNotFoundError as exc:
            code, message = "input_not_found", str(exc)
        except subprocess.CalledProcessError as exc:
            code = "backend_execution_failed"
            message = (exc.stderr or "pdftoppm failed")[-2000:]
        except (ValueError, OSError, TypeError, KeyError, RuntimeError) as exc:
            code, message = "invalid_input", str(exc)
        metadata["tool_wall_time_sec"] = perf_counter() - started
        return ToolExecution(
            success=False,
            error_code=code,
            error_message=message or code,
            metadata=metadata,
        )


def _page_number(path: Path) -> int:
    match = re.search(r"-(\d+)\.png$", path.name)
    if match is None:
        raise ValueError(f"cannot determine page number from {path.name}")
    return int(match.group(1))
