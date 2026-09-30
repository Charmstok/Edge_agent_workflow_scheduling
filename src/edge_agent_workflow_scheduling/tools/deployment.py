"""Construct existing local Tools and validate their persisted output artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image

from edge_agent_workflow_scheduling.tools import (
    DocumentToolConfig,
    ImagePreprocessConfig,
    ImagePreprocessTool,
    OCRConfig,
    OCRTool,
    PDFParseTool,
    PDFRenderConfig,
    PDFRenderTool,
    resolve_local_path,
)


def create_local_tool(tool_name, *, input_root, output_dir, timeout_sec, options=None):
    """Build one implementation per worker, without duplicating Tool algorithms."""
    options = options or {}
    common = {"local_root": Path(input_root), "output_dir": Path(output_dir)}
    if tool_name == "image_preprocess":
        return ImagePreprocessTool(ImagePreprocessConfig(**common, **options))
    common["timeout_sec"] = timeout_sec
    if tool_name == "ocr":
        return OCRTool(OCRConfig(**common, **options))
    if tool_name == "pdf_parse":
        return PDFParseTool(DocumentToolConfig(**common, **options))
    if tool_name == "pdf_render":
        return PDFRenderTool(PDFRenderConfig(**common, **options))
    raise ValueError(f"unsupported local Tool: {tool_name}")


def dependency_report(tool, timeout_sec: float) -> dict[str, Any]:
    packages = {
        "Pillow": importlib.metadata.version("Pillow"),
        "jsonschema": importlib.metadata.version("jsonschema"),
    }
    report = {"packages": packages, "executables": {}}
    if hasattr(tool, "check_available"):
        tool.check_available()
    if tool.tool_name == "pdf_parse":
        packages["pypdf"] = importlib.metadata.version("pypdf")
    if tool.tool_name in {"ocr", "pdf_render"}:
        executable = tool.config.executable
        flag = "--version" if tool.tool_name == "ocr" else "-v"
        result = subprocess.run(
            [executable, flag], capture_output=True, text=True, timeout=timeout_sec, check=True
        )
        version = (result.stdout + result.stderr).splitlines()[0]
        report["executables"][executable] = {"path": shutil.which(executable), "version": version}
        if tool.tool_name == "ocr":
            languages = subprocess.run(
                [executable, "--list-langs"],
                capture_output=True,
                text=True,
                check=True,
                timeout=timeout_sec,
            ).stdout.splitlines()
            if any(language not in languages for language in tool.config.language.split("+")):
                raise ModuleNotFoundError(f"Tesseract language unavailable: {tool.config.language}")
            report["ocr_languages"] = languages[1:]
    return report


def _artifact(uri: str, root: Path | None) -> Path:
    path = resolve_local_path(uri, Path(".")).resolve()
    if root is not None and not path.is_relative_to(root.resolve()):
        raise ValueError("Tool output artifact escaped its replica output directory")
    if not path.is_file():
        raise ValueError(f"Tool output artifact missing: {path}")
    return path


def _image_digest(uri, root):
    with Image.open(_artifact(uri, root)) as image:
        return {
            "size": list(image.size),
            "mode": image.mode,
            "pixels_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
        }


def canonical_tool_output(tool_name: str, output: Any, *, artifact_root=None) -> dict:
    """Check schema, counts and artifacts; return path-independent content."""
    if not isinstance(output, dict):
        raise ValueError("Tool output must be an object")
    root = Path(artifact_root) if artifact_root is not None else None
    if tool_name == "image_preprocess":
        return {"image": _image_digest(output["output_uri"], root)}
    if output.get("schema_version") != 1:
        raise ValueError("document output requires schema_version=1")
    if tool_name in {"ocr", "pdf_parse"}:
        content = _artifact(output["text_uri"], root).read_bytes()
        if hashlib.sha256(content).hexdigest() != output["text_sha256"]:
            raise ValueError("text artifact digest mismatch")
        text = content.decode("utf-8")
        if output["text_chars"] != len(text) or not isinstance(output["text"], str):
            raise ValueError("text output schema mismatch")
        if not isinstance(output["text_truncated"], bool) or output["input_count"] < 1:
            raise ValueError("invalid document count/truncation")
        normalized = " ".join(text.split())
        return {"normalized_text": normalized, "input_count": output["input_count"]}
    if tool_name == "pdf_render":
        documents = output["documents"]
        if not isinstance(documents, list) or len(documents) != output["input_count"]:
            raise ValueError("render input_count mismatch")
        canonical = []
        page_count = 0
        for document in documents:
            pages = document["pages"]
            if not pages or [page["page_number"] for page in pages] != list(
                range(1, len(pages) + 1)
            ):
                raise ValueError("render page ordering/count mismatch")
            canonical.append([_image_digest(page["image_uri"], root) for page in pages])
            page_count += len(pages)
        if (
            page_count < 1
            or output["page_count"] != page_count
            or output["image_count"] != page_count
        ):
            raise ValueError("render page_count/image_count mismatch")
        return {"documents": canonical}
    raise ValueError(f"unsupported Tool output: {tool_name}")


def canonical_digest(content: dict) -> str:
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
