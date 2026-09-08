"""Shared implementation for bounded document-backed Tools.

This module contains execution plumbing only. Concrete Tools live in their own
modules so their public contracts and backend behavior remain easy to inspect.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Any, ClassVar

from edge_agent_workflow_scheduling.tools.base import ToolExecution, ToolSpec
from edge_agent_workflow_scheduling.tools.paths import resolve_local_path


@dataclass(frozen=True, slots=True)
class DocumentToolConfig:
    """Local paths and bounded output policy shared by document Tools."""

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


class DocumentExtractionTool:
    """Base for text-extraction Tools with one deadline per invocation."""

    config: DocumentToolConfig
    tool_name: str
    description: ClassVar[str]
    implementation_version: ClassVar[str] = "1.0.0"
    backend_name: ClassVar[str]
    work_unit_name: ClassVar[str]

    @property
    def spec(self) -> ToolSpec:
        return {
            "type": "function",
            "name": self.tool_name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {"input_uri": {"type": "string", "minLength": 1}},
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
            "backend": self.backend_name,
            "execution_time_source": "measured_wall_clock",
            "energy_source": "unavailable",
            "work_features": {},
            "completed_inputs": 0,
            "backend_time_sec": 0.0,
            "timeout_scope": "invocation",
        }
        try:
            _remaining(deadline)
            paths = input_paths(arguments, self.config.local_root)
            metadata["work_features"] = {
                "input_count": len(paths),
                "input_bytes": sum(path.stat().st_size for path in paths),
            }
            self.check_available()
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            prefix = hashlib.sha256(invocation_id.encode()).hexdigest()[:16]
            with tempfile.TemporaryDirectory(prefix="tool-") as scratch:
                texts: list[str] = []
                for index, path in enumerate(paths):
                    text_path = Path(scratch) / f"{index}.txt"
                    observed = self._extract(path, text_path, deadline)
                    metadata["backend_version"] = observed.pop("backend_version")
                    metadata["backend_time_sec"] += observed.pop("backend_time_sec")
                    metadata["backend_configuration"] = observed.pop(
                        "backend_configuration", {}
                    )
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
            metadata["work_units"] = metadata["work_features"].get(self.work_unit_name, 0)
            metadata["work_unit"] = self.work_unit_name.removesuffix("_count")
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

    def check_available(self) -> None:
        raise NotImplementedError

    def _extract(self, path: Path, output: Path, deadline: float) -> dict[str, Any]:
        raise NotImplementedError


def _validate_timeout(timeout_sec: float | None) -> None:
    if timeout_sec is not None and (
        isinstance(timeout_sec, bool)
        or not isinstance(timeout_sec, int | float)
        or not isfinite(timeout_sec)
        or timeout_sec <= 0
    ):
        raise ValueError("timeout_sec must be finite and positive")


def _remaining(deadline: float) -> float:
    remaining = deadline - perf_counter()
    if remaining <= 0:
        raise TimeoutError("document Tool budget exhausted")
    return remaining


def input_paths(arguments: dict[str, Any], root: Path, *, suffix: str | None = None) -> list[Path]:
    """Resolve a single file or a JSON batch manifest into validated paths."""

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
        if suffix is not None and item.suffix.lower() != suffix:
            raise ValueError(f"expected {suffix} input, got {item.suffix or 'no extension'}")
    return paths
