"""Image preprocessing Tool backed by Pillow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlparse

from PIL import Image, ImageFilter

from edge_agent_workflow_scheduling.common import ToolCall
from edge_agent_workflow_scheduling.tools.base import ToolExecution, ToolSpec

ImageOperation = Literal["grayscale", "resize", "blur", "threshold", "edge_detect"]

ALL_IMAGE_OPERATIONS: tuple[ImageOperation, ...] = (
    "grayscale",
    "resize",
    "blur",
    "threshold",
    "edge_detect",
)
DEFAULT_IMAGE_OPERATIONS: tuple[ImageOperation, ...] = (
    "grayscale",
    "resize",
    "blur",
    "threshold",
)


@dataclass(frozen=True, slots=True)
class ImageProfile:
    """Image dimensions and mode used in execution metadata."""

    width: int
    height: int
    mode: str

    @property
    def pixel_count(self) -> int:
        return self.width * self.height


@dataclass(frozen=True, slots=True)
class ImagePreprocessConfig:
    """Configuration for the Pillow image preprocessing backend."""

    output_dir: Path
    local_root: Path = Path(".")
    operations: tuple[ImageOperation, ...] = DEFAULT_IMAGE_OPERATIONS
    operation_repeat: int = 1
    resize_scale: float = 0.5
    threshold: int = 128
    output_suffix: str = ".png"

    def __post_init__(self) -> None:
        if self.operation_repeat < 1:
            raise ValueError("operation_repeat must be at least 1")
        if not self.operations:
            raise ValueError("operations must be non-empty")
        unsupported = sorted(set(self.operations) - set(ALL_IMAGE_OPERATIONS))
        if unsupported:
            raise ValueError(f"unsupported operations: {unsupported}")
        if self.resize_scale <= 0:
            raise ValueError("resize_scale must be positive")
        if not 0 <= self.threshold <= 255:
            raise ValueError("threshold must be between 0 and 255")
        if not self.output_suffix.startswith("."):
            raise ValueError("output_suffix must start with '.'")


@dataclass(frozen=True, slots=True)
class ImagePreprocessTool:
    """Execute configurable image preprocessing with Pillow."""

    config: ImagePreprocessConfig
    tool_name: str = "image_preprocess"

    @property
    def spec(self) -> ToolSpec:
        return {
            "type": "function",
            "name": self.tool_name,
            "description": "Apply configurable preprocessing operations to a local image.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_uri": {"type": "string"},
                    "operations": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(ALL_IMAGE_OPERATIONS)},
                    },
                    "operation_repeat": {"type": "integer", "minimum": 1},
                },
                "required": ["input_uri"],
                "additionalProperties": False,
            },
            "strict": True,
        }

    def __call__(self, tool_call: ToolCall) -> ToolExecution:
        input_uri, operations, operation_repeat = self._parse_arguments(tool_call.arguments)
        input_path = resolve_local_path(input_uri, self.config.local_root)
        output_path = self._output_path(tool_call)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(input_path) as source:
            input_profile = ImageProfile(source.width, source.height, source.mode)
            image = source.copy()

        image = self._apply_operations(image, operations, operation_repeat)
        output_profile = ImageProfile(image.width, image.height, image.mode)
        image.save(output_path)

        operation_count = len(operations) * operation_repeat
        return ToolExecution(
            output_uri=output_path.resolve().as_uri(),
            metadata={
                "backend": "pillow",
                "operations": list(operations),
                "operation_repeat": operation_repeat,
                "operation_count": operation_count,
                "input_width": input_profile.width,
                "input_height": input_profile.height,
                "input_pixels": input_profile.pixel_count,
                "input_mode": input_profile.mode,
                "output_width": output_profile.width,
                "output_height": output_profile.height,
                "output_pixels": output_profile.pixel_count,
                "output_mode": output_profile.mode,
                "estimated_work_units": input_profile.pixel_count * operation_count,
            },
        )

    def _apply_operations(
        self,
        image: Image.Image,
        operations: tuple[ImageOperation, ...],
        operation_repeat: int,
    ) -> Image.Image:
        for _ in range(operation_repeat):
            for operation in operations:
                if operation == "grayscale":
                    image = image.convert("L")
                elif operation == "resize":
                    size = (
                        max(1, int(image.width * self.config.resize_scale)),
                        max(1, int(image.height * self.config.resize_scale)),
                    )
                    image = image.resize(size, Image.Resampling.LANCZOS)
                elif operation == "blur":
                    image = image.filter(ImageFilter.GaussianBlur(radius=1))
                elif operation == "threshold":
                    image = image.convert("L").point(
                        lambda pixel: 255 if pixel >= self.config.threshold else 0
                    )
                elif operation == "edge_detect":
                    image = image.convert("L").filter(ImageFilter.FIND_EDGES)
        return image

    def _parse_arguments(
        self,
        arguments: dict[str, object],
    ) -> tuple[str, tuple[ImageOperation, ...], int]:
        allowed_arguments = {"input_uri", "operations", "operation_repeat"}
        unsupported = sorted(set(arguments) - allowed_arguments)
        if unsupported:
            raise ValueError(f"unsupported arguments: {unsupported}")

        input_uri = arguments.get("input_uri")
        if not isinstance(input_uri, str) or not input_uri.strip():
            raise ValueError("input_uri must be a non-empty string")

        raw_operations = arguments.get("operations")
        if raw_operations is None:
            operations = self.config.operations
        else:
            if not isinstance(raw_operations, list) or not raw_operations:
                raise ValueError("operations must be a non-empty array")
            if any(not isinstance(operation, str) for operation in raw_operations):
                raise ValueError("operations must contain only strings")
            unsupported_operations = sorted(set(raw_operations) - set(ALL_IMAGE_OPERATIONS))
            if unsupported_operations:
                raise ValueError(f"unsupported operations: {unsupported_operations}")
            operations = tuple(raw_operations)

        operation_repeat = arguments.get("operation_repeat", self.config.operation_repeat)
        if (
            isinstance(operation_repeat, bool)
            or not isinstance(operation_repeat, int)
            or operation_repeat < 1
        ):
            raise ValueError("operation_repeat must be a positive integer")
        return input_uri, operations, operation_repeat

    def _output_path(self, tool_call: ToolCall) -> Path:
        safe_id = tool_call.tool_call_id.replace("/", "_")
        return self.config.output_dir / f"{safe_id}{self.config.output_suffix}"


def resolve_local_path(uri: str, local_root: Path) -> Path:
    """Resolve file, local, or plain path URIs to a local path."""

    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    if parsed.scheme == "local":
        raw_path = unquote(f"{parsed.netloc}{parsed.path}")
        return local_root / raw_path.lstrip("/")
    if parsed.scheme:
        raise ValueError(f"unsupported input URI scheme {parsed.scheme!r}")
    path = Path(uri)
    return path if path.is_absolute() else local_root / path
