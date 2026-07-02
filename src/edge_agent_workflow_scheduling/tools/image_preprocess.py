"""Image preprocessing tool implementation."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Literal, Self
from urllib.parse import unquote, urlparse

from edge_agent_workflow_scheduling.common import ToolCall
from edge_agent_workflow_scheduling.tools.base import ToolExecution, ToolSpec

ImageOperation = Literal["grayscale", "resize", "blur", "threshold", "edge_detect"]

_ALL_IMAGE_OPERATIONS: tuple[ImageOperation, ...] = (
    "grayscale",
    "resize",
    "blur",
    "threshold",
    "edge_detect",
)

_DEFAULT_OPERATIONS: tuple[ImageOperation, ...] = (
    "grayscale",
    "resize",
    "blur",
    "threshold",
)


@dataclass(frozen=True, slots=True)
class ImageBuffer:
    """Small grayscale image container used by the standard-library fallback."""

    width: int
    height: int
    pixels: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.width < 1:
            msg = "width must be positive"
            raise ValueError(msg)
        if self.height < 1:
            msg = "height must be positive"
            raise ValueError(msg)
        if len(self.pixels) != self.width * self.height:
            msg = "pixel count does not match image dimensions"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ImageProfile:
    """Image dimensions and mode used for tool workload metadata."""

    width: int
    height: int
    mode: str

    @property
    def pixel_count(self) -> int:
        return self.width * self.height

    @classmethod
    def from_buffer(cls, image: ImageBuffer) -> Self:
        return cls(width=image.width, height=image.height, mode="L")


@dataclass(frozen=True, slots=True)
class ImagePreprocessConfig:
    """Configuration for the image preprocessing tool."""

    output_dir: Path
    local_root: Path = Path(".")
    operations: tuple[ImageOperation, ...] = _DEFAULT_OPERATIONS
    operation_repeat: int = 1
    resize_scale: float = 0.5
    threshold: int = 128
    output_suffix: str = ".pgm"

    def __post_init__(self) -> None:
        if self.operation_repeat < 1:
            msg = "operation_repeat must be at least 1"
            raise ValueError(msg)
        if not self.operations:
            msg = "operations must be non-empty"
            raise ValueError(msg)
        unsupported_operations = sorted(set(self.operations) - set(_ALL_IMAGE_OPERATIONS))
        if unsupported_operations:
            msg = f"unsupported operations: {unsupported_operations}"
            raise ValueError(msg)
        if self.resize_scale <= 0:
            msg = "resize_scale must be positive"
            raise ValueError(msg)
        if not 0 <= self.threshold <= 255:
            msg = "threshold must be between 0 and 255"
            raise ValueError(msg)
        if not self.output_suffix.startswith("."):
            msg = "output_suffix must start with '.'"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ImagePreprocessTool:
    """Configurable image preprocessing executor.

    The tool uses Pillow when available. In minimal environments without Pillow,
    it falls back to a standard-library Netpbm reader/writer for PPM/PGM images.
    """

    config: ImagePreprocessConfig
    tool_type: str = "image_preprocess"

    @property
    def spec(self) -> ToolSpec:
        return {
            "type": "function",
            "name": self.tool_type,
            "description": (
                "Reads a local image, applies configurable preprocessing operations, "
                "and writes a processed image for downstream OCR or document parsing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "input_uri": {
                        "type": "string",
                        "description": (
                            "Local image URI using file://, local://, or a filesystem path."
                        ),
                    },
                    "operations": {
                        "type": "array",
                        "description": "Ordered preprocessing operations to apply.",
                        "items": {
                            "type": "string",
                            "enum": list(_ALL_IMAGE_OPERATIONS),
                        },
                    },
                    "operation_repeat": {
                        "type": "integer",
                        "description": "Number of times to repeat the configured operations.",
                        "enum": [1, 5, 10, 20],
                    },
                    "resize_scale": {
                        "type": "number",
                        "description": "Scale factor used by the resize operation.",
                        "exclusiveMinimum": 0,
                    },
                    "threshold": {
                        "type": "integer",
                        "description": "Threshold value used by the threshold operation.",
                        "minimum": 0,
                        "maximum": 255,
                    },
                },
                "required": ["input_uri"],
                "additionalProperties": False,
            },
            "strict": True,
        }

    def __call__(self, tool_call: ToolCall) -> ToolExecution:
        input_path = resolve_local_path(tool_call.input_uri, self.config.local_root)
        output_path = self._build_output_path(tool_call)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        input_image_profile: ImageProfile
        output_image_profile: ImageProfile

        try:
            input_image_profile, output_image_profile = self._run_with_pillow(
                input_path,
                output_path,
            )
        except ModuleNotFoundError:
            image = read_netpbm(input_path)
            input_image_profile = ImageProfile.from_buffer(image)
            processed_image = self._preprocess_fallback(image)
            output_image_profile = ImageProfile.from_buffer(processed_image)
            write_pgm(output_path, processed_image)

        return ToolExecution(
            output_uri=output_path.resolve().as_uri(),
            metadata=self.build_workload_metadata(
                input_profile=input_image_profile,
                output_profile=output_image_profile,
            ),
        )

    def estimate_work_units(self, image_width: int, image_height: int) -> int:
        """Return a rough deterministic work estimate for scheduling experiments."""

        operation_count = len(self.config.operations) * self.config.operation_repeat
        return image_width * image_height * operation_count

    def build_workload_metadata(
        self,
        *,
        input_profile: ImageProfile,
        output_profile: ImageProfile,
    ) -> dict[str, object]:
        operation_count = len(self.config.operations) * self.config.operation_repeat
        return {
            "tool_type": self.tool_type,
            "description": self.spec["description"],
            "operations": list(self.config.operations),
            "operation_repeat": self.config.operation_repeat,
            "operation_count": operation_count,
            "input_width": input_profile.width,
            "input_height": input_profile.height,
            "input_pixels": input_profile.pixel_count,
            "input_mode": input_profile.mode,
            "output_width": output_profile.width,
            "output_height": output_profile.height,
            "output_pixels": output_profile.pixel_count,
            "output_mode": output_profile.mode,
            "estimated_work_units": self.estimate_work_units(
                input_profile.width,
                input_profile.height,
            ),
        }

    def _run_with_pillow(
        self,
        input_path: Path,
        output_path: Path,
    ) -> tuple[ImageProfile, ImageProfile]:
        image_module, image_filter_module = _load_pillow_modules()

        source_image = image_module.open(input_path)
        input_profile = ImageProfile(
            width=source_image.width,
            height=source_image.height,
            mode=source_image.mode,
        )
        image = source_image.convert("L")
        for _ in range(self.config.operation_repeat):
            for operation in self.config.operations:
                if operation == "grayscale":
                    image = image.convert("L")
                elif operation == "resize":
                    width = max(1, int(image.width * self.config.resize_scale))
                    height = max(1, int(image.height * self.config.resize_scale))
                    image = image.resize((width, height))
                elif operation == "blur":
                    image = image.filter(image_filter_module.GaussianBlur(radius=1))
                elif operation == "threshold":
                    image = image.point(lambda pixel: 255 if pixel >= self.config.threshold else 0)
                elif operation == "edge_detect":
                    image = image.filter(image_filter_module.FIND_EDGES)
                else:
                    _raise_unknown_operation(operation)

        image.save(output_path)
        output_profile = ImageProfile(width=image.width, height=image.height, mode=image.mode)
        return input_profile, output_profile

    def _preprocess_fallback(self, image: ImageBuffer) -> ImageBuffer:
        processed_image = image
        for _ in range(self.config.operation_repeat):
            for operation in self.config.operations:
                if operation == "grayscale":
                    continue
                if operation == "resize":
                    processed_image = resize_nearest(processed_image, self.config.resize_scale)
                elif operation == "blur":
                    processed_image = blur_3x3(processed_image)
                elif operation == "threshold":
                    processed_image = threshold_image(processed_image, self.config.threshold)
                elif operation == "edge_detect":
                    processed_image = edge_detect(processed_image)
                else:
                    _raise_unknown_operation(operation)
        return processed_image

    def _build_output_path(self, tool_call: ToolCall) -> Path:
        safe_tool_call_id = tool_call.tool_call_id.replace("/", "_")
        return self.config.output_dir / f"{safe_tool_call_id}{self.config.output_suffix}"


def resolve_local_path(uri: str, local_root: Path) -> Path:
    """Resolve file, local, or plain path URIs into filesystem paths."""

    parsed_uri = urlparse(uri)
    if parsed_uri.scheme == "file":
        return Path(unquote(parsed_uri.path))

    if parsed_uri.scheme == "local":
        raw_path = unquote(f"{parsed_uri.netloc}{parsed_uri.path}")
        if parsed_uri.netloc:
            return local_root / raw_path.lstrip("/")
        return Path(raw_path) if raw_path.startswith("/") else local_root / raw_path

    if parsed_uri.scheme:
        msg = f"unsupported input URI scheme {parsed_uri.scheme!r}"
        raise ValueError(msg)

    path = Path(uri)
    return path if path.is_absolute() else local_root / path


def _load_pillow_modules():
    return import_module("PIL.Image"), import_module("PIL.ImageFilter")


def read_netpbm(path: Path) -> ImageBuffer:
    """Read a binary PPM/PGM file into a grayscale image buffer."""

    with path.open("rb") as image_file:
        magic = _read_token(image_file)
        if magic not in {b"P5", b"P6"}:
            msg = "Pillow is required for non-Netpbm images"
            raise ValueError(msg)

        width = int(_read_token(image_file))
        height = int(_read_token(image_file))
        max_value = int(_read_token(image_file))
        if max_value <= 0 or max_value > 255:
            msg = "only 8-bit Netpbm images are supported"
            raise ValueError(msg)

        if magic == b"P5":
            raw_pixels = image_file.read(width * height)
            if len(raw_pixels) != width * height:
                msg = "truncated PGM image data"
                raise ValueError(msg)
            pixels = tuple(_scale_pixel(value, max_value) for value in raw_pixels)
            return ImageBuffer(width=width, height=height, pixels=pixels)

        raw_pixels = image_file.read(width * height * 3)
        if len(raw_pixels) != width * height * 3:
            msg = "truncated PPM image data"
            raise ValueError(msg)

        pixels = []
        for offset in range(0, len(raw_pixels), 3):
            red = _scale_pixel(raw_pixels[offset], max_value)
            green = _scale_pixel(raw_pixels[offset + 1], max_value)
            blue = _scale_pixel(raw_pixels[offset + 2], max_value)
            pixels.append((299 * red + 587 * green + 114 * blue) // 1000)
        return ImageBuffer(width=width, height=height, pixels=tuple(pixels))


def write_pgm(path: Path, image: ImageBuffer) -> None:
    """Write a grayscale image buffer as binary PGM."""

    header = f"P5\n{image.width} {image.height}\n255\n".encode("ascii")
    path.write_bytes(header + bytes(_clamp_pixel(pixel) for pixel in image.pixels))


def resize_nearest(image: ImageBuffer, scale: float) -> ImageBuffer:
    new_width = max(1, int(image.width * scale))
    new_height = max(1, int(image.height * scale))
    pixels = []
    for y_pos in range(new_height):
        source_y = min(image.height - 1, int(y_pos / scale))
        for x_pos in range(new_width):
            source_x = min(image.width - 1, int(x_pos / scale))
            pixels.append(image.pixels[source_y * image.width + source_x])
    return ImageBuffer(width=new_width, height=new_height, pixels=tuple(pixels))


def blur_3x3(image: ImageBuffer) -> ImageBuffer:
    pixels = []
    for y_pos in range(image.height):
        for x_pos in range(image.width):
            total = 0
            count = 0
            for dy in (-1, 0, 1):
                source_y = y_pos + dy
                if source_y < 0 or source_y >= image.height:
                    continue
                for dx in (-1, 0, 1):
                    source_x = x_pos + dx
                    if source_x < 0 or source_x >= image.width:
                        continue
                    total += image.pixels[source_y * image.width + source_x]
                    count += 1
            pixels.append(total // count)
    return ImageBuffer(width=image.width, height=image.height, pixels=tuple(pixels))


def threshold_image(image: ImageBuffer, threshold: int) -> ImageBuffer:
    pixels = tuple(255 if pixel >= threshold else 0 for pixel in image.pixels)
    return ImageBuffer(width=image.width, height=image.height, pixels=pixels)


def edge_detect(image: ImageBuffer) -> ImageBuffer:
    pixels = []
    for y_pos in range(image.height):
        for x_pos in range(image.width):
            center = image.pixels[y_pos * image.width + x_pos]
            right = image.pixels[y_pos * image.width + min(image.width - 1, x_pos + 1)]
            down = image.pixels[min(image.height - 1, y_pos + 1) * image.width + x_pos]
            pixels.append(_clamp_pixel(abs(center - right) + abs(center - down)))
    return ImageBuffer(width=image.width, height=image.height, pixels=tuple(pixels))


def _read_token(image_file) -> bytes:
    token = bytearray()
    while True:
        char = image_file.read(1)
        if not char:
            if token:
                return bytes(token)
            msg = "unexpected end of Netpbm header"
            raise ValueError(msg)
        if char == b"#":
            image_file.readline()
            continue
        if char.isspace():
            if token:
                return bytes(token)
            continue
        token.extend(char)


def _scale_pixel(value: int, max_value: int) -> int:
    if max_value == 255:
        return value
    return round(value * 255 / max_value)


def _clamp_pixel(value: int) -> int:
    return min(255, max(0, value))


def _raise_unknown_operation(operation: str) -> None:
    msg = f"unsupported image operation {operation!r}"
    raise ValueError(msg)
