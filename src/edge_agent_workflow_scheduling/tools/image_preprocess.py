"""Image preprocessing tool implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlparse

from edge_agent_workflow_scheduling.common import ToolCall

ImageOperation = Literal["grayscale", "resize", "blur", "threshold", "edge_detect"]

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

    def __call__(self, tool_call: ToolCall) -> str:
        input_path = resolve_local_path(tool_call.input_uri, self.config.local_root)
        output_path = self._build_output_path(tool_call)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._run_with_pillow(input_path, output_path)
        except ModuleNotFoundError:
            image = read_netpbm(input_path)
            processed_image = self._preprocess_fallback(image)
            write_pgm(output_path, processed_image)

        return output_path.resolve().as_uri()

    def estimate_work_units(self, image_width: int, image_height: int) -> int:
        """Return a rough deterministic work estimate for scheduling experiments."""

        operation_count = len(self.config.operations) * self.config.operation_repeat
        return image_width * image_height * operation_count

    def _run_with_pillow(self, input_path: Path, output_path: Path) -> None:
        from PIL import Image, ImageFilter

        image = Image.open(input_path).convert("L")
        for _ in range(self.config.operation_repeat):
            for operation in self.config.operations:
                if operation == "grayscale":
                    image = image.convert("L")
                elif operation == "resize":
                    width = max(1, int(image.width * self.config.resize_scale))
                    height = max(1, int(image.height * self.config.resize_scale))
                    image = image.resize((width, height))
                elif operation == "blur":
                    image = image.filter(ImageFilter.GaussianBlur(radius=1))
                elif operation == "threshold":
                    image = image.point(lambda pixel: 255 if pixel >= self.config.threshold else 0)
                elif operation == "edge_detect":
                    image = image.filter(ImageFilter.FIND_EDGES)
                else:
                    _raise_unknown_operation(operation)

        image.save(output_path)

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
