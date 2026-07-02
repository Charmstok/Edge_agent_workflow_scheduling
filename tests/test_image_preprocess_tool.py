from pathlib import Path

import pytest

from edge_agent_workflow_scheduling.common import ToolCall
from edge_agent_workflow_scheduling.tools import (
    ImagePreprocessConfig,
    ImagePreprocessTool,
    ToolExecution,
    ToolRegistry,
    read_netpbm,
)
from edge_agent_workflow_scheduling.workers import LocalWorker


def test_image_preprocess_tool_reads_input_and_writes_output(tmp_path: Path) -> None:
    input_path = tmp_path / "input.ppm"
    output_dir = tmp_path / "outputs"
    _write_test_ppm(input_path, width=8, height=6)
    tool = ImagePreprocessTool(
        ImagePreprocessConfig(
            output_dir=output_dir,
            operations=("grayscale", "blur", "threshold", "edge_detect"),
            operation_repeat=2,
        )
    )
    tool_call = ToolCall(
        tool_call_id="tc_image_001",
        agent_id="agent_1",
        tool_type="image_preprocess",
        input_uri=input_path.as_uri(),
        input_size_mb=0.01,
        image_count=1,
    )

    execution = tool(tool_call)
    assert isinstance(execution, ToolExecution)

    output_path = Path(execution.output_uri.removeprefix("file://"))
    output_image = read_netpbm(output_path)

    assert output_path.exists()
    assert output_path.name == "tc_image_001.pgm"
    assert output_image.width == 8
    assert output_image.height == 6
    assert execution.metadata["description"] == tool.spec["description"]
    assert execution.metadata["input_width"] == 8
    assert execution.metadata["input_height"] == 6
    assert execution.metadata["input_pixels"] == 48
    assert execution.metadata["operation_count"] == 8
    assert execution.metadata["estimated_work_units"] == 384


def test_image_preprocess_tool_exposes_description_and_configured_operations(
    tmp_path: Path,
) -> None:
    tool = ImagePreprocessTool(
        ImagePreprocessConfig(
            output_dir=tmp_path / "outputs",
            operations=("resize", "threshold"),
            operation_repeat=5,
        )
    )

    spec = tool.spec
    assert spec["type"] == "function"
    assert spec["name"] == "image_preprocess"
    assert "preprocessing operations" in spec["description"]
    assert spec["strict"] is True
    assert spec["parameters"] == {
        "type": "object",
        "properties": {
            "input_uri": {
                "type": "string",
                "description": "Local image URI using file://, local://, or a filesystem path.",
            },
            "operations": {
                "type": "array",
                "description": "Ordered preprocessing operations to apply.",
                "items": {
                    "type": "string",
                    "enum": ["grayscale", "resize", "blur", "threshold", "edge_detect"],
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
    }


def test_image_preprocess_tool_supports_local_uri_resolution(tmp_path: Path) -> None:
    local_root = tmp_path / "local_root"
    input_path = local_root / "data" / "image.ppm"
    output_dir = tmp_path / "outputs"
    input_path.parent.mkdir(parents=True)
    _write_test_ppm(input_path, width=4, height=4)
    tool = ImagePreprocessTool(
        ImagePreprocessConfig(
            output_dir=output_dir,
            local_root=local_root,
            operations=("threshold",),
        )
    )
    tool_call = ToolCall(
        tool_call_id="tc_local_uri",
        agent_id="agent_1",
        tool_type="image_preprocess",
        input_uri="local://data/image.ppm",
        input_size_mb=0.01,
        image_count=1,
    )

    execution = tool(tool_call)

    assert execution.output_uri is not None
    assert Path(execution.output_uri.removeprefix("file://")).exists()


def test_image_preprocess_tool_integrates_with_local_worker(tmp_path: Path) -> None:
    input_path = tmp_path / "input.ppm"
    output_dir = tmp_path / "outputs"
    _write_test_ppm(input_path, width=16, height=16)
    registry = ToolRegistry()
    registry.register(
        ImagePreprocessTool(
            ImagePreprocessConfig(
                output_dir=output_dir,
                operations=("blur", "threshold", "edge_detect"),
                operation_repeat=3,
            )
        )
    )
    worker = LocalWorker(
        worker_id="worker_local_1",
        supported_tools=registry.supported_tools(),
        tool_executors=registry.as_executor_mapping(),
    )
    tool_call = ToolCall(
        tool_call_id="tc_worker_image",
        agent_id="agent_1",
        tool_type="image_preprocess",
        input_uri=input_path.as_uri(),
        input_size_mb=0.01,
        image_count=1,
    )

    result = worker.run_tool(tool_call)

    assert result.success is True
    assert result.output_uri is not None
    assert Path(result.output_uri.removeprefix("file://")).exists()
    assert result.execution_time_sec > 0
    assert result.metadata["description"] == registry.get("image_preprocess").spec["description"]
    assert result.metadata["operation_repeat"] == 3
    assert result.metadata["operation_count"] == 9
    assert result.metadata["input_pixels"] == 256
    assert result.metadata["estimated_work_units"] == 2304


def test_image_preprocess_work_units_scale_with_resolution_and_repeat(tmp_path: Path) -> None:
    tool_repeat_1 = ImagePreprocessTool(
        ImagePreprocessConfig(
            output_dir=tmp_path / "outputs_1",
            operations=("blur", "threshold"),
            operation_repeat=1,
        )
    )
    tool_repeat_5 = ImagePreprocessTool(
        ImagePreprocessConfig(
            output_dir=tmp_path / "outputs_5",
            operations=("blur", "threshold"),
            operation_repeat=5,
        )
    )

    assert tool_repeat_1.estimate_work_units(20, 20) == 800
    assert tool_repeat_1.estimate_work_units(40, 40) == 3200
    assert tool_repeat_5.estimate_work_units(20, 20) == 4000


def test_image_preprocess_rejects_unsupported_uri_scheme(tmp_path: Path) -> None:
    tool = ImagePreprocessTool(ImagePreprocessConfig(output_dir=tmp_path / "outputs"))
    tool_call = ToolCall(
        tool_call_id="tc_bad_uri",
        agent_id="agent_1",
        tool_type="image_preprocess",
        input_uri="s3://bucket/image.png",
        input_size_mb=1.0,
        image_count=1,
    )

    with pytest.raises(ValueError, match="unsupported input URI scheme"):
        tool(tool_call)


def test_image_preprocess_config_validates_values(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="operation_repeat must be at least 1"):
        ImagePreprocessConfig(output_dir=tmp_path, operation_repeat=0)

    with pytest.raises(ValueError, match="operations must be non-empty"):
        ImagePreprocessConfig(output_dir=tmp_path, operations=())

    with pytest.raises(ValueError, match="resize_scale must be positive"):
        ImagePreprocessConfig(output_dir=tmp_path, resize_scale=0)

    with pytest.raises(ValueError, match="threshold must be between 0 and 255"):
        ImagePreprocessConfig(output_dir=tmp_path, threshold=300)


def test_tool_registry_registers_and_rejects_duplicate_tools(tmp_path: Path) -> None:
    registry = ToolRegistry()
    tool = ImagePreprocessTool(ImagePreprocessConfig(output_dir=tmp_path / "outputs"))

    registry.register(tool)

    assert registry.supported_tools() == ["image_preprocess"]
    assert registry.get("image_preprocess") is tool
    assert registry.tools() == [tool.spec]
    assert registry.specs() == [tool.spec]
    assert registry.as_executor_mapping() == {"image_preprocess": tool}
    with pytest.raises(ValueError, match="already registered"):
        registry.register(tool)
    with pytest.raises(KeyError, match="is not registered"):
        registry.get("ocr")


def _write_test_ppm(path: Path, *, width: int, height: int) -> None:
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    pixels = bytearray()
    for y_pos in range(height):
        for x_pos in range(width):
            pixels.extend(
                (
                    (x_pos * 17) % 256,
                    (y_pos * 23) % 256,
                    ((x_pos + y_pos) * 13) % 256,
                )
            )
    path.write_bytes(header + bytes(pixels))
