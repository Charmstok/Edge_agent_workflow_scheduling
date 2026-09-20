"""Controlled real-Tool sampling with queue, resource, and distribution measurements."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from statistics import fmean, median, pstdev
from threading import Event, Lock, Thread
from time import perf_counter
from typing import Any, Literal, Self
from uuid import uuid4

from PIL import __version__ as pillow_version

from edge_agent_workflow_scheduling.common import (
    CallStatus,
    ScheduleDecision,
    ToolCall,
    ToolResult,
)
from edge_agent_workflow_scheduling.executors import LocalToolExecutor
from edge_agent_workflow_scheduling.profiler.trace import build_tool_trace_record
from edge_agent_workflow_scheduling.resources import ToolReplicaProfile
from edge_agent_workflow_scheduling.tools import (
    ALL_IMAGE_OPERATIONS,
    DocumentToolConfig,
    ImagePreprocessConfig,
    ImagePreprocessTool,
    OCRConfig,
    OCRTool,
    PDFParseTool,
    ToolRegistry,
)
from edge_agent_workflow_scheduling.workers import LocalWorker

try:
    import psutil
except ModuleNotFoundError:  # pragma: no cover - exercised through dependency injection
    psutil = None

SamplingPhase = Literal["cold_start", "warmup", "measurement"]
SUPPORTED_TOOLS = ("image_preprocess", "ocr", "pdf_parse")
SUPPORTED_SCALES = ("small", "medium", "large")


def _positive_number(value: Any, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be finite and positive")


def _integer(value: Any, name: str, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _unique_names(values: Any, name: str, allowed: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(values, list | tuple) or not values:
        raise ValueError(f"{name} must be a non-empty sequence")
    if any(not isinstance(value, str) or value not in allowed for value in values):
        raise ValueError(f"{name} contains an unsupported value")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(values)


@dataclass(frozen=True, slots=True)
class ToolSamplingConfig:
    """Versioned matrix and phase sizes for one local Tool sampling experiment."""

    sampling_id: str
    tools: tuple[str, ...]
    scales: tuple[str, ...]
    concurrency_levels: tuple[int, ...]
    cold_start_runs: int
    warmup_runs_per_worker: int
    measurement_repetitions_per_worker: int
    timeout_sec: float
    resource_sample_interval_sec: float
    input_templates: dict[str, str]
    image_preprocess: dict[str, Any]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.sampling_id, str) or not self.sampling_id.strip():
            raise ValueError("sampling_id must be non-empty")
        object.__setattr__(self, "tools", _unique_names(self.tools, "tools", SUPPORTED_TOOLS))
        object.__setattr__(self, "scales", _unique_names(self.scales, "scales", SUPPORTED_SCALES))
        _integer(self.schema_version, "schema_version", 1)
        if self.schema_version != 1:
            raise ValueError("unsupported schema_version")
        _integer(self.cold_start_runs, "cold_start_runs", 1)
        _integer(self.warmup_runs_per_worker, "warmup_runs_per_worker")
        _integer(
            self.measurement_repetitions_per_worker,
            "measurement_repetitions_per_worker",
            1,
        )
        if not self.concurrency_levels:
            raise ValueError("concurrency_levels must not be empty")
        for value in self.concurrency_levels:
            _integer(value, "concurrency level", 1)
        if len(set(self.concurrency_levels)) != len(self.concurrency_levels):
            raise ValueError("concurrency_levels must not contain duplicates")
        _positive_number(self.timeout_sec, "timeout_sec")
        _positive_number(self.resource_sample_interval_sec, "resource_sample_interval_sec")
        if set(self.input_templates) != set(self.tools):
            raise ValueError("input_templates must define exactly the requested tools")
        for tool_name, template in self.input_templates.items():
            if not isinstance(template, str) or "{scale}" not in template:
                raise ValueError(f"input template for {tool_name} must contain {{scale}}")
        operations = self.image_preprocess.get("operations")
        if not isinstance(operations, list) or not operations:
            raise ValueError("image_preprocess.operations must be a non-empty list")
        if set(operations) - set(ALL_IMAGE_OPERATIONS):
            raise ValueError("image_preprocess.operations contains an unsupported operation")
        _integer(
            self.image_preprocess.get("operation_repeat"),
            "image_preprocess.operation_repeat",
            1,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        value = deepcopy(data)
        value["tools"] = tuple(value["tools"])
        value["scales"] = tuple(value["scales"])
        value["concurrency_levels"] = tuple(value["concurrency_levels"])
        return cls(**value)

    @classmethod
    def from_json(cls, path: str | Path) -> Self:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self), allow_nan=False))


@dataclass(slots=True)
class _QueueState:
    submitted: int = 0
    started: int = 0
    finished: int = 0
    running: int = 0
    lock: Lock = field(default_factory=Lock)


class ProcessTreeSampler:
    """Poll CPU and RSS for this profiler process and its active Tool subprocesses."""

    def __init__(self, interval_sec: float, *, psutil_module: Any = psutil) -> None:
        _positive_number(interval_sec, "interval_sec")
        self.interval_sec = interval_sec
        self.psutil = psutil_module
        self.samples: list[dict[str, Any]] = []
        self._stop = Event()
        self._thread: Thread | None = None
        self._started_at = 0.0
        self._last_at = 0.0
        self._last_cpu_times: dict[int, float] = {}

    def start(self) -> None:
        self._started_at = perf_counter()
        self._last_at = self._started_at
        if self.psutil is None:
            return
        self.psutil.cpu_percent(interval=None)
        self._collect()
        self._thread = Thread(target=self._run, name="tool-resource-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        if self.psutil is None:
            return unavailable_resource_summary(
                "psutil is not installed; install requirements-tools.txt"
            )
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_sec * 4))
        self._collect()
        return summarize_resource_samples(self.samples, self.interval_sec)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_sec):
            self._collect()

    def _collect(self) -> None:
        now = perf_counter()
        try:
            root = self.psutil.Process(os.getpid())
            try:
                children = root.children(recursive=True)
            except (self.psutil.Error, OSError):
                children = []
            processes = [root, *children]
            current_cpu_times: dict[int, float] = {}
            rss_bytes = 0
            thread_count = 0
            process_count = 0
            for process in processes:
                observed_process = False
                try:
                    cpu = process.cpu_times()
                    current_cpu_times[process.pid] = cpu.user + cpu.system
                except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
                    pass
                else:
                    observed_process = True
                try:
                    rss_bytes += process.memory_info().rss
                    observed_process = True
                except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
                    pass
                try:
                    thread_count += process.num_threads()
                    observed_process = True
                except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
                    pass
                process_count += observed_process
            elapsed = now - self._last_at
            cpu_delta = sum(
                max(0.0, value - self._last_cpu_times[pid])
                for pid, value in current_cpu_times.items()
                if pid in self._last_cpu_times
            )
            logical_cpus = self.psutil.cpu_count(logical=True) or 1
            process_cpu_percent = (
                100.0 * cpu_delta / elapsed / logical_cpus if elapsed > 0 else None
            )
            try:
                system_cpu_percent = self.psutil.cpu_percent(interval=None)
            except (self.psutil.Error, OSError):
                system_cpu_percent = None
            self.samples.append(
                {
                    "offset_sec": now - self._started_at,
                    "process_tree_cpu_percent": process_cpu_percent,
                    "system_cpu_percent": system_cpu_percent,
                    "process_tree_rss_bytes": rss_bytes if process_count else None,
                    "process_count": process_count,
                    "thread_count": thread_count,
                }
            )
            self._last_cpu_times = current_cpu_times
            self._last_at = now
        except (self.psutil.Error, OSError) as exc:
            self.samples.append({"offset_sec": now - self._started_at, "sampling_error": str(exc)})


def unavailable_resource_summary(reason: str) -> dict[str, Any]:
    return {
        "sampling_method": "process_tree_polling",
        "cpu": {"status": "unavailable", "reason": reason},
        "memory": {"status": "unavailable", "reason": reason},
        "gpu_utilization": {
            "status": "unavailable",
            "reason": "GPU utilization is not sampled by this prototype",
        },
    }


def summarize_resource_samples(
    samples: list[dict[str, Any]], interval_sec: float
) -> dict[str, Any]:
    valid = [sample for sample in samples if "sampling_error" not in sample]
    cpu_values = [
        sample["process_tree_cpu_percent"]
        for sample in valid
        if sample.get("process_tree_cpu_percent") is not None
    ]
    system_values = [sample["system_cpu_percent"] for sample in valid]
    rss_values = [sample["process_tree_rss_bytes"] for sample in valid]
    cpu_status = "available" if cpu_values else "insufficient_samples"
    memory_status = "available" if rss_values else "unavailable"
    return {
        "sampling_method": "psutil process-tree polling",
        "sample_interval_sec": interval_sec,
        "sample_count": len(samples),
        "sampling_error_count": len(samples) - len(valid),
        "cpu": {
            "status": cpu_status,
            "process_tree_percent_mean": fmean(cpu_values) if cpu_values else None,
            "process_tree_percent_max": max(cpu_values) if cpu_values else None,
            "system_percent_mean": fmean(system_values) if system_values else None,
            "normalization": "100% equals all logical CPUs; short-lived children may be missed",
        },
        "memory": {
            "status": memory_status,
            "process_tree_rss_peak_bytes": max(rss_values) if rss_values else None,
            "aggregation": "sum of sampled RSS for profiler process and live descendants",
        },
        "process_count_peak": max((sample["process_count"] for sample in valid), default=None),
        "thread_count_peak": max((sample["thread_count"] for sample in valid), default=None),
        "gpu_utilization": {
            "status": "unavailable",
            "reason": "GPU inventory is recorded, but utilization is not sampled",
        },
    }


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between 0 and 1")
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize_measurements(records: list[dict[str, Any]], wall_time_sec: float) -> dict[str, Any]:
    if not records:
        raise ValueError("measurement summary requires at least one record")
    _positive_number(wall_time_sec, "wall_time_sec")
    queue_times = [record["result"]["queue_wait_time_sec"] for record in records]
    execution_times = [record["result"]["execution_time_sec"] for record in records]
    total_times = [
        record["result"]["queue_wait_time_sec"]
        + record["result"]["input_transfer_time_sec"]
        + record["result"]["execution_time_sec"]
        + record["result"]["output_transfer_time_sec"]
        for record in records
    ]
    success_count = sum(record["result"]["success"] for record in records)
    failure_codes: dict[str, int] = {}
    for record in records:
        code = record["result"].get("error_code")
        if code:
            failure_codes[code] = failure_codes.get(code, 0) + 1

    def distribution(values: list[float]) -> dict[str, float]:
        return {
            "min": min(values),
            "mean": fmean(values),
            "median": median(values),
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
            "max": max(values),
            "population_stddev": pstdev(values),
        }

    return {
        "sample_count": len(records),
        "included_count": len(records),
        "excluded_count": 0,
        "exclusion_rule": "none; all terminal measurement calls are retained",
        "success_count": success_count,
        "failure_count": len(records) - success_count,
        "timeout_count": failure_codes.get("timeout", 0),
        "success_rate": success_count / len(records),
        "failure_codes": failure_codes,
        "phase_wall_time_sec": wall_time_sec,
        "throughput_calls_per_sec": len(records) / wall_time_sec,
        "queue_wait_time_sec_total": sum(queue_times),
        "execution_time_sec_total": sum(execution_times),
        "total_latency_sec_total": sum(total_times),
        "queue_wait_time_sec": distribution(queue_times),
        "execution_time_sec": distribution(execution_times),
        "total_latency_sec": distribution(total_times),
    }


def collect_host_inventory(*, psutil_module: Any = psutil) -> dict[str, Any]:
    """Return a privacy-limited host description and explicit metric availability."""

    hardware = _mac_hardware_inventory()
    total_memory = None
    physical_cpus = None
    if psutil_module is not None:
        try:
            total_memory = psutil_module.virtual_memory().total
            physical_cpus = psutil_module.cpu_count(logical=False)
        except (psutil_module.Error, OSError):
            pass
    return {
        "captured_at": datetime.now(UTC).isoformat(),
        "hostname_hash": hashlib.sha256(platform.node().encode()).hexdigest(),
        "os": platform.platform(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "machine_name": hardware.get("machine_name"),
        "machine_model": hardware.get("machine_model"),
        "model_number": hardware.get("model_number"),
        "cpu_or_chip": hardware.get("chip_type") or platform.processor() or None,
        "logical_cpu_count": os.cpu_count(),
        "physical_cpu_count": physical_cpus,
        "memory_bytes": total_memory,
        "reported_physical_memory": hardware.get("physical_memory"),
        "gpu_devices": hardware.get("gpu_devices", []),
        "gpu_utilization": {"status": "unavailable", "reason": "not sampled"},
        "resource_sampler": {
            "psutil_version": getattr(psutil_module, "__version__", None),
            "status": "available" if psutil_module is not None else "unavailable",
        },
    }


def _mac_hardware_inventory() -> dict[str, Any]:
    if platform.system() != "Darwin":
        return {}
    try:
        result = subprocess.run(
            [
                "/usr/sbin/system_profiler",
                "SPHardwareDataType",
                "SPDisplaysDataType",
                "-json",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        data = json.loads(result.stdout)
        hardware = (data.get("SPHardwareDataType") or [{}])[0]
        displays = data.get("SPDisplaysDataType") or []
        return {
            key: hardware.get(key)
            for key in (
                "machine_name",
                "machine_model",
                "model_number",
                "chip_type",
                "physical_memory",
            )
        } | {
            "gpu_devices": [
                {
                    "model": item.get("sppci_model") or item.get("_name"),
                    "core_count": item.get("sppci_cores"),
                    "vram": item.get("spdisplays_vram"),
                }
                for item in displays
            ]
        }
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return {}


def collect_dependency_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {"pillow": pillow_version}
    for executable in ("tesseract",):
        try:
            result = subprocess.run(
                [executable, "--version"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            versions[executable] = result.stdout.splitlines()[0]
        except (OSError, subprocess.SubprocessError):
            versions[executable] = "unavailable"
    try:
        import pypdf

        versions["pypdf"] = pypdf.__version__
    except ModuleNotFoundError:
        versions["pypdf"] = "unavailable"
    versions["psutil"] = getattr(psutil, "__version__", "unavailable")
    return versions


def run_tool_sampling(
    config: ToolSamplingConfig,
    *,
    config_path: str | Path,
    output_root: str | Path,
    experiment_id: str | None = None,
    psutil_module: Any = psutil,
) -> Path:
    """Execute the complete Tool/scale/concurrency matrix and persist raw observations."""

    config_path = Path(config_path).resolve()
    input_root = config_path.parent
    resolved_id = experiment_id or (
        f"{config.sampling_id}-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid4().hex[:8]}"
    )
    output_dir = Path(output_root) / resolved_id
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "experiment_id": resolved_id,
        "mode": "local_real_tool_sampling",
        "started_at": datetime.now(UTC).isoformat(),
        "sampling_config_path": str(config_path),
        "sampling_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "sampling_config": config.to_dict(),
        "host": collect_host_inventory(psutil_module=psutil_module),
        "dependency_versions": collect_dependency_versions(),
        "timing_method": "time.perf_counter monotonic wall clock",
        "transfer_model": "local_same_host_no_transfer",
        "cold_start_definition": (
            "first invocation of a newly constructed Tool/Executor for each configuration; "
            "operating-system caches are not cleared"
        ),
    }
    _write_json(output_dir / "manifest.json", manifest)
    combinations = []
    all_measurements: list[dict[str, Any]] = []
    for tool_name in config.tools:
        for scale in config.scales:
            source = input_root / config.input_templates[tool_name].format(scale=scale)
            if not source.is_file():
                raise FileNotFoundError(f"sampling input not found: {source}")
            for concurrency in config.concurrency_levels:
                combination_id = f"{tool_name}-{scale}-c{concurrency}"
                combination_dir = output_dir / combination_id
                combination_dir.mkdir()
                try:
                    executor = _build_executor(
                        config,
                        tool_name=tool_name,
                        concurrency=concurrency,
                        output_dir=combination_dir / "tool_outputs",
                        input_root=input_root,
                    )
                    tool = executor.worker.tool_registry.get(tool_name)
                    if hasattr(tool, "check_available"):
                        tool.check_available()
                except ModuleNotFoundError as exc:
                    combinations.append(
                        {
                            "combination_id": combination_id,
                            "tool_name": tool_name,
                            "scale": scale,
                            "concurrency": concurrency,
                            "status": "skipped",
                            "reason": str(exc),
                        }
                    )
                    continue
                arguments = _arguments(config, tool_name, source)
                phase_counts = {
                    "cold_start": config.cold_start_runs,
                    "warmup": config.warmup_runs_per_worker * concurrency,
                    "measurement": config.measurement_repetitions_per_worker * concurrency,
                }
                phase_reports = {}
                for phase, count in phase_counts.items():
                    records, resource_samples, resource_summary, wall_time = _run_phase(
                        executor,
                        arguments=arguments,
                        experiment_id=resolved_id,
                        combination_id=combination_id,
                        phase=phase,
                        count=count,
                        concurrency=concurrency,
                        timeout_sec=config.timeout_sec,
                        source=source,
                        interval_sec=config.resource_sample_interval_sec,
                        psutil_module=psutil_module,
                    )
                    _write_jsonl(combination_dir / f"{phase}.jsonl", records)
                    _write_jsonl(
                        combination_dir / f"{phase}_resource_samples.jsonl",
                        resource_samples,
                    )
                    _append_trace_jsonl(combination_dir / "trace.jsonl", records)
                    phase_reports[phase] = {
                        "requested_calls": count,
                        "submitted_calls": len(records),
                        "started_calls": sum(
                            record["started_offset_sec"] >= 0 for record in records
                        ),
                        "finished_calls": sum(
                            record["finished_offset_sec"] >= 0 for record in records
                        ),
                        "wall_time_sec": wall_time,
                        "resource_summary": resource_summary,
                    }
                    if phase == "measurement":
                        phase_reports[phase]["metrics"] = summarize_measurements(records, wall_time)
                        all_measurements.extend(records)
                combinations.append(
                    {
                        "combination_id": combination_id,
                        "tool_name": tool_name,
                        "scale": scale,
                        "concurrency": concurrency,
                        "status": "completed",
                        "source_path": str(source),
                        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "source_bytes": source.stat().st_size,
                        "tool_schema": tool.spec,
                        "resource_profile": executor.profile.to_dict(),
                        "phases": phase_reports,
                    }
                )
                _write_json(combination_dir / "summary.json", combinations[-1])
    summary = {
        "schema_version": 1,
        "experiment_id": resolved_id,
        "completed_at": datetime.now(UTC).isoformat(),
        "combination_count": len(combinations),
        "completed_combination_count": sum(item["status"] == "completed" for item in combinations),
        "skipped_combination_count": sum(item["status"] == "skipped" for item in combinations),
        "measurement_call_count": len(all_measurements),
        "all_terminal_measurements_retained": True,
        "combinations": combinations,
    }
    _write_json(output_dir / "summary.json", summary)
    return output_dir


def _build_executor(
    config: ToolSamplingConfig,
    *,
    tool_name: str,
    concurrency: int,
    output_dir: Path,
    input_root: Path,
) -> LocalToolExecutor:
    if tool_name == "image_preprocess":
        tool = ImagePreprocessTool(
            ImagePreprocessConfig(
                output_dir=output_dir,
                local_root=input_root,
                operations=tuple(config.image_preprocess["operations"]),
                operation_repeat=config.image_preprocess["operation_repeat"],
            )
        )
        version = f"pillow-{pillow_version}"
    elif tool_name == "ocr":
        tool = OCRTool(
            OCRConfig(
                output_dir=output_dir,
                local_root=input_root,
                timeout_sec=config.timeout_sec,
            )
        )
        version = tool.implementation_version
    else:
        tool = PDFParseTool(
            DocumentToolConfig(
                output_dir=output_dir,
                local_root=input_root,
                timeout_sec=config.timeout_sec,
            )
        )
        version = tool.implementation_version
    registry = ToolRegistry()
    registry.register(tool)
    system_name = platform.system().casefold()
    profile = ToolReplicaProfile(
        replica_id=f"{tool_name}-{system_name}-local",
        tool_name=tool_name,
        node_id=f"{system_name}-local",
        platform=system_name,
        implementation_version=version,
        executor_type="local",
        max_concurrency=concurrency,
        metadata={
            "source": "local_real",
            "latency_profile": "uncalibrated",
            "energy": "unavailable",
            "quality": "uncalibrated",
            "platform_details": platform.platform(),
        },
    )
    return LocalToolExecutor(LocalWorker(profile, registry))


def _arguments(config: ToolSamplingConfig, tool_name: str, source: Path) -> dict[str, Any]:
    if tool_name == "image_preprocess":
        return {
            "input_uri": source.resolve().as_uri(),
            "operations": list(config.image_preprocess["operations"]),
            "operation_repeat": config.image_preprocess["operation_repeat"],
        }
    return {"input_uri": source.resolve().as_uri()}


def _run_phase(
    executor: LocalToolExecutor,
    *,
    arguments: dict[str, Any],
    experiment_id: str,
    combination_id: str,
    phase: SamplingPhase,
    count: int,
    concurrency: int,
    timeout_sec: float,
    source: Path,
    interval_sec: float,
    psutil_module: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], float]:
    phase_started = perf_counter()
    phase_started_at = datetime.now(UTC).isoformat()
    sampler = ProcessTreeSampler(interval_sec, psutil_module=psutil_module)
    sampler.start()
    queue = _QueueState()
    futures = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for sequence in range(count):
            call_id = f"{experiment_id}-{combination_id}-{phase}-{sequence:04d}"
            call = ToolCall(
                tool_call_id=call_id,
                call_id=f"function-{call_id}",
                run_id=experiment_id,
                agent_id="tool-profiler",
                tool_name=executor.profile.tool_name,
                arguments=deepcopy(arguments),
                metadata={
                    "task_type": "tool_profile_sampling",
                    "combination_id": combination_id,
                    "phase": phase,
                },
            )
            call.transition_to(CallStatus.QUEUED)
            decision = ScheduleDecision(
                call_id=call_id,
                call_kind="tool",
                selected_target=executor.profile.replica_id,
                policy_name="fixed_profile_target",
                reason="profile one declared local target",
                candidate_target_ids=[executor.profile.replica_id],
                action_mask=[True],
            )
            submitted_at = perf_counter()
            with queue.lock:
                queue.submitted += 1
                queue_depth_after_submit = queue.submitted - queue.started
            futures.append(
                pool.submit(
                    _execute_sample,
                    executor,
                    call,
                    decision,
                    queue,
                    submitted_at,
                    phase_started,
                    phase_started_at,
                    queue_depth_after_submit,
                    sequence,
                    timeout_sec,
                    source,
                    combination_id,
                    phase,
                )
            )
    records = [future.result() for future in futures]
    phase_wall_time = perf_counter() - phase_started
    resource_summary = sampler.stop()
    return records, sampler.samples, resource_summary, phase_wall_time


def _execute_sample(
    executor: LocalToolExecutor,
    call: ToolCall,
    decision: ScheduleDecision,
    queue: _QueueState,
    submitted_at: float,
    phase_started: float,
    phase_started_at: str,
    queue_depth_after_submit: int,
    sequence: int,
    timeout_sec: float,
    source: Path,
    combination_id: str,
    phase: SamplingPhase,
) -> dict[str, Any]:
    started_at = perf_counter()
    with queue.lock:
        queue.started += 1
        queue.running += 1
        queue_depth_at_start = queue.submitted - queue.started
        running_at_start = queue.running
    call.transition_to(CallStatus.RUNNING)
    result = executor.execute(call, timeout_sec=timeout_sec)
    finished_at = perf_counter()
    with queue.lock:
        queue.running -= 1
        queue.finished += 1
        running_at_finish = queue.running
    queue_wait = started_at - submitted_at
    result = replace(
        result,
        queue_wait_time_sec=queue_wait,
        metadata={
            **result.metadata,
            "sampling_phase": phase,
            "sampling_combination_id": combination_id,
            "transfer_measurement": "local_same_host_no_transfer",
        },
    )
    call.transition_to(CallStatus.SUCCEEDED if result.success else CallStatus.FAILED)
    return {
        "schema_version": 1,
        "combination_id": combination_id,
        "phase": phase,
        "sequence": sequence,
        "phase_started_at": phase_started_at,
        "submitted_offset_sec": submitted_at - phase_started,
        "started_offset_sec": started_at - phase_started,
        "finished_offset_sec": finished_at - phase_started,
        "queue_depth_after_submit": queue_depth_after_submit,
        "queue_depth_at_start": queue_depth_at_start,
        "running_at_start": running_at_start,
        "running_at_finish": running_at_finish,
        "source_path": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_bytes": source.stat().st_size,
        "call": call.to_dict(),
        "decision": decision.to_dict(),
        "result": result.to_dict(),
    }


def _append_trace_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        for record in records:
            call = ToolCall.from_dict(record["call"])
            decision = ScheduleDecision.from_dict(record["decision"])
            result = ToolResult.from_dict(record["result"])
            stream.write(
                build_tool_trace_record(
                    tool_call=call,
                    decision=decision,
                    result=result,
                    timeout=result.error_code == "timeout",
                ).to_json()
                + "\n"
            )


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
