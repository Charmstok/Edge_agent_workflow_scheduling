"""Small versioned task datasets and deterministic multi-Agent arrival plans."""

from __future__ import annotations

import hashlib
import json
import random
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from math import isfinite
from pathlib import Path
from string import Formatter
from typing import Any, Literal, Self

TaskSplit = Literal["calibration", "validation"]


def _text(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _integer(value: Any, name: str, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _positive(value: Any, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be finite and positive")


def _names(values: Any, name: str, *, allow_empty: bool = False) -> None:
    if not isinstance(values, list | tuple) or (not values and not allow_empty):
        raise ValueError(f"{name} must be a sequence of names")
    for value in values:
        _text(value, name)
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")


def _object(value: Any, name: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    json.dumps(value, allow_nan=False)


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class TaskSample:
    """One input; source IDs group all derivatives, including composite tasks."""

    task_id: str
    task_type: str
    data_version: str
    source_ids: tuple[str, ...]
    split: TaskSplit
    input_size: Literal["small", "medium", "large"]
    size_features: dict[str, int | float]
    allowed_tools: tuple[str, ...]
    goal: str
    scoring_rule: dict[str, Any]
    reference_answer: dict[str, Any]
    input_payload: dict[str, Any] = field(default_factory=dict)
    artifact_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("task_id", "task_type", "data_version", "goal"):
            _text(getattr(self, name), name)
        _names(self.source_ids, "source_ids")
        _names(self.allowed_tools, "allowed_tools", allow_empty=True)
        _names(self.artifact_refs, "artifact_refs", allow_empty=True)
        if self.split not in {"calibration", "validation"}:
            raise ValueError("split must be calibration or validation")
        if self.input_size not in {"small", "medium", "large"}:
            raise ValueError("input_size must be small, medium, or large")
        for name in ("size_features", "input_payload", "scoring_rule", "reference_answer"):
            _object(getattr(self, name), name)
        if not self.size_features:
            raise ValueError("size_features must not be empty")
        for name, value in self.size_features.items():
            _positive(value, name)
        for name in ("name", "version", "description"):
            _text(self.scoring_rule.get(name), f"scoring_rule.{name}")
        if not self.reference_answer:
            raise ValueError("reference_answer must not be empty")
        if not self.input_payload and not self.artifact_refs:
            raise ValueError("task requires inline input or artifact_refs")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        value = dict(data)
        for name in ("source_ids", "allowed_tools", "artifact_refs"):
            _names(value.get(name, []), name, allow_empty=name != "source_ids")
            value[name] = tuple(value.get(name, []))
        return cls(**value)


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """Prompt, ordered Tool contracts and limits, without binding a model target."""

    system_prompt: str
    system_prompt_version: str
    user_template: str
    user_template_version: str
    tool_schemas: tuple[dict[str, Any], ...]
    tool_schema_version: str
    max_rounds: int = 8
    max_tool_calls: int = 16
    timeout_sec: float = 120.0
    sampling_parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "system_prompt", "system_prompt_version", "user_template",
            "user_template_version", "tool_schema_version",
        ):
            _text(getattr(self, name), name)
        _integer(self.max_rounds, "max_rounds", 1)
        _integer(self.max_tool_calls, "max_tool_calls")
        _positive(self.timeout_sec, "timeout_sec")
        _object(self.sampling_parameters, "sampling_parameters")
        for schema in self.tool_schemas:
            _object(schema, "tool schema")
            _text(schema.get("name"), "tool name")
            if schema.get("type") != "function":
                raise ValueError("tool schema type must be function")
            _object(schema.get("parameters"), "tool parameters")
        _names(self.tool_order, "tool_order", allow_empty=True)
        fields = {name for _, name, _, _ in Formatter().parse(self.user_template) if name}
        if not fields <= {"task_id", "task_type", "goal", "input_json", "artifacts_json"}:
            raise ValueError("unsupported user_template placeholder")

    @property
    def tool_order(self) -> tuple[str, ...]:
        return tuple(schema["name"] for schema in self.tool_schemas)

    def tools_for(self, sample: TaskSample) -> list[dict[str, Any]]:
        return deepcopy([
            schema for schema in self.tool_schemas if schema["name"] in sample.allowed_tools
        ])

    def render_user_task(self, sample: TaskSample) -> str:
        return self.user_template.format(
            task_id=sample.task_id,
            task_type=sample.task_type,
            goal=sample.goal,
            input_json=json.dumps(sample.input_payload, ensure_ascii=False, sort_keys=True),
            artifacts_json=json.dumps(sample.artifact_refs, ensure_ascii=False),
        )


@dataclass(frozen=True, slots=True)
class ArrivalPlan:
    """Requested arrivals, not execution times or a precomputed call DAG."""

    mode: Literal["fixed", "poisson", "burst"]
    request_count: int
    seed: int
    interval_sec: float
    concurrency_limit: int
    burst_size: int = 1

    def __post_init__(self) -> None:
        if self.mode not in {"fixed", "poisson", "burst"}:
            raise ValueError("mode must be fixed, poisson, or burst")
        _integer(self.request_count, "request_count", 1)
        _integer(self.seed, "arrival seed")
        _integer(self.concurrency_limit, "concurrency_limit", 1)
        _integer(self.burst_size, "burst_size", 1)
        _positive(self.interval_sec, "interval_sec")

    def arrival_times(self) -> tuple[float, ...]:
        rng = random.Random(self.seed)
        current = 0.0
        times = []
        for index in range(self.request_count):
            if self.mode == "fixed":
                current = index * self.interval_sec
            elif self.mode == "burst":
                current = (index // self.burst_size) * self.interval_sec
            elif index:
                current += rng.expovariate(1.0 / self.interval_sec)
            times.append(current)
        return tuple(times)


@dataclass(frozen=True, slots=True)
class WorkloadConfig:
    """Self-contained experiment inputs; Scheduler/profile seeds remain external."""

    workload_id: str
    workload_version: str
    dataset_id: str
    tasks: tuple[TaskSample, ...]
    agent: AgentConfig
    arrival_plans: dict[str, ArrivalPlan]
    input_seed: int
    agent_count: int = 4
    schema_version: int = 1

    def __post_init__(self) -> None:
        for name in ("workload_id", "workload_version", "dataset_id"):
            _text(getattr(self, name), name)
        _integer(self.input_seed, "input_seed")
        _integer(self.agent_count, "agent_count", 1)
        _integer(self.schema_version, "schema_version", 1)
        if self.schema_version != 1:
            raise ValueError("unsupported workload schema_version")
        if not self.tasks:
            raise ValueError("tasks must not be empty")
        _names([task.task_id for task in self.tasks], "task_id")
        if not self.arrival_plans:
            raise ValueError("arrival_plans must not be empty")
        _names(list(self.arrival_plans), "scenario names")
        groups: dict[str, str] = {}
        artifact_splits: dict[str, str] = {}
        for sample in self.tasks:
            if not set(sample.allowed_tools) <= set(self.agent.tool_order):
                raise ValueError(f"unknown allowed_tools for {sample.task_id}")
            self.agent.render_user_task(sample)
            for source_id in sample.source_ids:
                if groups.setdefault(source_id, sample.split) != sample.split:
                    raise ValueError(f"source_id overlaps calibration and validation: {source_id}")
            for reference in sample.artifact_refs:
                if artifact_splits.setdefault(reference, sample.split) != sample.split:
                    raise ValueError(f"artifact overlaps calibration and validation: {reference}")

    def artifact_hashes(self, artifact_root: str | Path) -> dict[str, str]:
        root = Path(artifact_root)
        hashes = {}
        for sample in self.tasks:
            for reference in sample.artifact_refs:
                path = root / reference
                if not path.is_file():
                    raise FileNotFoundError(f"missing task artifact: {path}")
                hashes[reference] = hashlib.sha256(path.read_bytes()).hexdigest()
        return hashes

    def generate(
        self, scenario: str, *, split: TaskSplit, artifact_root: str | Path,
    ) -> dict[str, Any]:
        """Validate all files and materialize seeded task choices and arrival offsets."""

        if split not in {"calibration", "validation"}:
            raise ValueError("split must be calibration or validation")
        if scenario not in self.arrival_plans:
            raise ValueError(f"unknown scenario: {scenario}")
        artifacts = self.artifact_hashes(artifact_root)
        samples = sorted(
            (sample for sample in self.tasks if sample.split == split),
            key=lambda sample: sample.task_id,
        )
        if not samples:
            raise ValueError(f"no samples in split: {split}")
        rng = random.Random(self.input_seed)
        requests = []
        for index, arrival in enumerate(self.arrival_plans[scenario].arrival_times()):
            if index % len(samples) == 0:
                rng.shuffle(samples)
            sample = samples[index % len(samples)]
            input_data = {
                "input_payload": sample.input_payload,
                "artifacts": {name: artifacts[name] for name in sample.artifact_refs},
            }
            requests.append({
                "run_id": f"{self.workload_id}-{scenario}-{split}-{index:06d}",
                "agent_id": f"agent-{index % self.agent_count:03d}",
                "task_id": sample.task_id,
                "task_type": sample.task_type,
                "arrival_offset_sec": arrival,
                "input_digest": _digest(input_data),
                "user_task": self.agent.render_user_task(sample),
                "allowed_tools": list(sample.allowed_tools),
                "size_features": sample.size_features,
            })
        return deepcopy({
            "schema_version": self.schema_version,
            "workload_config": self.to_dict(),
            "workload_digest": _digest(self.to_dict()),
            "artifact_hashes": artifacts,
            "scenario": scenario,
            "split": split,
            "concurrency_limit": self.arrival_plans[scenario].concurrency_limit,
            "requests": requests,
        })

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self), allow_nan=False))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        value = dict(data)
        value["tasks"] = tuple(TaskSample.from_dict(item) for item in value["tasks"])
        agent = dict(value["agent"])
        agent["tool_schemas"] = tuple(agent["tool_schemas"])
        value["agent"] = AgentConfig(**agent)
        value["arrival_plans"] = {
            name: ArrivalPlan(**plan) for name, plan in value["arrival_plans"].items()
        }
        return cls(**value)

    @classmethod
    def from_json(cls, path: str | Path) -> Self:
        workload = cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        workload.artifact_hashes(Path(path).parent)
        return workload
