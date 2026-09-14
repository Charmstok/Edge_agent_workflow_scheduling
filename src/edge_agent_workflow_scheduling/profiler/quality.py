"""Versioned rule scoring and task-specific LLM quality calibration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from math import isfinite, sqrt
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Literal, Self

from edge_agent_workflow_scheduling.common import WorkloadConfig
from edge_agent_workflow_scheduling.common.workload import TaskSample
from edge_agent_workflow_scheduling.profiler.models import TraceBundle
from edge_agent_workflow_scheduling.profiler.privacy import content_digest
from edge_agent_workflow_scheduling.resources import LLMInstanceProfile

SCORING_REGISTRY_VERSION = "task-quality-scorers-v1"
SUPPORTED_SCORERS = frozenset({("exact_fields", "v1")})
MissingTaskTypePolicy = Literal["error", "omit"]


@dataclass(frozen=True, slots=True)
class QualityCalibrationConfig:
    """Reproducible quality-profile construction policy."""

    quality_profile_version: str
    workload_config: str
    llm_catalog: str
    system_prompt_suffix: str
    system_prompt_version: str
    llm_ids: tuple[str, ...]
    repeats: int
    max_output_tokens: int
    missing_task_type_policy: MissingTaskTypePolicy
    allow_default_quality: bool
    default_quality: float | None
    minimum_calibration_samples_per_task_type: int = 1
    confidence_z: float = 1.96
    schema_version: int = 1
    scoring_registry_version: str = SCORING_REGISTRY_VERSION

    def __post_init__(self) -> None:
        for value, name in (
            (self.quality_profile_version, "quality_profile_version"),
            (self.workload_config, "workload_config"),
            (self.llm_catalog, "llm_catalog"),
            (self.system_prompt_suffix, "system_prompt_suffix"),
            (self.system_prompt_version, "system_prompt_version"),
            (self.scoring_registry_version, "scoring_registry_version"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        if self.scoring_registry_version != SCORING_REGISTRY_VERSION:
            raise ValueError("unsupported scoring_registry_version")
        if not self.llm_ids or any(
            not isinstance(item, str) or not item.strip() for item in self.llm_ids
        ):
            raise ValueError("llm_ids must contain non-empty model IDs")
        if len(set(self.llm_ids)) != len(self.llm_ids):
            raise ValueError("llm_ids must not contain duplicates")
        for value, name in (
            (self.repeats, "repeats"),
            (self.max_output_tokens, "max_output_tokens"),
            (
                self.minimum_calibration_samples_per_task_type,
                "minimum_calibration_samples_per_task_type",
            ),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.missing_task_type_policy not in {"error", "omit"}:
            raise ValueError("missing_task_type_policy must be error or omit")
        if not isinstance(self.allow_default_quality, bool):
            raise ValueError("allow_default_quality must be a boolean")
        if self.allow_default_quality:
            _fraction(self.default_quality, "default_quality")
        elif self.default_quality is not None:
            raise ValueError("default_quality requires allow_default_quality=true")
        if (
            isinstance(self.confidence_z, bool)
            or not isinstance(self.confidence_z, int | float)
            or not isfinite(self.confidence_z)
            or self.confidence_z <= 0
        ):
            raise ValueError("confidence_z must be finite and positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        value = dict(data)
        value["llm_ids"] = tuple(value["llm_ids"])
        return cls(**value)

    @classmethod
    def from_json(cls, path: str | Path) -> Self:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("quality calibration config must be a JSON object")
        return cls.from_dict(data)


@dataclass(frozen=True, slots=True)
class TaskScore:
    """One deterministic final-answer score with model attribution metadata."""

    run_id: str
    task_id: str
    task_type: str
    split: str
    status: str
    scoring_rule_name: str
    scoring_rule_version: str
    raw_score: float
    normalized_score: float
    score_mapping: str
    score_source: str
    model_ids: tuple[str, ...] = ()
    attributable_model_id: str | None = None
    timeout: bool = False
    failure_reason: str | None = None
    matched_items: int = 0
    total_items: int = 0
    details: dict[str, Any] = field(default_factory=dict)
    trace_digest: str | None = None
    trace_ref: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        value = dict(data)
        value["model_ids"] = tuple(value.get("model_ids", ()))
        return cls(**value)


def validate_scoring_rules(workload: WorkloadConfig) -> None:
    """Reject unsupported task scoring rules before running an experiment."""

    for sample in workload.tasks:
        key = (sample.scoring_rule["name"], sample.scoring_rule["version"])
        if key not in SUPPORTED_SCORERS:
            raise ValueError(
                f"unsupported scoring rule for {sample.task_id!r}: {key[0]} {key[1]}"
            )


def score_task_output(
    sample: TaskSample,
    final_output: str | None,
    *,
    run_id: str,
    status: str = "completed",
    timeout: bool = False,
    model_ids: tuple[str, ...] = (),
    trace_digest: str | None = None,
    trace_ref: str | None = None,
) -> TaskScore:
    """Score one final answer according to its versioned workload rule."""

    key = (sample.scoring_rule["name"], sample.scoring_rule["version"])
    if key not in SUPPORTED_SCORERS:
        raise ValueError(f"unsupported scoring rule: {key[0]} {key[1]}")
    attributable_model = model_ids[0] if len(model_ids) == 1 else None
    common = {
        "run_id": run_id,
        "task_id": sample.task_id,
        "task_type": sample.task_type,
        "split": sample.split,
        "status": status,
        "scoring_rule_name": key[0],
        "scoring_rule_version": key[1],
        "score_mapping": "identity_from_fraction_to_[0,1]",
        "score_source": "deterministic_rule",
        "model_ids": model_ids,
        "attributable_model_id": attributable_model,
        "timeout": timeout,
        "trace_digest": trace_digest,
        "trace_ref": trace_ref,
    }
    if timeout or status != "completed":
        reason = "timeout" if timeout else "run_failed"
        return TaskScore(
            **common,
            raw_score=0.0,
            normalized_score=0.0,
            failure_reason=reason,
            total_items=len(sample.reference_answer),
        )
    if final_output is None or not final_output.strip():
        return TaskScore(
            **common,
            raw_score=0.0,
            normalized_score=0.0,
            failure_reason="empty_answer",
            total_items=len(sample.reference_answer),
        )
    answer, parse_mode = _parse_answer_object(final_output)
    if answer is None:
        return TaskScore(
            **common,
            raw_score=0.0,
            normalized_score=0.0,
            failure_reason="invalid_json",
            total_items=len(sample.reference_answer),
        )
    if not isinstance(answer, dict):
        return TaskScore(
            **common,
            raw_score=0.0,
            normalized_score=0.0,
            failure_reason="answer_is_not_object",
            total_items=len(sample.reference_answer),
        )

    field_matches = {
        name: name in answer and _exact_value(answer[name], expected)
        for name, expected in sample.reference_answer.items()
    }
    matched = sum(field_matches.values())
    total = len(field_matches)
    raw_score = matched / total
    return TaskScore(
        **common,
        raw_score=raw_score,
        normalized_score=raw_score,
        matched_items=matched,
        total_items=total,
        details={
            "field_matches": field_matches,
            "extra_fields_ignored": sorted(set(answer) - set(field_matches)),
            "parse_mode": parse_mode,
        },
    )


def score_trace_bundle(
    trace: TraceBundle,
    workload: WorkloadConfig,
    *,
    trace_ref: str | None = None,
) -> TaskScore:
    """Score a terminal TraceBundle and attribute only single-model Agent runs."""

    if trace.manifest.dataset_id != workload.dataset_id:
        raise ValueError(
            f"trace dataset {trace.manifest.dataset_id!r} does not match {workload.dataset_id!r}"
        )
    samples = {sample.task_id: sample for sample in workload.tasks}
    try:
        sample = samples[trace.run.task_id]
    except KeyError as exc:
        raise ValueError(f"trace task_id {trace.run.task_id!r} is absent from workload") from exc
    if trace.run.task_id not in trace.manifest.sample_ids:
        raise ValueError("trace manifest sample_ids does not contain the run task_id")
    model_ids = tuple(
        sorted({call.selected_target for call in trace.calls if call.call_kind == "llm"})
    )
    timed_out = trace.run.error_code == "timeout" or any(call.timeout for call in trace.calls)
    return score_task_output(
        sample,
        trace.run.final_output,
        run_id=trace.run.run_id,
        status=trace.run.status,
        timeout=timed_out,
        model_ids=model_ids,
        trace_digest=content_digest(trace.to_dict()),
        trace_ref=trace_ref,
    )


def score_trace_bundles(
    traces: list[TraceBundle],
    workload: WorkloadConfig,
    *,
    trace_refs: list[str | None] | None = None,
) -> list[TaskScore]:
    """Score comparable traces after checking prompt, Tool and budget parity."""

    validate_scoring_rules(workload)
    refs = trace_refs or [None] * len(traces)
    if len(refs) != len(traces):
        raise ValueError("trace_refs must align with traces")
    protocol_by_task: dict[str, str] = {}
    scores = []
    for trace, reference in zip(traces, refs, strict=True):
        protocol = _protocol_digest(trace)
        existing = protocol_by_task.setdefault(trace.run.task_id, protocol)
        if existing != protocol:
            raise ValueError(
                f"task {trace.run.task_id!r} was run with different prompts, Tools, or budgets"
            )
        scores.append(score_trace_bundle(trace, workload, trace_ref=reference))
    return scores


def build_quality_report(
    scores: list[TaskScore],
    workload: WorkloadConfig,
    config: QualityCalibrationConfig,
) -> dict[str, Any]:
    """Fit calibration profiles and report validation performance separately."""

    if not scores:
        raise ValueError("scores must not be empty")
    task_types = sorted({sample.task_type for sample in workload.tasks})
    attributable = [score for score in scores if score.attributable_model_id is not None]
    mixed = [score.run_id for score in scores if len(score.model_ids) > 1]
    no_model = [score.run_id for score in scores if not score.model_ids]
    model_ids = sorted({score.attributable_model_id for score in attributable})
    profiles: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for model_id in model_ids:
        calibration_entries: dict[str, dict[str, Any]] = {}
        validation_entries: dict[str, dict[str, Any]] = {}
        for task_type in task_types:
            calibration_scores = [
                score for score in attributable
                if score.attributable_model_id == model_id
                and score.task_type == task_type
                and score.split == "calibration"
            ]
            holdout_scores = [
                score for score in attributable
                if score.attributable_model_id == model_id
                and score.task_type == task_type
                and score.split == "validation"
            ]
            if len(calibration_scores) < config.minimum_calibration_samples_per_task_type:
                if config.missing_task_type_policy == "error":
                    raise ValueError(
                        f"model {model_id!r} has {len(calibration_scores)} calibration samples "
                        f"for {task_type!r}; requires "
                        f"{config.minimum_calibration_samples_per_task_type}"
                    )
            else:
                calibration_entries[task_type] = _aggregate_scores(
                    calibration_scores, config.confidence_z
                )
            if holdout_scores:
                validation_entries[task_type] = _aggregate_scores(
                    holdout_scores, config.confidence_z
                )

        quality_profile = {
            task_type: aggregate["mean"]
            for task_type, aggregate in calibration_entries.items()
        }
        if config.allow_default_quality:
            quality_profile["default"] = config.default_quality
        profiles.append(
            {
                "llm_id": model_id,
                "quality_profile": quality_profile,
                "calibration": calibration_entries,
            }
        )
        validation.append(
            {
                "llm_id": model_id,
                "task_types": {
                    task_type: {
                        **aggregate,
                        "calibration_mean": calibration_entries.get(task_type, {}).get("mean"),
                        "generalization_gap": (
                            aggregate["mean"] - calibration_entries[task_type]["mean"]
                            if task_type in calibration_entries
                            else None
                        ),
                    }
                    for task_type, aggregate in validation_entries.items()
                },
            }
        )

    return {
        "schema_version": 1,
        "quality_profile_version": config.quality_profile_version,
        "scoring_registry_version": config.scoring_registry_version,
        "dataset_id": workload.dataset_id,
        "workload_id": workload.workload_id,
        "workload_version": workload.workload_version,
        "workload_digest": content_digest(workload.to_dict()),
        "score_mapping": "identity_from_fraction_to_[0,1]",
        "missing_task_type_policy": config.missing_task_type_policy,
        "allow_default_quality": config.allow_default_quality,
        "default_quality": config.default_quality,
        "minimum_calibration_samples_per_task_type": (
            config.minimum_calibration_samples_per_task_type
        ),
        "confidence_z": config.confidence_z,
        "profiles": profiles,
        "validation": validation,
        "score_count": len(scores),
        "attributable_score_count": len(attributable),
        "mixed_model_runs_excluded_from_model_profiles": mixed,
        "runs_without_llm_excluded_from_model_profiles": no_model,
        "provenance": {
            "task_ids": sorted({score.task_id for score in scores}),
            "run_ids": sorted(score.run_id for score in scores),
            "scoring_rules": sorted(
                {f"{score.scoring_rule_name}:{score.scoring_rule_version}" for score in scores}
            ),
        },
    }


def apply_quality_report(
    profiles: list[LLMInstanceProfile],
    report: dict[str, Any],
) -> list[LLMInstanceProfile]:
    """Return profiles updated only for model IDs covered by calibration data."""

    calibrated = {entry["llm_id"]: entry for entry in report["profiles"]}
    updated: list[LLMInstanceProfile] = []
    for profile in profiles:
        entry = calibrated.get(profile.llm_id)
        if entry is None:
            updated.append(profile)
            continue
        metadata = {
            **profile.metadata,
            "quality_status": "calibrated_task_score",
            "quality_profile_version": report["quality_profile_version"],
            "quality_dataset_id": report["dataset_id"],
            "quality_workload_digest": report["workload_digest"],
            "quality_scoring_registry_version": report["scoring_registry_version"],
            "quality_missing_task_type_policy": report["missing_task_type_policy"],
            "quality_allow_default": report["allow_default_quality"],
            "quality_calibration": entry["calibration"],
        }
        updated.append(
            replace(profile, quality_profile=dict(entry["quality_profile"]), metadata=metadata)
        )
    return updated


def write_quality_artifacts(
    output_dir: str | Path,
    scores: list[TaskScore],
    report: dict[str, Any],
    profiles: list[LLMInstanceProfile],
) -> None:
    """Write raw scores, aggregate report and directly loadable profiles."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "scores.jsonl").open("w", encoding="utf-8") as output:
        for score in scores:
            output.write(json.dumps(score.to_dict(), ensure_ascii=False, sort_keys=True))
            output.write("\n")
    (directory / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (directory / "profiles.json").write_text(
        json.dumps(
            {"llm_instances": [profile.to_dict() for profile in profiles]},
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _aggregate_scores(scores: list[TaskScore], confidence_z: float) -> dict[str, Any]:
    values = [score.normalized_score for score in scores]
    mean = fmean(values)
    deviation = pstdev(values)
    half_width = confidence_z * deviation / sqrt(len(values))
    return {
        "sample_count": len(values),
        "mean": mean,
        "population_stddev": deviation,
        "confidence_interval": [max(0.0, mean - half_width), min(1.0, mean + half_width)],
        "minimum": min(values),
        "maximum": max(values),
        "task_ids": sorted({score.task_id for score in scores}),
        "run_ids": sorted(score.run_id for score in scores),
    }


def _protocol_digest(trace: TraceBundle) -> str:
    manifest = trace.manifest
    return content_digest(
        {
            "dataset_id": manifest.dataset_id,
            "system_prompt": manifest.system_prompt,
            "system_prompt_version": manifest.system_prompt_version,
            "user_template": manifest.user_template,
            "user_template_version": manifest.user_template_version,
            "tool_schemas": manifest.tool_schemas,
            "tool_order": manifest.tool_order,
            "tool_implementation_versions": manifest.tool_implementation_versions,
            "sampling_parameters": manifest.sampling_parameters,
            "agent_limits": manifest.agent_limits,
        }
    )


def _exact_value(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, str):
        return isinstance(actual, str) and actual.strip() == expected.strip()
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return all(
            key in actual and _exact_value(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _exact_value(left, right) for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected


def _parse_answer_object(value: str) -> tuple[dict[str, Any] | None, str | None]:
    """Accept raw JSON or one standalone JSON markdown fence."""

    try:
        answer = json.loads(value)
    except json.JSONDecodeError:
        lines = value.strip().splitlines()
        if len(lines) < 3 or lines[0].strip().lower() not in {"```", "```json"}:
            return None, None
        if lines[-1].strip() != "```":
            return None, None
        try:
            answer = json.loads("\n".join(lines[1:-1]).strip())
        except json.JSONDecodeError:
            return None, None
        parse_mode = "standalone_json_fence"
    else:
        parse_mode = "raw_json"
    return (answer, parse_mode) if isinstance(answer, dict) else (None, None)


def _fraction(value: object, field_name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{field_name} must be between 0 and 1")
