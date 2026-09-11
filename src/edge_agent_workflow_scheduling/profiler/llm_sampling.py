"""Bounded real LLM measurements and provenance-preserving offline benchmark import.

The exported rate is (provider input + output tokens) / whole client request time,
matching the existing ProfileLLMExecutor formula. It is NOT a decode-only rate.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from edge_agent_workflow_scheduling.common import LLMCall
from edge_agent_workflow_scheduling.config import load_llm_profiles
from edge_agent_workflow_scheduling.executors import (
    create_openai_chat_executor,
    create_openai_responses_executor,
)
from edge_agent_workflow_scheduling.resources import LLMInstanceProfile

RATE_DEFINITION = "sum(input_tokens+output_tokens)/sum(client_request_seconds)"
TIMING_SCOPE = "client_request_including_network_and_server_wait"


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _positive(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def validate_config(config: dict) -> None:
    for key in ("repeats", "max_requests", "max_output_tokens", "output_token_budget"):
        _positive(config[key], key)
    timeout = config["timeout_sec"]
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, int | float)
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("timeout_sec must be finite and positive")
    if not config.get("version") or not config.get("samples"):
        raise ValueError("version and samples are required")
    seen = set()
    for sample in config["samples"]:
        for key in ("sample_id", "task_type", "input_size", "prompt"):
            if not isinstance(sample.get(key), str) or not sample[key].strip():
                raise ValueError(f"sample {key} must be non-empty")
        if sample["sample_id"] in seen:
            raise ValueError("duplicate sample_id")
        seen.add(sample["sample_id"])


def _redact(value: Any, secrets: list[str]) -> Any:
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, list):
        return [_redact(item, secrets) for item in value]
    if isinstance(value, dict):
        return {
            key: _redact(item, secrets)
            for key, item in value.items()
            if key.lower() not in {"api_key", "authorization", "password", "secret"}
        }
    return value


def _tokens(usage: dict | None, *names: str) -> int | None:
    for name in names:
        value = (usage or {}).get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def run_llm_sampling(
    config: dict,
    profiles_path: Path,
    output_root: Path,
    *,
    llm_ids: list[str] | None = None,
    executor_factory=None,
) -> Path:
    validate_config(config)
    profiles = load_llm_profiles(profiles_path)
    if llm_ids:
        unknown = set(llm_ids) - {p.llm_id for p in profiles}
        if unknown:
            raise ValueError(f"unknown llm_ids: {sorted(unknown)}")
        profiles = [p for p in profiles if p.llm_id in llm_ids]
    secrets = [os.environ[name] for p in profiles for name in p.secret_env_vars if os.getenv(name)]
    output = output_root / (datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    revision = (
        subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip()
        or None
    )
    manifest = {
        "schema_version": "llm-measurement-v1",
        "source_kind": "measured",
        "created_at": datetime.now(UTC).isoformat(),
        "config": deepcopy(config),
        "profiles": [p.to_dict() for p in profiles],
        "code_revision": revision,
        "implementation_sha256": _hash(Path(__file__)),
        "model_revision_status": "provider model identifier only; weight revision unavailable",
        "host": {
            "os": platform.system(),
            "release": platform.release(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
        },
        "hardware": None,
        "hardware_status": "not collected; client host is not necessarily server",
        "timing_scope": TIMING_SCOPE,
        "rate_definition": RATE_DEFINITION,
        "concurrency": 1,
        "warmup_policy": "none; all requests retained in order",
        "budget_policy": (
            "reserve max_output_tokens before each attempt; never refund failures; no SDK retries"
        ),
        "input_budget": "fixed finite prompts times max_requests; provider input tokens retained",
        "energy_joules": None,
        "cost": None,
        "quality_status": "unscored; milestone 4.5",
        "function_calling_verified": False,
    }
    write_json(output / "manifest.json", _redact(manifest, secrets))
    rows, skipped = [], []
    requests = reserved = 0
    stop = None
    with (output / "samples.jsonl").open("w", encoding="utf-8") as stream:
        for profile in profiles:
            deployment = profile.deployment_config
            reason = None
            if not deployment.get("enabled", True):
                reason = "deployment_disabled"
            elif deployment.get("model_env") and not os.getenv(deployment["model_env"]):
                reason = "model_environment_missing"
            elif deployment.get("requires_api_key", True) and (
                not profile.secret_env_vars
                or any(not os.getenv(n) for n in profile.secret_env_vars)
            ):
                reason = "credentials_missing"
            if reason:
                skipped.append({"llm_id": profile.llm_id, "reason": reason})
                continue
            params = deepcopy(deployment.get("model_parameters", {}))
            # Reject alternate provider caps so extra_body cannot silently defeat the budget.
            for options in (params, params.get("extra_body", {})):
                if any(
                    k in options
                    for k in ("max_tokens", "max_completion_tokens", "max_output_tokens", "n")
                ):
                    raise ValueError("output token limits belong in sampling config only")
            cap_key = (
                "max_tokens" if profile.executor_type == "openai_chat" else "max_output_tokens"
            )
            params[cap_key] = config["max_output_tokens"]
            measured_profile = replace(
                profile, deployment_config={**deployment, "model_parameters": params}
            )
            factories = {
                "openai_chat": create_openai_chat_executor,
                "openai_responses": create_openai_responses_executor,
            }
            factory = executor_factory or factories[profile.executor_type]
            try:
                executor = factory(measured_profile)
            except (RuntimeError, ImportError) as exc:
                skipped.append(
                    {
                        "llm_id": profile.llm_id,
                        "reason": "executor_unavailable",
                        "error_type": type(exc).__name__,
                    }
                )
                continue
            try:
                for sample in config["samples"]:
                    for repeat in range(config["repeats"]):
                        if (
                            requests >= config["max_requests"]
                            or reserved + config["max_output_tokens"]
                            > config["output_token_budget"]
                        ):
                            stop = "request_or_output_token_budget_exhausted"
                            break
                        requests += 1
                        reserved += config["max_output_tokens"]
                        call = LLMCall(
                            llm_call_id=f"sample-{requests}",
                            run_id=output.name,
                            agent_id="llm-sampler",
                            input_items=[{"role": "user", "content": sample["prompt"]}],
                            estimated_output_tokens=config["max_output_tokens"],
                            metadata={"task_type": sample["task_type"]},
                        )
                        result = executor.execute(call, timeout_sec=config["timeout_sec"])
                        usage = result.metadata.get("usage")
                        row = {
                            "llm_id": profile.llm_id,
                            "model": profile.model,
                            "endpoint": profile.base_url,
                            "model_version": result.response_model,
                            "sample_id": sample["sample_id"],
                            "task_type": sample["task_type"],
                            "input_size": sample["input_size"],
                            "repeat": repeat,
                            "prompt": sample["prompt"],
                            "sampling_parameters": params,
                            "input_tokens": _tokens(usage, "prompt_tokens", "input_tokens"),
                            "output_tokens": _tokens(usage, "completion_tokens", "output_tokens"),
                            "request_time_sec": result.inference_time_sec,
                            "success": result.success,
                            "usage": usage,
                            "energy_joules": None,
                            "cost": None,
                            "result": result.to_dict(),
                            "source_kind": "measured",
                        }
                        # The legacy result schema defaults missing energy to zero. Do not
                        # allow this placeholder to masquerade as an observed measurement.
                        row["result"]["energy_joules"] = None
                        row = _redact(row, secrets)
                        rows.append(row)
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                        stream.flush()
                    if stop:
                        break
            finally:
                client = getattr(executor, "client", None)
                if client is not None and hasattr(client, "close"):
                    client.close()
            if stop:
                break
    summary = summarize(rows)
    summary.update(
        {
            "skipped": skipped,
            "stop_reason": stop,
            "request_count": requests,
            "reserved_output_tokens": reserved,
            "status": (
                "skipped"
                if not rows
                else "failed"
                if not summary["success_count"]
                else "partial"
                if summary["failure_count"] or stop
                else "completed"
            ),
        }
    )
    write_json(output / "summary.json", summary)
    exported = export_profiles(
        rows,
        profiles,
        source_kind="measured",
        source_ref="samples.jsonl",
        source_hash=_hash(output / "samples.jsonl"),
    )
    write_json(output / "profiles.json", {"llm_instances": exported})
    return output


def summarize(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["llm_id"], row["task_type"], row["input_size"])].append(row)
    result = []
    for (llm_id, task, size), samples in groups.items():
        times = sorted(
            r["request_time_sec"] for r in samples if r.get("request_time_sec") is not None
        )
        result.append(
            {
                "llm_id": llm_id,
                "task_type": task,
                "input_size": size,
                "sample_count": len(samples),
                "success_count": sum(r["success"] for r in samples),
                "mean_sec": statistics.mean(times) if times else None,
                "p50_sec": statistics.median(times) if times else None,
                "p95_sec": times[math.ceil(0.95 * len(times)) - 1] if times else None,
                "latency_population": "all terminal requests including failures",
            }
        )
    return {
        "groups": result,
        "success_count": sum(r["success"] for r in rows),
        "failure_count": sum(not r["success"] for r in rows),
    }


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_profiles(
    rows: list[dict],
    profiles: list[LLMInstanceProfile],
    *,
    source_kind: str,
    source_ref: str,
    source_hash: str,
) -> list[dict]:
    exported = []
    for profile in profiles:
        selected = [
            r
            for r in rows
            if r["llm_id"] == profile.llm_id
            and r["success"]
            and r.get("input_tokens") is not None
            and r.get("output_tokens") is not None
            and r.get("request_time_sec") is not None
            and r["request_time_sec"] > 0
            and r["input_tokens"] + r["output_tokens"] > 0
        ]
        if not selected:
            continue
        rate = sum(r["input_tokens"] + r["output_tokens"] for r in selected) / sum(
            r["request_time_sec"] for r in selected
        )
        metadata = {
            "source_kind": source_kind,
            "source_ref": source_ref,
            "source_sha256": source_hash,
            "profile_version": "llm-measurement-v1",
            "rate_definition": RATE_DEFINITION,
            "timing_scope": TIMING_SCOPE,
            "function_calling_verified": False,
            "quality_status": "unscored",
            "energy_status": "unavailable",
            "sample_count": len(selected),
            "excluded_count": sum(r["llm_id"] == profile.llm_id for r in rows) - len(selected),
            "exclusion_rule": "failed or missing/zero timing or token counts; raw records retained",
            "scope": {
                "task_types": sorted({r["task_type"] for r in selected}),
                "input_sizes": sorted({r["input_size"] for r in selected}),
                "input_token_range": [
                    min(r["input_tokens"] for r in selected),
                    max(r["input_tokens"] for r in selected),
                ],
                "output_token_range": [
                    min(r["output_tokens"] for r in selected),
                    max(r["output_tokens"] for r in selected),
                ],
                "sample_ids": sorted({r["sample_id"] for r in selected}),
                "concurrency": 1,
            },
            "scope_policy": (
                "aggregate only for recorded workload; caller must restrict replay; "
                "no out-of-range accuracy claim"
            ),
        }
        exported.append(
            replace(
                profile,
                llm_id=profile.llm_id + "-measured-profile",
                executor_type="profile",
                capabilities=[],
                quality_profile={},
                energy_profile={},
                token_profile={"tokens_per_sec": rate},
                deployment_config={},
                secret_env_vars=[],
                metadata=metadata,
            ).to_dict()
        )
    return exported


def import_benchmark(path: Path, output: Path) -> Path:
    """Import the documented JSON interchange format, preserving unknown measurements."""
    data = json.loads(path.read_text(encoding="utf-8"))
    provenance = data["provenance"]
    for key in ("source", "version", "task_definition", "scoring_rule", "timing_scope"):
        if not isinstance(provenance.get(key), str) or not provenance[key].strip():
            raise ValueError(f"benchmark provenance requires {key}")
    for key in ("hardware", "runtime_config"):
        if key not in provenance:
            raise ValueError(f"benchmark provenance requires {key} (null if unknown)")
    if provenance.get("concurrency", 1) != 1:
        raise ValueError("only single-request benchmark latency is supported")
    if provenance["timing_scope"] != TIMING_SCOPE:
        raise ValueError("benchmark timing cannot be converted to existing executor rate")
    profiles = [LLMInstanceProfile.from_dict(p) for p in data["llm_instances"]]
    ids = {p.llm_id for p in profiles}
    if len(ids) != len(profiles):
        raise ValueError("duplicate profile ID")
    for row in data["samples"]:
        if row["llm_id"] not in ids or not isinstance(row["success"], bool):
            raise ValueError("unknown llm_id or invalid success flag")
        for key in ("sample_id", "task_type", "input_size"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"invalid {key}")
        for key in ("input_tokens", "output_tokens", "request_time_sec"):
            value = row.get(key)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"invalid {key}")
            if key.endswith("tokens") and value is not None and not isinstance(value, int):
                raise ValueError(f"{key} must be an integer")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "benchmark.json", data)
    write_json(
        output / "manifest.json",
        {"source_kind": "benchmark", "source_sha256": _hash(path), "provenance": provenance},
    )
    write_json(output / "summary.json", summarize(data["samples"]))
    exported = export_profiles(
        data["samples"],
        profiles,
        source_kind="benchmark",
        source_ref="benchmark.json",
        source_hash=_hash(path),
    )
    write_json(output / "profiles.json", {"llm_instances": exported})
    return output
