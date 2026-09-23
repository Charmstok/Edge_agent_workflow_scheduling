"""Fit and optionally validate a traceable resource-profile catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from edge_agent_workflow_scheduling.profiler import (
    ProfileValidationConfig,
    SyntheticEnergyConfig,
    fit_profile_catalog,
    validate_profile_catalog,
    write_profile_catalog,
    write_profile_validation_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool-sampling-run", type=Path, action="append")
    parser.add_argument("--llm-benchmark", type=Path)
    parser.add_argument("--profile-version", default="arch-linux-calibrated-v1")
    parser.add_argument("--fallback-policy", choices=("error", "aggregate"), default="error")
    parser.add_argument("--synthetic-energy-config", type=Path)
    parser.add_argument("--profile-output", type=Path, required=True)
    parser.add_argument("--tool-holdout-run", type=Path)
    parser.add_argument("--validation-config", type=Path)
    parser.add_argument("--validation-output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()

    energy = None
    if args.synthetic_energy_config:
        energy = SyntheticEnergyConfig.from_mapping(
            _read_object(args.synthetic_energy_config)
        )
    catalog = fit_profile_catalog(
        profile_version=args.profile_version,
        tool_sampling_run=args.tool_sampling_run,
        llm_benchmark=args.llm_benchmark,
        fallback_policy=args.fallback_policy,
        synthetic_energy=energy,
    )
    profile_path = write_profile_catalog(
        catalog, args.profile_output, overwrite=args.overwrite
    )

    validation: dict[str, Any] | None = None
    validation_output: str | None = None
    if args.tool_holdout_run:
        if args.validation_config is None or args.validation_output is None:
            parser.error(
                "--tool-holdout-run requires --validation-config and --validation-output"
            )
        config = ProfileValidationConfig.from_json(args.validation_config)
        report, samples = validate_profile_catalog(
            profile_path, args.tool_holdout_run, config
        )
        report["provenance"].update(
            {
                "validation_config_ref": str(args.validation_config),
                "validation_config_sha256": _sha256(args.validation_config),
            }
        )
        output = write_profile_validation_artifacts(
            args.validation_output,
            report,
            samples,
            overwrite=args.overwrite,
        )
        validation = {
            "status": report["status"],
            "sample_count": len(samples),
            "report": str(output),
        }
        validation_output = str(output)
        if args.require_pass and report["status"] != "passed":
            raise SystemExit(1)

    summary = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "profile_version": args.profile_version,
        "profile_catalog": str(profile_path),
        "profile_catalog_sha256": _sha256(profile_path),
        "inputs": {
            "tool_sampling_runs": [str(path) for path in args.tool_sampling_run or []],
            "llm_benchmark": str(args.llm_benchmark) if args.llm_benchmark else None,
            "synthetic_energy_config": (
                str(args.synthetic_energy_config)
                if args.synthetic_energy_config
                else None
            ),
        },
        "validation": validation,
        "validation_output": validation_output,
    }
    summary_path = profile_path.parent / "calibration_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"profile_catalog": str(profile_path), "validation": validation}, indent=2))


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
