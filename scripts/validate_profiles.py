"""Validate fitted profiles against independent holdout observations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from edge_agent_workflow_scheduling.profiler import (
    ProfileValidationConfig,
    validate_profile_catalog,
    write_profile_validation_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/profile_validation_v1.json"),
    )
    parser.add_argument(
        "--profile-catalog",
        type=Path,
        default=Path("data/profile_calibration/arch-linux-calibrated-v1/profiles.json"),
    )
    parser.add_argument("--tool-holdout-run", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/profile_validation/arch-linux-profile-validation-v1"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Exit nonzero after writing artifacts when a threshold fails",
    )
    args = parser.parse_args()

    config = ProfileValidationConfig.from_json(args.config)
    report, samples = validate_profile_catalog(
        args.profile_catalog,
        args.tool_holdout_run,
        config,
    )
    report["provenance"].update(
        {
            "validation_config_ref": str(args.config),
            "validation_config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        }
    )
    output = write_profile_validation_artifacts(
        args.output_dir,
        report,
        samples,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "sample_count": len(samples),
                "output": str(output),
                "thresholds": report["tool_execution_time"]["thresholds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.require_pass and report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
