#!/usr/bin/env python3
"""Validate local Qwen and Ark profiles offline, with optional bounded live probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from edge_agent_workflow_scheduling.config import load_llm_profiles
from edge_agent_workflow_scheduling.executors.llm_deployment import LLMDeployment
from edge_agent_workflow_scheduling.profiler.llm_deployment import (
    catalog_report,
    live_checks,
    offline_contract,
    write_report,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/llm_deployment_validation_v1.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/llm_deployment_validation"))
    parser.add_argument(
        "--live-local",
        action="store_true",
        help="Probe local endpoints and real automatic Function Calling",
    )
    parser.add_argument(
        "--cloud-smoke",
        action="store_true",
        help="Explicitly enable up to one capped request per cloud profile; requires ARK_API_KEY",
    )
    parser.add_argument(
        "--require-local-function-calling",
        action="store_true",
        help="Fail unless a requested real local Function Calling check passes",
    )
    args = parser.parse_args()
    if args.require_local_function_calling and not args.live_local:
        parser.error("--require-local-function-calling needs --live-local")
    config = json.loads(args.config.read_text())
    profiles = load_llm_profiles(config["catalog"])
    directory = args.output_dir.resolve()
    report = {
        "version": config["version"],
        "catalog": catalog_report(profiles, config),
        "configuration": config,
        "cloud_smoke_requested": args.cloud_smoke,
    }
    with LLMDeployment(profiles, offline_rates=config["offline_tokens_per_sec"]) as deployment:
        report["offline"] = offline_contract(deployment, directory / "offline")
    if args.live_local or args.cloud_smoke:
        report["live"] = live_checks(
            profiles,
            config,
            output_dir=directory / "live",
            live_local=args.live_local,
            cloud_smoke=args.cloud_smoke,
        )
    else:
        report["live"] = {
            "status": "skipped",
            "reason": "live_probe_not_requested",
            "local_function_calling_verified": False,
        }
    passed = report["offline"]["passed"] and set(report["offline"]["selected_targets"]) == set(
        config["required_local_ids"]
    )
    if "checks" in report["live"]:
        passed = passed and all(c["status"] != "failed" for c in report["live"]["checks"].values())
    if args.require_local_function_calling:
        passed = passed and report["live"]["local_function_calling_verified"]
    report["passed"] = passed
    write_report(directory / "summary.json", report)
    print(
        f"passed={passed} "
        f"real_local_function_calling={report['live']['local_function_calling_verified']}"
    )
    print(f"summary={directory / 'summary.json'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
