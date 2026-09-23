#!/usr/bin/env python3
"""Validate the complete local RL resource catalog."""

from __future__ import annotations

import argparse
import tomllib
from collections import Counter
from pathlib import Path

from edge_agent_workflow_scheduling.config import load_llm_profiles, load_tool_profiles


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/rl_resources_arch_local_v1.toml"),
    )
    args = parser.parse_args()
    root = Path.cwd()
    raw = tomllib.loads(args.config.read_text(encoding="utf-8"))

    llm_path = root / raw["llm_profile_catalog"]
    tool_path = root / raw["tool_profile_catalog"]
    llms = load_llm_profiles(llm_path)
    tools = load_tool_profiles(tool_path)
    llm_ids = {profile.llm_id for profile in llms}
    missing_local = set(raw["required_local_llm_ids"]) - llm_ids
    missing_cloud = set(raw["required_cloud_llm_ids"]) - llm_ids
    if missing_local or missing_cloud:
        raise SystemExit(f"missing LLM profiles: local={sorted(missing_local)}, cloud={sorted(missing_cloud)}")

    counts = Counter(profile.tool_name for profile in tools)
    required_tools = set(raw["required_tool_names"])
    missing_tools = required_tools - counts.keys()
    too_few = {
        tool_name: counts[tool_name]
        for tool_name in required_tools
        if counts[tool_name] < raw["minimum_replicas_per_tool"]
    }
    if missing_tools or too_few:
        raise SystemExit(
            f"invalid Tool catalog: missing={sorted(missing_tools)}, too_few={too_few}"
        )

    cloud = raw["cloud_api"]
    if cloud["api_key_env"] != "ARK_API_KEY":
        raise SystemExit("cloud API key must use ARK_API_KEY")
    print(f"resource_catalog={raw['resource_catalog_id']}")
    print(f"llm_profiles={len(llms)} local={sorted(raw['required_local_llm_ids'])} cloud={sorted(raw['required_cloud_llm_ids'])}")
    print(f"tool_replicas={len(tools)} per_tool={dict(sorted(counts.items()))}")
    print(f"cloud_base_url={cloud['base_url']} api_key_env={cloud['api_key_env']}")
    print("status=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
