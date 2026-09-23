"""Configuration loading for version-controlled experiment profiles."""

from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path

from edge_agent_workflow_scheduling.resources import LLMInstanceProfile, ToolReplicaProfile


def load_llm_profiles(path: str | Path) -> list[LLMInstanceProfile]:
    """Load ordered LLM instance profiles from a TOML file."""

    config_path = Path(path)
    if config_path.suffix == ".json":
        data = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        with config_path.open("rb") as config_file:
            data = tomllib.load(config_file)
    raw_profiles = data.get("llm_instances")
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise ValueError("config must contain at least one [[llm_instances]] entry")

    profiles: list[LLMInstanceProfile] = []
    seen_ids: set[str] = set()
    for index, raw_profile in enumerate(raw_profiles):
        if not isinstance(raw_profile, dict):
            raise ValueError(f"llm_instances[{index}] must be a TOML table")
        deployment = raw_profile.get("deployment_config", {})
        for field, env_field in (("model", "model_env"), ("base_url", "base_url_env")):
            env_name = deployment.get(env_field)
            if env_name and os.getenv(env_name):
                raw_profile[field] = os.environ[env_name]
        try:
            profile = LLMInstanceProfile.from_dict(raw_profile)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid llm_instances[{index}]: {exc}") from exc
        if profile.llm_id in seen_ids:
            raise ValueError(f"duplicate llm_id {profile.llm_id!r} in config")
        seen_ids.add(profile.llm_id)
        profiles.append(profile)
    return profiles


def load_llm_profile(path: str | Path, llm_id: str) -> LLMInstanceProfile:
    """Load one named LLM profile from a TOML catalog."""

    for profile in load_llm_profiles(path):
        if profile.llm_id == llm_id:
            return profile
    raise KeyError(f"llm_id {llm_id!r} was not found in {Path(path)}")


def load_tool_profiles(path: str | Path) -> list[ToolReplicaProfile]:
    """Load ordered Tool replica profiles from a TOML or JSON catalog."""

    config_path = Path(path)
    if config_path.suffix == ".json":
        data = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        with config_path.open("rb") as config_file:
            data = tomllib.load(config_file)
    raw_profiles = data.get("tool_replicas")
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise ValueError("config must contain at least one [[tool_replicas]] entry")

    profiles: list[ToolReplicaProfile] = []
    seen_ids: set[str] = set()
    for index, raw_profile in enumerate(raw_profiles):
        if not isinstance(raw_profile, dict):
            raise ValueError(f"tool_replicas[{index}] must be a TOML table")
        deployment = raw_profile.get("deployment_config", {})
        if not isinstance(deployment, dict):
            raise ValueError(f"tool_replicas[{index}].deployment_config must be a table")
        for field, env_field in (("node_id", "node_id_env"), ("platform", "platform_env")):
            env_name = deployment.get(env_field)
            if env_name and os.getenv(env_name):
                raw_profile[field] = os.environ[env_name]
        try:
            profile = ToolReplicaProfile.from_dict(raw_profile)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid tool_replicas[{index}]: {exc}") from exc
        if profile.replica_id in seen_ids:
            raise ValueError(f"duplicate replica_id {profile.replica_id!r} in config")
        seen_ids.add(profile.replica_id)
        profiles.append(profile)
    return profiles


def load_tool_profile(path: str | Path, replica_id: str) -> ToolReplicaProfile:
    """Load one named Tool replica profile from a catalog."""

    for profile in load_tool_profiles(path):
        if profile.replica_id == replica_id:
            return profile
    raise KeyError(f"replica_id {replica_id!r} was not found in {Path(path)}")
