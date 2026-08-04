"""Configuration loading for version-controlled experiment profiles."""

from __future__ import annotations

import tomllib
from pathlib import Path

from edge_agent_workflow_scheduling.resources import LLMInstanceProfile


def load_llm_profiles(path: str | Path) -> list[LLMInstanceProfile]:
    """Load ordered LLM instance profiles from a TOML file."""

    config_path = Path(path)
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
