"""Public request provenance and secret removal for provider executors."""

from __future__ import annotations

import hashlib
import json
import os


def public_llm_value(value, profile):
    secrets = [os.getenv(name) for name in profile.secret_env_vars if os.getenv(name)]

    def clean(item):
        if isinstance(item, str):
            for secret in secrets:
                item = item.replace(secret, "[REDACTED]")
            return item
        if isinstance(item, dict):
            return {
                key: "[REDACTED]"
                if key.casefold().replace("-", "_")
                in {
                    "api_key",
                    "authorization",
                    "password",
                    "secret",
                    "access_token",
                    "refresh_token",
                    "credential",
                    "credentials",
                    "bearer_token",
                }
                else clean(value)
                for key, value in item.items()
            }
        if isinstance(item, list):
            return [clean(value) for value in item]
        return item

    return clean(value)


def request_metadata(profile, call, *, parameters, tools, timeout_sec, elapsed_sec=0.0):
    summary = public_llm_value(
        {
            "model": profile.model,
            "model_parameters": parameters,
            "tool_names": [tool["name"] for tool in tools or []],
            "input_item_count": len(call.input_items),
            "estimated_input_tokens": call.input_tokens,
            "timeout_sec": timeout_sec,
        },
        profile,
    )
    digest = hashlib.sha256(
        json.dumps(summary, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    return {
        "provider": profile.provider,
        "model": profile.model,
        "request_parameters": summary,
        "request_parameters_sha256": digest,
        "energy_status": "unavailable",
        "energy_value_semantics": "legacy_schema_placeholder_not_measurement",
        "client_request_time_sec": elapsed_sec,
        "network_inclusive_request_time_sec": elapsed_sec,
        "network_only_time_sec": None,
        "timing_scope": "client_request_including_network_and_server_wait",
        "network_timing_note": "transport and server compute cannot be separated by this client",
    }
