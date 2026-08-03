"""Deterministic hashing and minimal secret redaction for experiment artifacts."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any

_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "bearer_token",
        "credential",
        "credentials",
        "access_token",
        "password",
        "passwd",
        "refresh_token",
        "secret",
    }
)


def canonical_json(value: Any) -> str:
    """Serialize a JSON value deterministically for hashing and comparison."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def content_digest(value: Any) -> str:
    """Return a SHA-256 digest for a JSON-serializable value."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sanitize_for_trace(value: Any) -> Any:
    """Deep-copy a JSON value while replacing values stored under secret keys."""

    if isinstance(value, dict):
        return {key: _sanitize_mapping_item(key, item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_trace(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_trace(item) for item in value]
    return deepcopy(value)


def _is_sensitive_key(key: object) -> bool:
    if not isinstance(key, str):
        return False
    normalized = key.casefold().replace("-", "_")
    return normalized in _SENSITIVE_KEYS or normalized.endswith(
        ("_api_key", "_password", "_secret")
    )


def _sanitize_mapping_item(key: str, value: Any) -> Any:
    if _is_sensitive_key(key):
        return _redacted(value)
    if key in {"arguments", "output"} and isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            pass
        else:
            return canonical_json(sanitize_for_trace(parsed))
    return sanitize_for_trace(value)


def _redacted(value: Any) -> str:
    return f"[REDACTED sha256:{content_digest(value)}]"
