"""Shared local artifact URI resolution helpers."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse


def resolve_local_path(uri: str, local_root: Path) -> Path:
    """Resolve file, local, or plain path URIs to a local path."""

    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    if parsed.scheme == "local":
        raw_path = unquote(f"{parsed.netloc}{parsed.path}")
        return local_root / raw_path.lstrip("/")
    if parsed.scheme:
        raise ValueError(f"unsupported input URI scheme {parsed.scheme!r}")
    path = Path(uri)
    return path if path.is_absolute() else local_root / path
