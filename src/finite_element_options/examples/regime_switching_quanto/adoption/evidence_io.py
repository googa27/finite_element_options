"""Canonical JSON and hash IO helpers for adoption evidence artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from finite_element_options.examples.regime_switching_quanto._types import json_safe


def canonical_json(payload: Any) -> str:
    """Serialize payload as deterministic JSON with no local path dependence."""

    return json.dumps(json_safe(payload), sort_keys=True, separators=(",", ":"))


def canonical_json_sha256(payload: Any) -> str:
    """Return SHA-256 of :func:`canonical_json`."""

    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return SHA-256 of a file without storing its path."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_atomic_json(path: str | Path, payload: Any) -> str:
    """Atomically write canonical JSON and return the artifact hash."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = canonical_json(payload) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
    finally:
        temporary_path.unlink(missing_ok=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = [
    "canonical_json",
    "canonical_json_sha256",
    "file_sha256",
    "write_atomic_json",
]
