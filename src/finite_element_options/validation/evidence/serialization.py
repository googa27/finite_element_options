"""Canonical JSON, hashing, and numeric normalization for validation evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
import hashlib
from importlib.metadata import distribution
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, cast

import numpy as np


def json_safe(value: Any) -> Any:
    """Convert dataclasses, arrays, NumPy scalars, and timestamps to JSON values."""

    if is_dataclass(value):
        return json_safe(asdict(cast(Any, value)))
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.datetime64):
        return str(value.astype("datetime64[D]"))
    if isinstance(value, (datetime, date)):
        if type(value).__name__ == "NaTType":
            return None
        return value.date().isoformat() if isinstance(value, datetime) else value.isoformat()
    return value


def canonical_json(payload: Any) -> str:
    """Serialize payload as deterministic JSON with no local path dependence."""

    return json.dumps(json_safe(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_json_sha256(payload: Any) -> str:
    """Return SHA-256 of :func:`canonical_json`."""

    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def quantize_json_floats(payload: Any, *, significant_digits: int = 10) -> Any:
    """Normalize finite JSON floats to stable cross-platform significant digits."""

    if not 6 <= significant_digits <= 17:
        raise ValueError("significant_digits must be within [6, 17]")
    value = json_safe(payload)
    if isinstance(value, dict):
        return {
            str(key): quantize_json_floats(item, significant_digits=significant_digits)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [quantize_json_floats(item, significant_digits=significant_digits) for item in value]
    if isinstance(value, bool) or not isinstance(value, float):
        return value
    if not math.isfinite(value):
        raise ValueError("quantized evidence floats must be finite")
    normalized = float(format(value, f".{significant_digits}g"))
    return 0.0 if normalized == 0.0 else normalized


def quantize_upper_bound(value: float, *, significant_digits: int = 10) -> float:
    """Round a non-negative bound outward at the declared significant-digit precision."""

    number = float(value)
    if not 6 <= significant_digits <= 17:
        raise ValueError("significant_digits must be within [6, 17]")
    if not math.isfinite(number) or number < 0.0:
        raise ValueError("upper bound must be finite and non-negative")
    if number == 0.0:
        return 0.0
    exponent = math.floor(math.log10(number))
    quantum = 10.0 ** (exponent - significant_digits + 1)
    return math.ceil(number / quantum) * quantum


def distribution_install_mode(name: str) -> str:
    """Classify an installed distribution as wheel, editable, or unknown."""

    direct_url_text = distribution(name).read_text("direct_url.json")
    if direct_url_text is None:
        return "wheel"
    try:
        direct_url = json.loads(direct_url_text)
    except json.JSONDecodeError:
        return "unknown"
    if direct_url.get("dir_info", {}).get("editable") is True:
        return "editable"
    return "wheel"


def file_sha256(path: str | Path) -> str:
    """Return SHA-256 of a file without storing its path."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_atomic_json(path: str | Path, payload: Any) -> str:
    """Atomically write canonical JSON and return the artifact hash."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json(payload) + "\n"
    descriptor, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
    return file_sha256(target)


__all__ = [
    "canonical_json",
    "canonical_json_sha256",
    "distribution_install_mode",
    "file_sha256",
    "json_safe",
    "quantize_json_floats",
    "quantize_upper_bound",
    "write_atomic_json",
]
