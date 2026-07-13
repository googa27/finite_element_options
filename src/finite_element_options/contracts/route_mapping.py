"""Route-mapping coercions for public FEM capability diagnostics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def state_dimension(value: Any) -> int:
    """Infer a positive state dimension from public state-variable metadata."""

    if isinstance(value, str):
        return 1
    if isinstance(value, Iterable) and not isinstance(value, Mapping):
        values = tuple(value)
        return len(values) or 1
    return 1


def coerce_dimension(value: Any) -> int:
    """Return an integer route dimension, or -1 for fail-closed diagnostics."""

    if isinstance(value, bool):
        return -1
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        return int(text) if text.isdigit() else -1
    if isinstance(value, Iterable) and not isinstance(value, Mapping):
        return state_dimension(value)
    return -1


__all__ = ["coerce_dimension", "state_dimension"]
