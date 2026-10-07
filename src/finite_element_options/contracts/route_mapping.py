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


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_present(
    mapping: Mapping[str, Any], keys: tuple[str, ...], *, default: Any
) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _tuple_of_strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable) and not isinstance(value, Mapping):
        return tuple(str(item) for item in value)
    return (str(value),)


def _boundary_condition_classes(value: Any) -> tuple[str, ...]:
    """Normalize public schema boundary formulas to FEM capability classes."""

    if isinstance(value, Mapping):
        raw_items: Iterable[tuple[str, Any]] = value.items()
    else:
        raw_items = (("", item) for item in _tuple_of_strings(value))

    classes: list[str] = []
    for location, item in raw_items:
        text = str(item).lower().replace("-", "_")
        location_text = str(location).lower().replace("-", "_")
        if "free" in text and "boundary" in text:
            boundary_class = "free_boundary"
        elif "robin" in text:
            boundary_class = "robin"
        elif "neumann" in text or "slope" in text:
            boundary_class = "neumann"
        elif (
            "dirichlet" in text or "absorbing" in text or text.strip() in {"0", "zero"}
        ):
            boundary_class = "dirichlet"
        elif ("linear" in text or "growth" in text) and any(
            marker in location_text
            for marker in (
                "s=0",
                "s_min",
                "lower",
                "left",
                "s_max",
                "upper",
                "right",
                "far_field",
            )
        ):
            boundary_class = "dirichlet"
        else:
            boundary_class = text
        if boundary_class not in classes:
            classes.append(boundary_class)
    return tuple(classes)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None
