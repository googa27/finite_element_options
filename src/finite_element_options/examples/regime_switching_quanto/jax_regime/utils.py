"""Shared lazy JAX loading and JSON conversion for the regime study."""

from __future__ import annotations

from typing import Any


def stack() -> tuple[Any, Any, Any]:
    """Load the isolated JAX stack and enable x64 numerical evidence."""

    try:
        import jax
        import jax.numpy as jnp
        import jax.random as jr
    except ModuleNotFoundError as exc:
        raise ImportError("JAX regime study requires finite-element-options[jax-regime].") from exc
    jax.config.update("jax_enable_x64", True)
    return jax, jnp, jr


def to_python(value: Any) -> Any:
    """Recursively convert JAX/NumPy-style values to JSON-safe Python values."""

    if isinstance(value, dict):
        return {str(key): to_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_python(item) for item in value]
    if hasattr(value, "tolist"):
        return to_python(value.tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
