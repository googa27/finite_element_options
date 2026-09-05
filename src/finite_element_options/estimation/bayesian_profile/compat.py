"""Compatibility loader for Bayesian APIs retained at the estimation facade."""

from __future__ import annotations

from types import ModuleType


def require_pymc() -> ModuleType:
    """Load PyMC or name the dedicated Bayesian extra needed by legacy APIs."""

    try:
        import pymc
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency-absence contract test
        raise ModuleNotFoundError(
            "install finite-element-options[calibration] on Python 3.11 or "
            "finite-element-options[bayesian] on Python 3.12"
        ) from exc
    return pymc


__all__ = ["require_pymc"]
