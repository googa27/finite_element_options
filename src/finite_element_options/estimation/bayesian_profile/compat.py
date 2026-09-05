"""Compatibility loader for Bayesian APIs retained at the estimation facade."""

from __future__ import annotations

from types import ModuleType


def require_pymc() -> ModuleType:
    """Load PyMC or name the dedicated Bayesian extra needed by legacy APIs."""

    try:
        import pymc
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency-absence contract test
        raise ModuleNotFoundError(
            "finite-element-options 0.2 moves PyMC support to Python 3.12; install "
            "finite-element-options[bayesian] in that environment"
        ) from exc
    return pymc


__all__ = ["require_pymc"]
