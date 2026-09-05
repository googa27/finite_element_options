"""Compatibility loader for Bayesian APIs retained at the estimation facade."""

from __future__ import annotations

from importlib.util import find_spec
from types import ModuleType


LEGACY_PYMC_HINT = (
    "finite-element-options 0.2 legacy PyMC calibration requires Python 3.12 and "
    "finite-element-options[calibration,bayesian]"
)


def _missing(name: str) -> bool:
    try:
        return find_spec(name) is None
    except ModuleNotFoundError:
        return True


def require_legacy_pymc_dependencies() -> None:
    """Fail before the Heston module can leak a raw optional-import error."""

    missing = [name for name in ("pandas", "pymc") if _missing(name)]
    if missing:
        raise ModuleNotFoundError(f"{LEGACY_PYMC_HINT}; missing {', '.join(missing)}")


def require_pymc() -> ModuleType:
    """Load PyMC or name the dedicated Bayesian extra needed by legacy APIs."""

    try:
        import pymc
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency-absence contract test
        raise ModuleNotFoundError(LEGACY_PYMC_HINT) from exc
    return pymc


__all__ = ["LEGACY_PYMC_HINT", "require_legacy_pymc_dependencies", "require_pymc"]
