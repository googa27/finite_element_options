"""Lazy optional dependency adoption helpers for regime-switching quanto research."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType
from typing import Literal

Maturity = Literal["boundary_only", "experimental"]


@dataclass(frozen=True)
class OptionalDependency:
    """JSON-safe optional dependency boundary descriptor."""

    module_name: str
    extra: str
    dependency: str
    install_hint: str
    maturity: Maturity


OPTIONAL_DEPENDENCIES: tuple[OptionalDependency, ...] = (
    OptionalDependency(
        module_name="arch",
        extra="volatility",
        dependency="arch>=8,<9",
        install_hint="finite-element-options[volatility]",
        maturity="experimental",
    ),
    OptionalDependency(
        module_name="ruptures",
        extra="changepoints",
        dependency="ruptures>=1.1.10,<2",
        install_hint="finite-element-options[changepoints]",
        maturity="experimental",
    ),
    OptionalDependency(
        module_name="QuantLib",
        extra="quantlib",
        dependency="QuantLib>=1.43,<2",
        install_hint="finite-element-options[quantlib]",
        maturity="boundary_only",
    ),
    OptionalDependency(
        module_name="iminuit",
        extra="identifiability",
        dependency="iminuit>=2.32,<3",
        install_hint="finite-element-options[identifiability]",
        maturity="experimental",
    ),
)

_REGISTRY_BY_MODULE = {item.module_name: item for item in OPTIONAL_DEPENDENCIES}


def optional_dependency_registry() -> tuple[OptionalDependency, ...]:
    """Return immutable descriptors for adoption-only optional dependencies."""

    return (*OPTIONAL_DEPENDENCIES,)


def require_optional(module_name: str) -> ModuleType:
    """Import an adoption optional dependency or raise an actionable extra error.

    Parameters
    ----------
    module_name:
        Top-level module name registered in :data:`OPTIONAL_DEPENDENCIES`.

    Returns
    -------
    types.ModuleType
        The imported module.  The module is intentionally returned only to
        internal adapters; public domain and result contracts stay JSON-safe.
    """

    descriptor = _REGISTRY_BY_MODULE.get(module_name)
    if descriptor is None:
        available = ", ".join(sorted(_REGISTRY_BY_MODULE))
        raise KeyError(f"unknown optional dependency {module_name!r}; available: {available}")
    try:
        return import_module(descriptor.module_name)
    except ModuleNotFoundError as exc:
        if exc.name != descriptor.module_name:
            raise
        raise ImportError(
            f"Optional dependency {descriptor.module_name!r} is required for this "
            f"regime-switching quanto adoption boundary; install "
            f"{descriptor.install_hint} ({descriptor.dependency})."
        ) from exc
