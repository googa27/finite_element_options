"""Domain specifications and named facet helpers for FEM meshes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Protocol, Sequence

import numpy as np

from finite_element_options.transform import CoordinateTransform

from math import exp, sqrt
from statistics import NormalDist


_DEFAULT_AXIS_NAMES = ("s", "v", "z")
_BOUNDARY_TOL_SCALE = 1.0e-12


@dataclass(frozen=True)
class DomainAxis:
    """One state-coordinate interval with truncation metadata."""

    name: str
    lower: float
    upper: float
    scale: str = "linear"
    truncation_policy: str = "explicit"
    tail_mass: float | None = None

    def __post_init__(self) -> None:
        """Validate interval bounds and metadata."""

        name = str(self.name).strip()
        if not name:
            raise ValueError("domain axis name must be non-empty")
        lower = float(self.lower)
        upper = float(self.upper)
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError(f"domain axis {name!r} bounds must be finite")
        if lower >= upper:
            raise ValueError(
                f"domain axis {name!r} requires lower < upper; got {lower} >= {upper}"
            )
        if self.tail_mass is not None:
            tail_mass = float(self.tail_mass)
            if not np.isfinite(tail_mass) or not 0.0 < tail_mass < 1.0:
                raise ValueError("tail_mass must lie strictly between 0 and 1")
            object.__setattr__(self, "tail_mass", tail_mass)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "scale", str(self.scale))
        object.__setattr__(self, "truncation_policy", str(self.truncation_policy))

    @property
    def min_label(self) -> str:
        """Return the canonical lower-facet label."""

        return f"{self.name}_min"

    @property
    def max_label(self) -> str:
        """Return the canonical upper-facet label."""

        return f"{self.name}_max"

    def to_public_dict(self) -> dict[str, float | str | None]:
        """Return JSON-safe public domain-axis metadata."""

        return {
            "name": self.name,
            "lower": self.lower,
            "upper": self.upper,
            "scale": self.scale,
            "truncation_policy": self.truncation_policy,
            "tail_mass": self.tail_mass,
        }


@dataclass(frozen=True)
class DomainSpec:
    """Finite tensor-product state domain with named boundary facets."""

    axes: tuple[DomainAxis, ...]
    coordinate_system: str = "physical"

    def __init__(
        self,
        axes: Iterable[DomainAxis],
        coordinate_system: str = "physical",
    ) -> None:
        """Materialize and validate a domain specification."""

        materialized = tuple(axes)
        if not materialized:
            raise ValueError("domain specification requires at least one axis")
        names = [axis.name for axis in materialized]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate domain axis name(s): {', '.join(duplicates)}")
        object.__setattr__(self, "axes", materialized)
        object.__setattr__(self, "coordinate_system", str(coordinate_system))

    @classmethod
    def from_extents(
        cls, extents: Sequence[float | Sequence[float] | DomainAxis]
    ) -> "DomainSpec":
        """Create a domain from legacy maxima, bound pairs, or axis records."""

        axes: list[DomainAxis] = []
        for axis, item in enumerate(extents):
            if isinstance(item, DomainAxis):
                axes.append(item)
                continue
            name = (
                _DEFAULT_AXIS_NAMES[axis]
                if axis < len(_DEFAULT_AXIS_NAMES)
                else f"x{axis}"
            )
            if isinstance(item, (str, bytes)):
                raise ValueError("domain extents must be numeric bounds, not strings")
            values = np.asarray(item, dtype=float)
            if values.ndim == 0:
                lower, upper = 0.0, float(values)
            elif values.shape == (2,):
                lower, upper = values
            else:
                raise ValueError("domain bound sequences must be (lower, upper) pairs")
            axes.append(DomainAxis(name, float(lower), float(upper)))
        return cls(tuple(axes))

    @property
    def dimension(self) -> int:
        """Return the number of state coordinates."""

        return len(self.axes)

    def tensor_endpoints(self) -> list[np.ndarray]:
        """Return two-point endpoint grids for each axis."""

        return [np.linspace(axis.lower, axis.upper, 2) for axis in self.axes]

    def boundary_predicates(self) -> dict[str, Callable[[np.ndarray], np.ndarray]]:
        """Return scikit-fem facet predicates keyed by canonical facet labels."""

        predicates: dict[str, Callable[[np.ndarray], np.ndarray]] = {}
        for axis_index, axis in enumerate(self.axes):
            labels = (
                (axis.min_label, axis.lower),
                (axis.max_label, axis.upper),
            )
            for label, value in labels:
                predicates[label] = _axis_boundary_predicate(axis_index, value)
        return predicates

    def transform(self, transform: CoordinateTransform) -> "DomainSpec":
        """Map physical axis bounds into transformed coordinates."""

        bounds = np.asarray(
            [[axis.lower, axis.upper] for axis in self.axes], dtype=float
        )
        transformed = transform.transform_state(bounds)
        axes = []
        for axis, values in zip(self.axes, transformed, strict=True):
            lower = float(np.min(values))
            upper = float(np.max(values))
            axes.append(
                DomainAxis(
                    axis.name,
                    lower,
                    upper,
                    scale=f"transformed:{axis.scale}",
                    truncation_policy=axis.truncation_policy,
                    tail_mass=axis.tail_mass,
                )
            )
        return DomainSpec(tuple(axes), coordinate_system="transformed")

    def to_public_dict(self) -> dict[str, object]:
        """Return JSON-safe public domain metadata."""

        return {
            "coordinate_system": self.coordinate_system,
            "dimension": self.dimension,
            "axes": [axis.to_public_dict() for axis in self.axes],
            "boundary_facets": tuple(self.boundary_predicates()),
        }


def ensure_domain_spec(
    domain: DomainSpec | Sequence[float | Sequence[float] | DomainAxis],
) -> DomainSpec:
    """Return ``domain`` as a validated :class:`DomainSpec`."""

    if isinstance(domain, DomainSpec):
        return domain
    return DomainSpec.from_extents(domain)


def attach_domain_metadata(mesh, domain: DomainSpec):
    """Attach domain metadata to a scikit-fem mesh object."""

    mesh.domain_spec = domain
    mesh.boundary_names = tuple(domain.boundary_predicates())
    return mesh


def _axis_boundary_predicate(axis_index: int, value: float):
    """Build an ``np.isclose`` predicate for one coordinate boundary."""

    atol = _BOUNDARY_TOL_SCALE * max(1.0, abs(float(value)))

    def predicate(x: np.ndarray) -> np.ndarray:
        return np.isclose(x[axis_index], value, rtol=0.0, atol=atol)

    return predicate


class _TruncationModel(Protocol):
    """Read-only scalar inputs for the existing domain truncation policy."""

    @property
    def strike(self) -> float:
        """Return the model scalar used by domain truncation."""
        ...

    @property
    def volatility(self) -> float:
        """Return the model scalar used by domain truncation."""
        ...

    @property
    def long_run_variance(self) -> float | None:
        """Return the model scalar used by domain truncation."""
        ...

    @property
    def maturity(self) -> float:
        """Return the model scalar used by domain truncation."""
        ...

    @property
    def rate(self) -> float:
        """Return the model scalar used by domain truncation."""
        ...

    @property
    def carry(self) -> float:
        """Return the model scalar used by domain truncation."""
        ...

    @property
    def variance_upper(self) -> float | None:
        """Return the model scalar used by domain truncation."""
        ...

    @property
    def vol_of_variance(self) -> float | None:
        """Return the model scalar used by domain truncation."""
        ...


def _spot_upper(model: _TruncationModel, alpha_tail: float) -> float:
    volatility = max(
        float(model.volatility), sqrt(max(float(model.long_run_variance or 0.0), 0.0))
    )
    maturity = max(float(model.maturity), 1.0e-12)
    drift = abs(float(model.rate - model.carry)) * maturity
    bounded_tail = max(min(float(alpha_tail), 0.5 - 1.0e-12), 1.0e-6)
    z_score = NormalDist().inv_cdf(1.0 - bounded_tail)
    diffusion = z_score * max(volatility, 1.0e-12) * sqrt(maturity)
    return model.strike * (1.0 + exp(drift + diffusion))


def _variance_upper(model: _TruncationModel) -> float:
    if model.variance_upper is not None:
        return float(model.variance_upper)
    base = max(float(model.long_run_variance or 0.0), float(model.volatility) ** 2)
    vol_of_variance = max(float(model.vol_of_variance or 0.0), 0.0)
    return max(
        5.0 * base + 2.0 * vol_of_variance * sqrt(max(model.maturity, 0.0)), 1.0e-10
    )


def _ui_mesh_topology_counts(dimension: int, intervals: int) -> tuple[int, int, int]:
    """Return element/node/stencil estimates aligned with ``space.mesh.create_mesh``."""

    if dimension == 1:
        return intervals, 2 * intervals + 1, 5  # ElementLineP2
    if dimension == 2:
        return 2 * intervals**2, (2 * intervals + 1) ** 2, 25  # ElementTriP2
    if dimension == 3:
        return 6 * intervals**3, (intervals + 1) ** 3, 81  # ElementTetP1
    return 6 * intervals**3, (intervals + 1) ** 3, 81
