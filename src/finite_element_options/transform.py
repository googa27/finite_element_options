"""Coordinate transformations for option pricing models.

This module defines lightweight transformation classes used to map
between physical variables and the coordinates employed by numerical
solvers.  Each transformation exposes :py:meth:`transform` and
:py:meth:`untransform` methods implementing the forward and inverse
mappings respectively.  Transformations are composable through the
:class:`CoordinateTransform` helper which handles state (price and
volatility) as well as time variables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


def _as_float_array(name: str, value) -> np.ndarray:
    """Return ``value`` as a finite floating array."""

    array = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _match_input_shape(value, array: np.ndarray):
    """Return a Python float for scalar inputs and an array otherwise."""

    if np.asarray(value).ndim == 0:
        return float(np.asarray(array, dtype=float))
    return array


def _require_strictly_positive(name: str, value) -> np.ndarray:
    """Return ``value`` as a finite array with strictly positive entries."""

    array = _as_float_array(name, value)
    if np.any(array <= 0.0):
        raise ValueError(f"{name} requires strictly positive physical values")
    return array


class Mapping(Protocol):
    """Protocol for one-dimensional forward/inverse coordinate mappings."""

    # The following methods define the expected interface and are not
    # executed directly; hence they are excluded from coverage metrics.
    def transform(self, x: float | np.ndarray) -> float | np.ndarray:  # pragma: no cover
        """Map ``x`` from the physical to the transformed domain."""
        ...

    def untransform(self, x: float | np.ndarray) -> float | np.ndarray:  # pragma: no cover
        """Map ``x`` from the transformed back to the physical domain."""
        ...

    def derivative(self, x: float | np.ndarray) -> float | np.ndarray:  # pragma: no cover
        r"""Return ``dy/dx`` for the physical-to-transformed map."""
        ...

    def second_derivative(self, x: float | np.ndarray) -> float | np.ndarray:  # pragma: no cover
        r"""Return ``d²y/dx²`` for the physical-to-transformed map."""
        ...


@dataclass
class Identity:
    """Trivial mapping leaving values unchanged."""

    def transform(self, x: float | np.ndarray) -> float | np.ndarray:
        """Return ``x`` unchanged."""
        return _match_input_shape(x, _as_float_array("identity coordinate", x))

    def untransform(self, x: float | np.ndarray) -> float | np.ndarray:
        """Return ``x`` unchanged."""
        return _match_input_shape(x, _as_float_array("identity coordinate", x))

    def derivative(self, x: float | np.ndarray) -> float | np.ndarray:
        r"""Return ``dy/dx = 1`` for the identity map."""
        return np.ones_like(_as_float_array("identity coordinate", x), dtype=float)

    def second_derivative(self, x: float | np.ndarray) -> float | np.ndarray:
        r"""Return ``d²y/dx² = 0`` for the identity map."""
        return np.zeros_like(_as_float_array("identity coordinate", x), dtype=float)


@dataclass
class LogPrice:
    """Logarithmic mapping for underlying price.

    ``s`` denotes the spot price.  The forward transform returns
    ``log(s)`` and the inverse applies the exponential map.
    """

    def transform(self, s: float | np.ndarray) -> float | np.ndarray:
        r"""Map strictly positive spot price ``s`` to log space ``\log s``."""
        return _match_input_shape(s, np.log(_require_strictly_positive("log-price transform", s)))

    def untransform(self, x: float | np.ndarray) -> float | np.ndarray:
        """Recover price ``s = e^x`` from log space."""
        return _match_input_shape(x, np.exp(_as_float_array("log-price coordinate", x)))

    def derivative(self, s: float | np.ndarray) -> float | np.ndarray:
        r"""Return ``d log(s) / ds = 1 / s``."""
        spot = _require_strictly_positive("log-price transform", s)
        return 1.0 / spot

    def second_derivative(self, s: float | np.ndarray) -> float | np.ndarray:
        r"""Return ``d² log(s) / ds² = -1 / s²``."""
        spot = _require_strictly_positive("log-price transform", s)
        return -1.0 / np.square(spot)


@dataclass
class SqrtVol:
    """Square-root mapping for (co)variance variables."""

    def transform(self, v: float | np.ndarray) -> float | np.ndarray:
        r"""Map non-negative variance ``v`` to its root ``\sqrt{v}``."""
        variance = _as_float_array("sqrt-variance transform", v)
        if np.any(variance < 0.0):
            raise ValueError("sqrt-variance transform requires non-negative values")
        return _match_input_shape(v, np.sqrt(variance))

    def untransform(self, y: float | np.ndarray) -> float | np.ndarray:
        """Square a non-negative root coordinate to recover variance ``v = y^2``."""
        root_variance = _as_float_array("sqrt-variance coordinate", y)
        if np.any(root_variance < 0.0):
            raise ValueError("sqrt-variance coordinate requires non-negative values")
        return _match_input_shape(y, np.square(root_variance))

    def derivative(self, v: float | np.ndarray) -> float | np.ndarray:
        r"""Return ``d sqrt(v) / dv = 1/(2 sqrt(v))``."""
        variance = _require_strictly_positive("sqrt-variance transform", v)
        return 0.5 / np.sqrt(variance)

    def second_derivative(self, v: float | np.ndarray) -> float | np.ndarray:
        r"""Return ``d² sqrt(v) / dv² = -1/(4 v^{3/2})``."""
        variance = _require_strictly_positive("sqrt-variance transform", v)
        return -0.25 / np.power(variance, 1.5)


@dataclass
class TimeToMaturity:
    """Reverse-time mapping turning absolute time into time-to-maturity."""

    maturity: float

    def transform(self, t: float | np.ndarray) -> float | np.ndarray:
        """Return the time-to-maturity ``T - t``."""

        return _match_input_shape(t, self.maturity - _as_float_array("time coordinate", t))

    def untransform(self, tau: float | np.ndarray) -> float | np.ndarray:
        """Recover the original time from time-to-maturity ``tau``."""

        return _match_input_shape(
            tau,
            self.maturity - _as_float_array("time-to-maturity coordinate", tau),
        )


@dataclass
class CoordinateTransform:
    """Composite transformation for price, volatility and time.

    Parameters default to :class:`Identity` when not supplied.  State mappings
    are componentwise.  For a transformed coordinate ``y = h(x)`` and physical
    generator

    ``0.5 A_ij(x) d²/dx_i dx_j + b_i(x) d/dx_i - r``,

    :meth:`transformed_coefficients` returns the Itô/chain-rule coefficients in
    ``y`` coordinates:

    ``A_y = Dh A Dh.T`` and
    ``b_y_i = h'_i b_i + 0.5 h''_i A_ii``.

    The returned diffusion divergence is the row divergence of ``A_y`` with
    respect to transformed coordinates, so the existing weak form can still use
    ``mu = b_y - 0.5 div(A_y)``.

    Custom mappings used for PDE assembly must provide ``derivative`` and
    ``second_derivative`` methods; mappings without those methods can still be
    used for pure coordinate round trips but cannot define transformed
    generator coefficients safely.
    """

    price: Mapping = field(default_factory=Identity)
    vol: Mapping = field(default_factory=Identity)
    time: Mapping = field(default_factory=Identity)

    def _mappings_for_dimension(self, dim: int) -> list[Mapping]:
        """Return component mappings for ``dim`` spatial coordinates."""

        if dim < 1:
            raise ValueError("state coordinates must contain at least one dimension")
        mappings: list[Mapping] = [self.price]
        if dim > 1:
            mappings.append(self.vol)
        mappings.extend(Identity() for _ in range(max(0, dim - len(mappings))))
        return mappings

    @staticmethod
    def _state_array(name: str, x: np.ndarray) -> np.ndarray:
        """Validate a state-coordinate array with shape ``(dim, npoints...)``."""

        array = _as_float_array(name, x)
        if array.ndim == 0:
            raise ValueError(f"{name} must include a coordinate dimension")
        return array

    @staticmethod
    def _mapping_derivatives(mapping: Mapping, physical_component: np.ndarray):
        """Return first and second derivatives for a coordinate mapping."""

        try:
            first = mapping.derivative(physical_component)
            second = mapping.second_derivative(physical_component)
        except AttributeError as exc:
            raise TypeError(
                "coordinate mappings used for PDE assembly must define "
                "derivative and second_derivative methods"
            ) from exc
        return np.asarray(first, dtype=float), np.asarray(second, dtype=float)

    def transform_state(self, x: np.ndarray) -> np.ndarray:
        """Transform spatial coordinates.

        ``x`` is expected to have shape ``(dim, n)`` with the first row
        representing the underlying price and, when present, the second
        row the volatility/variance coordinate.
        """

        out = self._state_array("physical state", x).copy()
        for axis, mapping in enumerate(self._mappings_for_dimension(out.shape[0])):
            out[axis] = mapping.transform(out[axis])
        return out

    def untransform_state(self, x: np.ndarray) -> np.ndarray:
        """Inverse mapping of :meth:`transform_state`."""

        out = self._state_array("transformed state", x).copy()
        for axis, mapping in enumerate(self._mappings_for_dimension(out.shape[0])):
            out[axis] = mapping.untransform(out[axis])
        return out

    def validate_transformed_state_domain(self, x: np.ndarray) -> None:
        """Fail closed when transformed mesh nodes lie outside mapping support."""

        physical_state = self.untransform_state(x)
        mappings = self._mappings_for_dimension(int(physical_state.shape[0]))
        for axis, mapping in enumerate(mappings):
            first, second = self._mapping_derivatives(mapping, physical_state[axis])
            if np.any(~np.isfinite(first)) or np.any(~np.isfinite(second)):
                raise ValueError("coordinate transform derivatives must be finite")
            if np.any(first == 0.0):
                raise ValueError("coordinate transform derivatives must be non-zero")

    def transformed_coefficients(self, dynamics, transformed_state: np.ndarray):
        """Return generator coefficients in transformed coordinates.

        The formula supports componentwise transforms.  It is exact for the
        current identity, log-price and square-root-variance maps and fails
        closed at singular transformed boundaries such as ``sqrt(v)=0``.
        """

        physical_state = self.untransform_state(transformed_state)
        dim = int(physical_state.shape[0])
        mappings = self._mappings_for_dimension(dim)
        derivatives = [
            self._mapping_derivatives(mapping, physical_state[i])
            for i, mapping in enumerate(mappings)
        ]
        first = [item[0] for item in derivatives]
        second = [item[1] for item in derivatives]
        if any(np.any(~np.isfinite(derivative)) for derivative in first + second):
            raise ValueError("coordinate transform derivatives must be finite")
        if any(np.any(derivative == 0.0) for derivative in first):
            raise ValueError("coordinate transform derivatives must be non-zero")

        physical_diffusion = dynamics.A(*physical_state)
        physical_divergence = dynamics.dA(*physical_state)
        physical_drift = dynamics.b(*physical_state)
        if len(physical_diffusion) != dim or len(physical_drift) != dim:
            raise ValueError("dynamics coefficient dimension does not match state dimension")

        diffusion = [
            [
                first[i] * first[j] * np.asarray(physical_diffusion[i][j], dtype=float)
                for j in range(dim)
            ]
            for i in range(dim)
        ]
        drift = [
            first[i] * np.asarray(physical_drift[i], dtype=float)
            + 0.5 * second[i] * np.asarray(physical_diffusion[i][i], dtype=float)
            for i in range(dim)
        ]
        diffusion_divergence = []
        for i in range(dim):
            row_divergence = first[i] * np.asarray(physical_divergence[i], dtype=float)
            row_divergence = row_divergence + second[i] * np.asarray(
                physical_diffusion[i][i], dtype=float
            )
            for j in range(dim):
                row_divergence = row_divergence + (
                    first[i]
                    * second[j]
                    / first[j]
                    * np.asarray(physical_diffusion[i][j], dtype=float)
                )
            diffusion_divergence.append(row_divergence)
        return diffusion, diffusion_divergence, drift

    def transform_time(self, t: float | np.ndarray) -> float | np.ndarray:
        """Transform time variable."""

        return self.time.transform(t)

    def untransform_time(self, tau: float | np.ndarray) -> float | np.ndarray:
        """Inverse mapping of :meth:`transform_time`."""

        return self.time.untransform(tau)
