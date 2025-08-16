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


class Mapping(Protocol):
    """Protocol for basic forward and inverse mappings."""

    # The following methods define the expected interface and are not
    # executed directly; hence they are excluded from coverage metrics.
    def transform(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover
        """Map ``x`` from the physical to the transformed domain."""

    def untransform(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover
        """Map ``x`` from the transformed back to the physical domain."""


@dataclass
class Identity:
    """Trivial mapping leaving values unchanged."""

    def transform(self, x: np.ndarray) -> np.ndarray:
        return x

    def untransform(self, x: np.ndarray) -> np.ndarray:
        return x


@dataclass
class LogPrice:
    """Logarithmic mapping for underlying price.

    ``s`` denotes the spot price.  The forward transform returns
    ``log(s)`` and the inverse applies the exponential map.
    """

    def transform(self, s: np.ndarray) -> np.ndarray:
        return np.log(s)

    def untransform(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)


@dataclass
class SqrtVol:
    """Square-root mapping for (co)variance variables."""

    def transform(self, v: np.ndarray) -> np.ndarray:
        return np.sqrt(v)

    def untransform(self, y: np.ndarray) -> np.ndarray:
        return y ** 2


@dataclass
class TimeToMaturity:
    """Reverse-time mapping turning absolute time into time-to-maturity."""

    maturity: float

    def transform(self, t: np.ndarray) -> np.ndarray:
        """Return the time-to-maturity ``T - t``."""

        return self.maturity - t

    def untransform(self, tau: np.ndarray) -> np.ndarray:
        """Recover the original time from time-to-maturity ``tau``."""

        return self.maturity - tau


@dataclass
class CoordinateTransform:
    """Composite transformation for price, volatility and time.

    Parameters default to :class:`Identity` when not supplied.
    """

    price: Mapping = field(default_factory=Identity)
    vol: Mapping = field(default_factory=Identity)
    time: Mapping = field(default_factory=Identity)

    def transform_state(self, x: np.ndarray) -> np.ndarray:
        """Transform spatial coordinates.

        ``x`` is expected to have shape ``(dim, n)`` with the first row
        representing the underlying price and, when present, the second
        row the volatility/variance coordinate.
        """

        out = x.copy()
        out[0] = self.price.transform(out[0])
        if out.shape[0] > 1:
            out[1] = self.vol.transform(out[1])
        return out

    def untransform_state(self, x: np.ndarray) -> np.ndarray:
        """Inverse mapping of :meth:`transform_state`."""

        out = x.copy()
        out[0] = self.price.untransform(out[0])
        if out.shape[0] > 1:
            out[1] = self.vol.untransform(out[1])
        return out

    def transform_time(self, t: np.ndarray) -> np.ndarray:
        """Transform time variable."""

        return self.time.transform(t)

    def untransform_time(self, tau: np.ndarray) -> np.ndarray:
        """Inverse mapping of :meth:`transform_time`."""

        return self.time.untransform(tau)
