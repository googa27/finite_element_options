"""Abstraction contracts used across the pricing framework."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Protocol, TypeAlias
import numpy as np
import skfem as fem
import scipy.sparse as sps

ArrayLikeFloat: TypeAlias = float | np.ndarray


class Payoff(ABC):
    """Payoff contract for option instruments."""

    @abstractmethod
    def call_payoff(self, s: ArrayLikeFloat) -> ArrayLikeFloat:
        """Return intrinsic value of a call option at spot ``s``."""

    @abstractmethod
    def put_payoff(self, s: ArrayLikeFloat) -> ArrayLikeFloat:
        """Return intrinsic value of a put option at spot ``s``."""

    @abstractmethod
    def call(
        self, th: float, s: ArrayLikeFloat, variance: ArrayLikeFloat
    ) -> ArrayLikeFloat:
        """Return call price from variance ``sigma**2``."""

    @abstractmethod
    def put(
        self, th: float, s: ArrayLikeFloat, variance: ArrayLikeFloat
    ) -> ArrayLikeFloat:
        """Return put price from variance ``sigma**2``."""


class DynamicsModel(Protocol):
    """Protocol for models supplying PDE coefficients."""

    r: float
    q: float

    def mean_variance(
        self,
        th: ArrayLikeFloat,
        v: ArrayLikeFloat,
        config: Any | None = None,
    ) -> ArrayLikeFloat:
        """Return effective average variance over the pricing horizon.

        This method feeds finite-time boundary oracles that consume a constant
        variance over ``[t, t+th]``. Models that also need terminal moments
        should expose those via a separate method/diagnostic.
        """
        raise NotImplementedError

    def A(self, *coords) -> list[list[float]]:
        """Diffusion matrix of the state variables."""
        ...

    def dA(self, *coords) -> list[float]:
        """Divergence of the diffusion matrix."""
        ...

    def b(self, *coords) -> list[float]:
        """Drift vector of the state variables."""
        ...

    def discount(self, state: np.ndarray, time: float) -> ArrayLikeFloat:
        """Return the reaction/discount coefficient field ``c(x,t)``."""
        ...

    def source(self, state: np.ndarray, time: float) -> ArrayLikeFloat:
        """Return the running source/load field ``f(x,t)``."""
        ...

    def boundary_term(self, is_call: bool, payoff: Payoff) -> fem.LinearForm:
        """Return natural boundary contribution if available."""
        raise NotImplementedError


class StochasticDynamicsModel(DynamicsModel, Protocol):
    """Extension of :class:`DynamicsModel` with stochastic coefficients."""

    def sample(self, rng: np.random.Generator) -> DynamicsModel:
        """Return a realisation with randomised coefficients."""


class SpaceDiscretization(Protocol):
    """Protocol defining the spatial operators required by time steppers."""

    Vh: fem.CellBasis

    def initial_condition(self) -> np.ndarray:
        """Return the initial condition projected on the space."""

    def matrices(
        self,
        theta: float,
        dt: float,
        *,
        start: float | None = None,
        end: float | None = None,
    ) -> tuple[sps.csr_matrix, sps.csr_matrix]:
        """Return system matrices for the θ-scheme endpoint interval."""

    def boundary_term(self, th: float) -> np.ndarray:
        """Return natural boundary vector at time ``th``."""

    def dirichlet(self, th: float) -> np.ndarray:
        """Return Dirichlet values at time ``th``."""

    def domain_diagnostics(self, *, horizon: float, tail_mass: float = 1.0e-6) -> dict:
        """Return public domain and boundary diagnostics."""
        ...

    def apply_dirichlet(
        self,
        A,
        b,
        boundaries: Iterable[str],
        u_dirichlet: np.ndarray,
    ) -> tuple:
        """Apply Dirichlet conditions to matrix ``A`` and vector ``b``."""


class BoundaryCondition(ABC):
    """Strategy for enforcing boundary conditions."""

    @abstractmethod
    def apply(self, space: SpaceDiscretization, A, b, th: float) -> tuple:
        """Return ``(A, b)`` after applying the boundary condition."""
