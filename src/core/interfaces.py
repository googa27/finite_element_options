"""Abstraction contracts used across the pricing framework."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Protocol
import numpy as np
import skfem as fem
import scipy.sparse as sps


class Payoff(ABC):
    """Payoff contract for option instruments."""

    @abstractmethod
    def call_payoff(self, s: float) -> float:
        """Return intrinsic value of a call option at spot ``s``."""

    @abstractmethod
    def put_payoff(self, s: float) -> float:
        """Return intrinsic value of a put option at spot ``s``."""

    @abstractmethod
    def call(self, th: float, s: float, v: float) -> float:
        """Return price of the call option."""

    @abstractmethod
    def put(self, th: float, s: float, v: float) -> float:
        """Return price of the put option."""


class DynamicsModel(Protocol):
    """Protocol for models supplying PDE coefficients."""

    r: float
    q: float

    def mean_variance(self, th: float, v: float) -> float:
        """Return expected variance at ``t+th`` given state ``v``."""

    def A(self, *coords) -> list[list[float]]:
        """Diffusion matrix of the state variables."""

    def dA(self, *coords) -> list[float]:
        """Divergence of the diffusion matrix."""

    def b(self, *coords) -> list[float]:
        """Drift vector of the state variables."""

    def boundary_term(self, is_call: bool, payoff: Payoff) -> fem.LinearForm:
        """Return natural boundary contribution if available."""
        raise NotImplementedError


class SpaceDiscretization(Protocol):
    """Protocol defining the spatial operators required by time steppers."""

    Vh: fem.CellBasis

    def initial_condition(self) -> np.ndarray:
        """Return the initial condition projected on the space."""

    def matrices(self, theta: float, dt: float) -> tuple[sps.csr_matrix, sps.csr_matrix]:
        """Return system matrices for the θ-scheme."""

    def boundary_term(self, th: float) -> np.ndarray:
        """Return natural boundary vector at time ``th``."""

    def dirichlet(self, th: float) -> np.ndarray:
        """Return Dirichlet values at time ``th``."""

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
