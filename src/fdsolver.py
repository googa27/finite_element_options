"""Finite difference solver for option pricing PDEs.

This module wraps the :mod:`findiff` package to provide a regular-grid
finite difference discretisation for the one-dimensional Black–Scholes
PDE.  The solver exposes an API similar to the finite element
``SpaceSolver`` so that either backend can be selected.

It also supplies simple Greek estimators based on the derivative
operators from :mod:`findiff`.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve
from findiff import FinDiff

from .acceleration import (
    NUMBA_AVAILABLE,
    call_payoff_grid,
    put_payoff_grid,
)


@dataclass
class FDSolver:
    """Finite difference spatial discretisation for Black–Scholes PDE.

    Parameters
    ----------
    s_grid:
        One-dimensional grid of underlying asset prices.
    dynamics:
        Model parameters supplying ``r``, ``q`` and ``sig`` attributes.
    payoff:
        Payoff object implementing ``call_payoff``/``put_payoff`` and
        ``call``/``put`` evaluation methods.
    is_call:
        ``True`` for a call option, ``False`` for a put.
    """

    s_grid: np.ndarray
    dynamics: object
    payoff: object
    is_call: bool = True

    def __post_init__(self) -> None:
        """Prepare differential operators after dataclass initialisation."""
        self.ds = float(self.s_grid[1] - self.s_grid[0])
        self.N = len(self.s_grid)
        # expose minimal interface expected by TimeStepper
        self.Vh = SimpleNamespace(N=self.N, doflocs=np.array([self.s_grid]))
        self.I = sps.identity(self.N)
        self._assemble_operator()

    # ------------------------------------------------------------------
    # Assembly helpers
    def _assemble_operator(self) -> None:
        """Assemble the spatial differential operator matrix ``L``."""
        sig = self.dynamics.sig
        r = self.dynamics.r
        q = self.dynamics.q
        s = self.s_grid

        d1 = FinDiff(0, self.ds, 1, acc=2).matrix((self.N,))
        d2 = FinDiff(0, self.ds, 2, acc=2).matrix((self.N,))

        S = sps.diags(s)
        S2 = sps.diags(s ** 2)

        self.L = (
            0.5 * sig**2 * S2 @ d2
            + (r - q) * S @ d1
            - r * self.I
        ).tocsr()

    # ------------------------------------------------------------------
    def initial_condition(self) -> np.ndarray:
        """Project the terminal payoff onto ``s_grid``.

        The payoff evaluation forms a hot loop during initialisation.  When
        available, a Numba-accelerated routine is used to compute the intrinsic
        values; otherwise we fall back to vectorised NumPy evaluation which in
        turn drops to a Python loop if the payoff does not support array
        inputs.
        """

        if self.is_call:
            if NUMBA_AVAILABLE and hasattr(self.payoff, "k"):
                return call_payoff_grid(self.s_grid, float(self.payoff.k))
            try:
                return self.payoff.call_payoff(self.s_grid)
            except Exception:
                return np.array([self.payoff.call_payoff(s) for s in self.s_grid])

        if NUMBA_AVAILABLE and hasattr(self.payoff, "k"):
            return put_payoff_grid(self.s_grid, float(self.payoff.k))
        try:
            return self.payoff.put_payoff(self.s_grid)
        except Exception:
            return np.array([self.payoff.put_payoff(s) for s in self.s_grid])

    def matrices(self, theta: float, dt: float) -> tuple[sps.csr_matrix, sps.csr_matrix]:
        """Return system matrices for the θ-scheme."""
        A = self.I - theta * dt * self.L
        B = self.I + (1 - theta) * dt * self.L
        return A.tocsr(), B.tocsr()

    def boundary_term(self, th: float) -> np.ndarray:  # pragma: no cover - no natural BC
        """Return natural boundary vector (unused for Black–Scholes)."""
        return np.zeros(self.N)

    def dirichlet(self, th: float) -> np.ndarray:
        """Return Dirichlet boundary values at time ``th``."""
        if self.is_call:
            left = 0.0
            right = self.s_grid[-1] - self.payoff.k * np.exp(-self.dynamics.r * th)
        else:  # put option
            left = self.payoff.k * np.exp(-self.dynamics.r * th)
            right = 0.0
        u = np.zeros(self.N)
        u[0] = left
        u[-1] = right
        return u

    def apply_dirichlet(self, A, b, _, u_dirichlet):
        """Apply Dirichlet boundary conditions to ``A`` and ``b``."""
        A = A.tolil()
        b = b.copy()
        A[0, :] = 0.0
        A[0, 0] = 1.0
        A[-1, :] = 0.0
        A[-1, -1] = 1.0
        b[0] = u_dirichlet[0]
        b[-1] = u_dirichlet[-1]
        return A.tocsr(), b


# ----------------------------------------------------------------------
# Greeks via finite differences

def delta(v: np.ndarray, ds: float) -> np.ndarray:
    """Compute Delta from a one-dimensional price grid ``v``."""
    return FinDiff(0, ds, 1, acc=2)(v)


def gamma(v: np.ndarray, ds: float) -> np.ndarray:
    """Compute Gamma from a one-dimensional price grid ``v``."""
    return FinDiff(0, ds, 2, acc=2)(v)


def vega(v: np.ndarray, dv: float) -> np.ndarray:
    """Compute Vega from a two-dimensional grid ``v`` (axis 1 is volatility)."""
    return FinDiff(1, dv, 1, acc=2)(v)


# ----------------------------------------------------------------------
# Convenience solver function

def solve_system(
    s_grid: np.ndarray,
    t: Iterable[float],
    dynamics,
    payoff,
    is_call: bool = True,
    theta: float = 0.5,
) -> np.ndarray:
    """Solve the Black–Scholes PDE on a regular grid.

    This is a high-level convenience wrapper returning the full time-space
    grid of option values, enabling parity with the finite element
    ``SpaceSolver`` + ``ThetaScheme`` combination.
    """

    solver = FDSolver(s_grid, dynamics, payoff, is_call=is_call)
    t = np.asarray(list(t))
    dt = t[1] - t[0]
    A, B = solver.matrices(theta, dt)

    v = np.empty((len(t), solver.N))
    v[0] = solver.initial_condition()

    for i, th in enumerate(t[:-1]):
        b = B @ v[i]
        u_d = solver.dirichlet(th + dt)
        A_enf, b_enf = solver.apply_dirichlet(A, b, [], u_d)
        v[i + 1] = spsolve(A_enf, b_enf)

    return v
