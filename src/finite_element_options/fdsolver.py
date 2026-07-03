"""Auditable 1D finite-difference reference for Black-Scholes options.

This module intentionally stays narrow.  The reference backend is valid only for
a one-dimensional Black-Scholes PDE on a finite, strictly increasing, uniform
spot grid and a uniform forward time-to-maturity grid ``tau``.  Unsupported
routes fail before discretisation so FEM comparisons do not inherit hidden
finite-difference defects.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Iterable

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from xarray import DataArray

from .data_utils import snapshot


_GRID_ATOL = 1.0e-12
_GRID_RTOL = 1.0e-10


@dataclass
class FDSolver:
    """Finite-difference spatial discretisation for the Black-Scholes PDE.

    Parameters
    ----------
    s_grid:
        One-dimensional, finite, non-negative, strictly increasing and uniform
        spot grid.  Nonuniform grids are rejected rather than discretised with
        incorrect uniform stencils.
    dynamics:
        Black-Scholes parameters supplying finite ``r``, ``q`` and positive
        ``sig`` attributes.
    payoff:
        Payoff object implementing ``call_payoff``/``put_payoff`` and carrying
        a strike ``k``.
    is_call:
        ``True`` for a call option, ``False`` for a put.
    """

    s_grid: np.ndarray
    dynamics: object
    payoff: object
    is_call: bool = True

    def __post_init__(self) -> None:
        """Validate inputs and assemble the reference tridiagonal operator."""
        self.s_grid = _validate_uniform_spot_grid(self.s_grid)
        self.ds = float(self.s_grid[1] - self.s_grid[0])
        self.N = len(self.s_grid)
        self._validate_black_scholes_parameters()
        self._validate_payoff_contract()
        # expose minimal interface expected by TimeStepper
        self.Vh = SimpleNamespace(N=self.N, doflocs=np.array([self.s_grid]))
        self.I = sps.identity(self.N, format="csr")
        self._assemble_operator()

    # ------------------------------------------------------------------
    # Assembly helpers
    def _validate_black_scholes_parameters(self) -> None:
        for name in ("r", "q", "sig"):
            value = getattr(self.dynamics, name, None)
            if value is None or not np.isfinite(float(value)):
                raise ValueError(f"Black-Scholes dynamics must define finite {name}")
        if float(self.dynamics.sig) <= 0.0:
            raise ValueError("Black-Scholes volatility sig must be positive")

    def _validate_payoff_contract(self) -> None:
        if not hasattr(self.payoff, "k"):
            raise ValueError("Black-Scholes FD reference payoff must expose strike k")
        if not np.isfinite(float(self.payoff.k)) or float(self.payoff.k) <= 0.0:
            raise ValueError("Black-Scholes FD reference strike k must be finite and positive")
        method = "call_payoff" if self.is_call else "put_payoff"
        if not callable(getattr(self.payoff, method, None)):
            raise ValueError(f"Black-Scholes FD reference payoff must define {method}")

    def _assemble_operator(self) -> None:
        """Assemble the spatial operator ``L`` using central differences.

        ``L`` represents

        ``0.5 σ² S² V_SS + (r-q) S V_S - r V``

        on interior grid nodes.  Boundary rows are left as zero rows because
        endpoint values are supplied through explicit Dirichlet elimination.
        """
        sigma = float(self.dynamics.sig)
        rate = float(self.dynamics.r)
        carry = float(self.dynamics.q)
        ds = self.ds
        lower = np.zeros(self.N, dtype=float)
        diag = np.zeros(self.N, dtype=float)
        upper = np.zeros(self.N, dtype=float)

        for i in range(1, self.N - 1):
            spot = self.s_grid[i]
            diffusion = 0.5 * sigma * sigma * spot * spot
            drift = (rate - carry) * spot
            lower[i] = diffusion / ds**2 - drift / (2.0 * ds)
            diag[i] = -2.0 * diffusion / ds**2 - rate
            upper[i] = diffusion / ds**2 + drift / (2.0 * ds)

        self.L = sps.diags(
            diagonals=[lower[1:], diag, upper[:-1]],
            offsets=[-1, 0, 1],
            shape=(self.N, self.N),
            format="csr",
        )

    # ------------------------------------------------------------------
    def initial_condition(self) -> np.ndarray:
        """Evaluate the terminal payoff on ``s_grid``.

        NumPy-aware payoff functions may return the full grid at once.  A scalar
        fallback is allowed only for explicit scalar-only type errors or the
        standard NumPy "ambiguous truth value" error.  Domain/value errors raised
        by vectorized payoff code propagate instead of being hidden.
        """

        payoff_fn = getattr(self.payoff, "call_payoff" if self.is_call else "put_payoff")
        return _evaluate_payoff_grid(payoff_fn, self.s_grid)

    def matrices(self, theta: float, dt: float) -> tuple[sps.csr_matrix, sps.csr_matrix]:
        """Return system matrices for the theta-scheme."""
        theta = float(theta)
        dt = float(dt)
        if not np.isfinite(theta) or theta < 0.0 or theta > 1.0:
            raise ValueError("theta must lie in [0, 1]")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        a_matrix = self.I - theta * dt * self.L
        b_matrix = self.I + (1.0 - theta) * dt * self.L
        return a_matrix.tocsr(), b_matrix.tocsr()

    def boundary_term(self, th: float) -> np.ndarray:  # pragma: no cover - no natural BC
        """Return natural boundary vector (unused for Black-Scholes)."""
        return np.zeros(self.N)

    def dirichlet(self, th: float) -> np.ndarray:
        """Return endpoint Dirichlet values at time-to-maturity ``th``.

        ``th`` is the forward transformed time ``tau = T - t``.  The high-call
        boundary uses the affine Black-Scholes far-field
        ``S_max exp(-q tau) - K exp(-r tau)``; the low-put boundary uses
        ``K exp(-r tau)``.
        """
        tau = float(th)
        if not np.isfinite(tau) or tau < 0.0:
            raise ValueError("Dirichlet time-to-maturity th must be finite and non-negative")
        strike = float(self.payoff.k)
        rate = float(self.dynamics.r)
        carry = float(self.dynamics.q)
        values = np.zeros(self.N, dtype=float)
        if self.is_call:
            values[0] = 0.0
            values[-1] = self.s_grid[-1] * np.exp(-carry * tau) - strike * np.exp(-rate * tau)
        else:
            values[0] = strike * np.exp(-rate * tau)
            values[-1] = 0.0
        return values

    def apply_dirichlet(self, A, b, _, u_dirichlet):
        """Apply endpoint Dirichlet conditions by algebraic elimination.

        Boundary columns are removed from interior equations and their known
        values are subtracted from the right-hand side.  Boundary rows are then
        replaced by identity equations.  This form makes the correction explicit
        for tests and keeps the matrix invariant across time steps.
        """
        matrix = A.tolil(copy=True)
        rhs = np.asarray(b, dtype=float).copy()
        u = np.asarray(u_dirichlet, dtype=float)
        if u.shape != (self.N,):
            raise ValueError("u_dirichlet must have the same length as s_grid")

        for index in (0, self.N - 1):
            column = np.asarray(matrix[:, index].toarray()).ravel()
            rhs -= column * u[index]

        for index in (0, self.N - 1):
            matrix[:, index] = 0.0
            matrix[index, :] = 0.0
            matrix[index, index] = 1.0
            rhs[index] = u[index]

        return matrix.tocsr(), rhs


# ----------------------------------------------------------------------
# Greeks via finite differences

def delta(v: np.ndarray, ds: float) -> np.ndarray:
    """Compute Delta from a one-dimensional price grid ``v``."""
    values = _validate_1d_values(v, "delta")
    return np.gradient(values, _validate_spacing(ds), edge_order=2)


def gamma(v: np.ndarray, ds: float) -> np.ndarray:
    """Compute Gamma from a one-dimensional price grid ``v``."""
    values = _validate_1d_values(v, "gamma")
    spacing = _validate_spacing(ds)
    return np.gradient(np.gradient(values, spacing, edge_order=2), spacing, edge_order=2)


def vega(v: np.ndarray, dv: float) -> np.ndarray:
    """Compute Vega only when an explicit volatility axis is present."""
    values = np.asarray(v, dtype=float)
    if values.ndim != 2:
        raise ValueError("vega requires a two-dimensional grid with an explicit volatility axis")
    return np.gradient(values, _validate_spacing(dv), axis=1, edge_order=2)


def _evaluate_payoff_grid(
    payoff_fn: Callable[[object], object], s_grid: np.ndarray
) -> np.ndarray:
    """Evaluate payoff methods on a grid with a narrow scalar fallback."""

    try:
        values = payoff_fn(s_grid)
    except TypeError:
        values = [payoff_fn(float(s)) for s in s_grid]
    except ValueError as exc:
        message = str(exc).lower()
        if "truth value" not in message or "ambiguous" not in message:
            raise
        values = [payoff_fn(float(s)) for s in s_grid]
    result = np.asarray(values, dtype=float)
    if result.shape != s_grid.shape:
        raise ValueError(
            f"payoff evaluation returned shape {result.shape}, expected {s_grid.shape}"
        )
    if not np.all(np.isfinite(result)):
        raise ValueError("payoff evaluation must return finite values")
    return result


# ----------------------------------------------------------------------
# Convenience solver function

def solve_system(
    s_grid: np.ndarray,
    t: Iterable[float],
    dynamics,
    payoff,
    is_call: bool = True,
    theta: float = 0.5,
) -> DataArray:
    """Solve the Black-Scholes PDE on validated uniform grids.

    ``t`` is a forward transformed time-to-maturity grid ``tau`` with ``t[0]=0``
    at terminal payoff and ``t[-1]`` at valuation horizon.  The returned
    :class:`xarray.DataArray` has dimensions ``time`` and ``space`` plus attrs
    documenting the reference route, grid/time semantics and linear residuals.
    """

    solver = FDSolver(s_grid, dynamics, payoff, is_call=is_call)
    tau_grid = _validate_uniform_time_grid(t)
    dt = float(tau_grid[1] - tau_grid[0])
    a_matrix, b_matrix = solver.matrices(theta, dt)

    values = np.empty((len(tau_grid), solver.N), dtype=float)
    values[0] = solver.initial_condition()
    max_residual = 0.0
    factorized_solver = None

    for i, tau_prev in enumerate(tau_grid[:-1]):
        tau_next = float(tau_grid[i + 1])
        rhs = b_matrix @ values[i]
        u_d = solver.dirichlet(tau_next)
        enforced_matrix, enforced_rhs = solver.apply_dirichlet(a_matrix, rhs, [], u_d)
        if factorized_solver is None:
            factorized_solver = spla.factorized(enforced_matrix.tocsc())
        next_values = factorized_solver(enforced_rhs)
        residual = enforced_matrix @ next_values - enforced_rhs
        max_residual = max(max_residual, float(np.max(np.abs(residual))))
        values[i + 1] = next_values

    result = snapshot(values, tau_grid, solver.s_grid)
    result.attrs.update(
        {
            "fd_backend": "black_scholes_uniform_1d",
            "time_orientation": "tau_time_to_maturity_forward",
            "coordinate_units": {"time": "year", "space": "spot"},
            "theta": float(theta),
            "dt": dt,
            "space_step": solver.ds,
            "spot_grid_uniform": True,
            "time_grid_uniform": True,
            "time_step_count": len(tau_grid) - 1,
            "factorization_reuse_count": len(tau_grid) - 1,
            "max_linear_residual_abs": max_residual,
            "convergence_status": "solved",
            "boundary_condition": "endpoint_dirichlet_elimination",
            "left_boundary_tau0": float(solver.dirichlet(0.0)[0]),
            "right_boundary_final_tau": float(solver.dirichlet(tau_grid[-1])[-1]),
        }
    )
    return result


def _validate_uniform_spot_grid(s_grid: Iterable[float]) -> np.ndarray:
    grid = np.asarray(list(s_grid), dtype=float)
    if grid.ndim != 1 or len(grid) < 3:
        raise ValueError("s_grid must be one-dimensional with at least three nodes")
    if not np.all(np.isfinite(grid)):
        raise ValueError("s_grid entries must be finite")
    if grid[0] < 0.0:
        raise ValueError("s_grid lower boundary must be non-negative")
    diffs = np.diff(grid)
    if np.any(diffs <= 0.0):
        raise ValueError("s_grid must be strictly increasing")
    if not np.allclose(diffs, diffs[0], rtol=_GRID_RTOL, atol=_GRID_ATOL):
        raise ValueError("s_grid must be uniform for the Black-Scholes FD reference backend")
    return grid


def _validate_uniform_time_grid(t: Iterable[float]) -> np.ndarray:
    grid = np.asarray(list(t), dtype=float)
    if grid.ndim != 1 or len(grid) < 2:
        raise ValueError("time grid must be one-dimensional with at least two nodes")
    if not np.all(np.isfinite(grid)):
        raise ValueError("time grid entries must be finite")
    if not np.isclose(grid[0], 0.0, rtol=0.0, atol=_GRID_ATOL):
        raise ValueError("time grid must start at zero time-to-maturity")
    diffs = np.diff(grid)
    if np.any(diffs <= 0.0):
        raise ValueError("time grid must be strictly increasing")
    if not np.allclose(diffs, diffs[0], rtol=_GRID_RTOL, atol=_GRID_ATOL):
        raise ValueError("time grid must be uniform for cached reference factorization")
    return grid


def _validate_spacing(spacing: float) -> float:
    value = float(spacing)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("finite-difference spacing must be finite and positive")
    return value


def _validate_1d_values(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) < 3:
        raise ValueError(f"{name} requires a one-dimensional grid with at least three values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} input values must be finite")
    return array
