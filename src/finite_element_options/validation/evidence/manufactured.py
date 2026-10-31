"""Manufactured-solution numerical kernel for FEM verification evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import splu
from skfem import Basis, ElementLineP1, Functional, LinearForm, MeshLine, asm
from skfem.models.poisson import laplace, mass
import sympy as sp

VerificationPerturbation = Literal[
    "none", "operator_sign", "source", "reaction", "boundary"
]
FAILURE_PERTURBATIONS: tuple[VerificationPerturbation, ...] = (
    "operator_sign",
    "source",
    "reaction",
    "boundary",
)


@dataclass(frozen=True, slots=True)
class ManufacturedRunConfig:
    """Controls for one manufactured finite-element solve."""

    elements: int
    time_steps: int
    perturbation: VerificationPerturbation = "none"


@dataclass(frozen=True, slots=True)
class ManufacturedRunResult:
    """Measured errors and residuals for one manufactured solve."""

    elements: int
    time_steps: int
    h: float
    dt: float
    dofs: int
    l2_error: float
    h1_error: float
    payoff_relevant_error: float
    algebraic_residual_inf: float
    boundary_residual_inf: float
    accepted: bool

    def to_public_dict(self) -> dict[str, int | float | bool]:
        """Return the stable public evidence row representation."""

        return {
            "elements": self.elements,
            "time_steps": self.time_steps,
            "h": self.h,
            "dt": self.dt,
            "dofs": self.dofs,
            "l2_error": self.l2_error,
            "h1_error": self.h1_error,
            "payoff_relevant_error": self.payoff_relevant_error,
            "algebraic_residual_inf": self.algebraic_residual_inf,
            "boundary_residual_inf": self.boundary_residual_inf,
            "accepted": self.accepted,
        }


def sympy_manufactured_problem() -> dict[str, Any]:
    """Return SymPy-derived source, boundary and exact-solution metadata."""

    x, tau = sp.symbols("x tau")
    diffusion = sp.Rational(7, 25)
    reaction = sp.Rational(3, 100)
    exact = x * (1 - x) * sp.exp(-tau) * (1 + sp.Rational(1, 5) * tau)
    source = sp.diff(exact, tau) - diffusion * sp.diff(exact, x, 2) + reaction * exact
    return {
        "strong_form": "u_tau - D*u_xx + r*u = f on x in [0,1], tau in [0,1]",
        "weak_form": "(v,u_tau) + D(v_x,u_x) + r(v,u) = (v,f), essential BCs from exact u",
        "time_integrator": "Crank-Nicolson: (M + dt/2 A)u[n+1] = (M - dt/2 A)u[n] + dt/2(F[n]+F[n+1])",
        "source_generation": "SymPy exact symbolic differentiation, then lambdify for quadrature",
        "exact_solution": str(sp.simplify(exact)),
        "source": str(sp.simplify(source)),
        "lower_boundary": str(sp.simplify(exact.subs(x, 0))),
        "upper_boundary": str(sp.simplify(exact.subs(x, 1))),
        "diffusion": float(diffusion),
        "reaction": float(reaction),
        "initial_condition": str(sp.simplify(exact.subs(tau, 0))),
    }


def run_manufactured_case(config: ManufacturedRunConfig) -> ManufacturedRunResult:
    """Solve the manufactured parabolic problem with P1 FEM and return errors."""

    if config.elements < 4 or config.time_steps < 1:
        raise ValueError(
            "manufactured solve requires >=4 elements and positive time steps"
        )
    nodes = np.linspace(0.0, 1.0, config.elements + 1)
    h = 1.0 / config.elements
    dt = 1.0 / config.time_steps
    diffusion = -1.0e-4 if config.perturbation == "operator_sign" else 0.28
    reaction = -0.03 if config.perturbation == "reaction" else 0.03
    basis = _p1_basis(nodes)
    mass_matrix = _assemble_mass(config.elements, h, basis=basis)
    stiffness = _assemble_stiffness(config.elements, h, basis=basis)
    operator = diffusion * stiffness + reaction * mass_matrix
    left_matrix = mass_matrix + 0.5 * dt * operator
    right_matrix = mass_matrix - 0.5 * dt * operator
    interior = np.arange(1, config.elements)
    boundary_dofs = np.asarray([0, config.elements], dtype=np.intp)
    boundary_positions = np.asarray([0, -1], dtype=np.intp)
    lu = splu(csc_matrix(left_matrix[np.ix_(interior, interior)]))
    solution = _exact_u(nodes, 0.0)
    max_residual = 0.0
    max_boundary = 0.0
    for step in range(config.time_steps):
        tau0 = step * dt
        tau1 = (step + 1) * dt
        load = (
            0.5
            * dt
            * (
                _load_vector(nodes, tau0, config.perturbation, basis=basis)
                + _load_vector(nodes, tau1, config.perturbation, basis=basis)
            )
        )
        rhs = right_matrix @ solution + load
        bc = _exact_u(nodes, tau1)
        if config.perturbation == "boundary":
            bc = bc.copy()
            bc[-1] += 0.05
        rhs_i = (
            rhs[interior]
            - left_matrix[np.ix_(interior, boundary_dofs)] @ bc[boundary_positions]
        )
        next_solution = solution.copy()
        next_solution[0] = bc[0]
        next_solution[-1] = bc[-1]
        next_solution[interior] = lu.solve(rhs_i)
        residual = (
            left_matrix[np.ix_(interior, interior)] @ next_solution[interior] - rhs_i
        )
        max_residual = max(max_residual, float(np.linalg.norm(residual, ord=np.inf)))
        exact_boundary = _exact_u(nodes[boundary_positions], tau1)
        max_boundary = max(
            max_boundary,
            float(np.max(np.abs(next_solution[boundary_positions] - exact_boundary))),
        )
        solution = next_solution
    l2, h1 = _error_norms(nodes, solution, 1.0, basis=basis)
    payoff_error = abs(
        float(np.interp(0.5, nodes, solution))
        - float(_exact_u(np.array([0.5]), 1.0)[0])
    )
    accepted = (
        l2 < 5.0e-4
        and h1 < 2.5e-2
        and payoff_error < 2.0e-4
        and max_residual < 1.0e-11
        and max_boundary < 1.0e-12
    )
    return ManufacturedRunResult(
        elements=config.elements,
        time_steps=config.time_steps,
        h=h,
        dt=dt,
        dofs=config.elements + 1,
        l2_error=l2,
        h1_error=h1,
        payoff_relevant_error=payoff_error,
        algebraic_residual_inf=max_residual,
        boundary_residual_inf=max_boundary,
        accepted=accepted,
    )


def _p1_basis(nodes: np.ndarray) -> Basis:
    """Use maintained P1 shapes and degree-five (three-point) quadrature."""

    return Basis(MeshLine(nodes), ElementLineP1(), intorder=5)


def _assemble_mass(
    elements: int, h: float, *, basis: Basis | None = None
) -> csr_matrix:
    selected = basis if basis is not None else _p1_basis(np.arange(elements + 1) * h)
    return asm(mass, selected).tocsr()


def _assemble_stiffness(
    elements: int, h: float, *, basis: Basis | None = None
) -> csr_matrix:
    selected = basis if basis is not None else _p1_basis(np.arange(elements + 1) * h)
    return asm(laplace, selected).tocsr()


def _exact_u(x: np.ndarray, tau: float) -> np.ndarray:
    return x * (1.0 - x) * np.exp(-tau) * (1.0 + 0.2 * tau)


def _exact_ux(x: np.ndarray, tau: float) -> np.ndarray:
    return (1.0 - 2.0 * x) * np.exp(-tau) * (1.0 + 0.2 * tau)


def _source(x: np.ndarray, tau: float) -> np.ndarray:
    exact = _exact_u(x, tau)
    u_tau = x * (1.0 - x) * np.exp(-tau) * (-0.8 - 0.2 * tau)
    return u_tau + 0.56 * np.exp(-tau) * (1.0 + 0.2 * tau) + 0.03 * exact


def _load_vector(
    nodes: np.ndarray,
    tau: float,
    perturbation: VerificationPerturbation,
    *,
    basis: Basis | None = None,
) -> np.ndarray:
    selected = basis if basis is not None else _p1_basis(nodes)

    @LinearForm
    def source_load(v, w):
        fq = _source(w.x[0], tau)
        if perturbation == "source":
            fq = 0.8 * fq
        return fq * v

    return np.asarray(asm(source_load, selected), dtype=float)


def _error_norms(
    nodes: np.ndarray,
    solution: np.ndarray,
    tau: float,
    *,
    basis: Basis | None = None,
) -> tuple[float, float]:
    selected = basis if basis is not None else _p1_basis(nodes)
    field = selected.interpolate(solution)

    @Functional
    def l2_error(w):
        return (w.solution - _exact_u(w.x[0], tau)) ** 2

    @Functional
    def h1_error(w):
        return (w.solution.grad[0] - _exact_ux(w.x[0], tau)) ** 2

    l2 = asm(l2_error, selected, solution=field)
    h1 = asm(h1_error, selected, solution=field)
    return float(np.sqrt(l2)), float(np.sqrt(h1))


__all__ = [
    "FAILURE_PERTURBATIONS",
    "ManufacturedRunConfig",
    "ManufacturedRunResult",
    "VerificationPerturbation",
    "run_manufactured_case",
    "sympy_manufactured_problem",
]
