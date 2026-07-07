"""Tests for the experimental FEniCS solver."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from finite_element_options.core.dynamics_black_scholes import DynamicsParametersBlackScholes
from finite_element_options.core.market import Market
from finite_element_options.core.vanilla_bs import EuropeanOptionBs
from finite_element_options.space import fenics_solver
from finite_element_options.space.fenics_solver import FenicsSolver

HAS_FENICS = fenics_solver.fem is not None


class _DivergedKSP:
    def getConvergedReason(self) -> int:
        return -3

    def getIterationNumber(self) -> int:
        return 17

    def getResidualNorm(self) -> float:
        return 2.5e-3

    def getOptionsPrefix(self) -> str:
        return "feo_test_"


class _ConvergedKSP:
    def getConvergedReason(self) -> int:
        return 2

    def getIterationNumber(self) -> int:
        return 4

    def getResidualNorm(self) -> float:
        return 1.0e-11

    def getOptionsPrefix(self) -> str:
        return "feo_test_"


@pytest.mark.skipif(not HAS_FENICS, reason="FEniCSx not installed")
def test_fenics_black_scholes() -> None:
    """Verify that the FEniCS solver reproduces analytic prices."""
    dh = DynamicsParametersBlackScholes(r=0.03, q=0.0, sig=0.2)
    mkt = Market(r=dh.r)
    bsopt = EuropeanOptionBs(k=1.0, q=dh.q, mkt=mkt)
    solver = FenicsSolver(
        domain=(0.0, 2.0), num_elements=16, dynamics=dh, payoff=bsopt, is_call=True
    )
    t = np.linspace(0.0, 1.0, 5)
    v = solver.solve(t)
    node = np.argmin(np.abs(solver.Vh.tabulate_dof_coordinates().ravel() - 1.0))
    price_num = v[-1, node]
    price_exact = bsopt.call(t[-1], 1.0, dh.sig**2)
    assert price_num == pytest.approx(price_exact, rel=1e-2)


def test_fenics_solver_uses_dolfinx_boundary_dof_apis_not_local_row_replacement() -> None:
    """The backend must not zero the whole matrix or assume local edge DOFs."""

    source = "\n".join(
        [
            inspect.getsource(FenicsSolver._locate_boundary_dofs),
            inspect.getsource(FenicsSolver._dirichlet_bcs),
            inspect.getsource(FenicsSolver.apply_dirichlet),
        ]
    )

    assert "mesh.locate_entities_boundary" in source
    assert "fem.locate_dofs_topological" in source
    assert "fem.dirichletbc" in source
    assert "loc.set(0.0)" not in source
    assert "diagonal().set(1.0)" not in source
    assert "vals[0]" not in source
    assert "vals[-1]" not in source


def test_fenics_solver_sets_ksp_operator_after_boundary_application_and_checks_reason() -> None:
    """PETSc divergence must fail explicitly with solver diagnostics."""

    source = "\n".join(
        [
            inspect.getsource(FenicsSolver._new_ksp),
            inspect.getsource(FenicsSolver._check_ksp_converged),
            inspect.getsource(FenicsSolver.solve),
        ]
    )

    assert "setOperators(A_bc)" in source
    assert "getConvergedReason" in source
    with pytest.raises(RuntimeError, match="PETSc KSP failed to converge"):
        FenicsSolver._check_ksp_converged(_DivergedKSP())
    FenicsSolver._check_ksp_converged(_ConvergedKSP())
