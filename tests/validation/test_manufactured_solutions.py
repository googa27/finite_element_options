"""Manufactured-solution evidence for FEM verification issue #117."""

from __future__ import annotations

from math import isfinite, log
from typing import cast

import pytest
import numpy as np
from scipy.sparse import issparse
import sympy as sp

from finite_element_options.validation import (
    ManufacturedRunConfig,
    run_manufactured_case,
    sympy_manufactured_problem,
)
from finite_element_options.validation.fem_evidence import VerificationPerturbation
from finite_element_options.validation.evidence import manufactured


def _orders(rows: list[tuple[float, float]]) -> list[float]:
    return [
        log(rows[index - 1][1] / rows[index][1])
        / log(rows[index - 1][0] / rows[index][0])
        for index in range(1, len(rows))
    ]


def test_manufactured_assembly_is_sparse_and_matches_p1_integral_oracles() -> None:
    """The evidence route must retain sparse structure at increasing mesh sizes."""

    for elements in (4, 32, 1024):
        h = 1.0 / elements
        mass = manufactured._assemble_mass(elements, h)
        stiffness = manufactured._assemble_stiffness(elements, h)
        assert issparse(mass) and issparse(stiffness)
        assert mass.nnz == stiffness.nnz == 3 * elements + 1
        x = np.linspace(0.0, 1.0, elements + 1)
        assert float(mass.sum()) == pytest.approx(1.0, abs=1e-13)
        assert float(x @ (mass @ x)) == pytest.approx(1.0 / 3.0, abs=1e-13)
        assert float(x @ (stiffness @ x)) == pytest.approx(1.0, abs=1e-11)
        assert np.max(np.abs(stiffness @ np.ones_like(x))) < 1e-11


def test_manufactured_load_and_error_norms_match_exact_integrals() -> None:
    """Polynomial source and interpolation errors have independent closed forms."""

    nodes = np.linspace(0.0, 1.0, 5)
    load = manufactured._load_vector(nodes, 0.0, "none")
    assert float(load.sum()) == pytest.approx(0.56 - 0.77 / 6.0, abs=1e-14)
    solution = nodes * (1.0 - nodes)
    l2, h1 = manufactured._error_norms(nodes, solution, 0.0)
    h = 0.25
    assert l2 == pytest.approx(h**2 / np.sqrt(30.0), abs=1e-14)
    assert h1 == pytest.approx(h / np.sqrt(3.0), abs=1e-14)


def test_sympy_manufactured_source_has_zero_symbolic_residual() -> None:
    problem = sympy_manufactured_problem()
    x, tau = sp.symbols("x tau")
    exact = sp.sympify(problem["exact_solution"])
    source = sp.sympify(problem["source"])
    diffusion = sp.Rational(7, 25)
    reaction = sp.Rational(3, 100)

    residual = sp.simplify(
        sp.diff(exact, tau)
        - diffusion * sp.diff(exact, x, 2)
        + reaction * exact
        - source
    )

    assert residual == 0
    assert sp.simplify(exact.subs(x, 0)) == 0
    assert sp.simplify(exact.subs(x, 1)) == 0
    assert problem["strong_form"].startswith("u_tau - D*u_xx")
    assert "Crank-Nicolson" in problem["time_integrator"]


@pytest.mark.parametrize(
    "perturbation",
    ["operator_sign", "source", "reaction", "boundary"],
)
def test_manufactured_perturbations_fail_closed(perturbation: str) -> None:
    result = run_manufactured_case(
        ManufacturedRunConfig(
            elements=64,
            time_steps=512,
            perturbation=cast(VerificationPerturbation, perturbation),
        )
    )

    assert not result.accepted
    if perturbation == "boundary":
        assert result.boundary_residual_inf > 1.0e-12
    else:
        assert result.l2_error > 5.0e-4 or result.h1_error > 2.5e-2


def test_manufactured_solution_has_three_spatial_and_temporal_orders() -> None:
    h_rows = [
        run_manufactured_case(ManufacturedRunConfig(elements=elements, time_steps=2048))
        for elements in (16, 32, 64)
    ]
    time_rows = [
        run_manufactured_case(ManufacturedRunConfig(elements=256, time_steps=steps))
        for steps in (4, 8, 16)
    ]

    assert all(row.accepted for row in h_rows)
    assert all(row.accepted for row in time_rows)
    h_l2_orders = _orders([(row.h, row.l2_error) for row in h_rows])
    h_h1_orders = _orders([(row.h, row.h1_error) for row in h_rows])
    h_payoff_orders = _orders([(row.h, row.payoff_relevant_error) for row in h_rows])
    time_l2_orders = _orders([(row.dt, row.l2_error) for row in time_rows])

    assert len(h_l2_orders) == len(h_h1_orders) == len(h_payoff_orders) == 2
    assert len(time_l2_orders) == 2
    assert all(isfinite(order) and order >= 1.8 for order in h_l2_orders)
    assert all(isfinite(order) and order >= 0.8 for order in h_h1_orders)
    assert all(isfinite(order) and order >= 1.8 for order in h_payoff_orders)
    assert all(isfinite(order) and order >= 1.5 for order in time_l2_orders)


@pytest.mark.parametrize(
    ("elements", "time_steps"),
    [(3, 8), (8, 0)],
)
def test_manufactured_solver_rejects_invalid_grids(
    elements: int, time_steps: int
) -> None:
    with pytest.raises(ValueError, match="requires >=4 elements"):
        run_manufactured_case(
            ManufacturedRunConfig(elements=elements, time_steps=time_steps)
        )
