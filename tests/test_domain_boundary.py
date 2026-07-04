"""Domain and boundary-condition regression tests for issue #39."""

from __future__ import annotations

import numpy as np
import pytest

from finite_element_options.core.dynamics_black_scholes import (
    DynamicsParametersBlackScholes,
)
from finite_element_options.core.dynamics_heston import DynamicsParametersHeston
from finite_element_options.core.market import Market
from finite_element_options.core.vanilla_bs import EuropeanOptionBs
from finite_element_options.space.boundary import DirichletBC
from finite_element_options.space.domain import DomainAxis, DomainSpec
from finite_element_options.space.mesh import create_mesh
from finite_element_options.space.solver import SpaceSolver
from finite_element_options.time_integration.stepper import ThetaScheme
from finite_element_options.transform import CoordinateTransform, LogPrice


def _black_scholes_space(*, lower: float = 0.0, upper: float = 200.0) -> SpaceSolver:
    dynamics = DynamicsParametersBlackScholes(r=0.05, q=0.02, sig=0.25)
    payoff = EuropeanOptionBs(k=100.0, q=dynamics.q, mkt=Market(r=dynamics.r))
    mesh, config = create_mesh([DomainAxis("s", lower, upper)], refine=2)
    return SpaceSolver(mesh, dynamics, payoff, is_call=True, config=config)


def test_create_mesh_supports_negative_domains_and_named_facets() -> None:
    """Meshes are no longer implicitly restricted to zero lower bounds."""

    mesh, _ = create_mesh([DomainAxis("r", -0.05, 0.15)], refine=1)

    assert np.min(mesh.p[0]) == pytest.approx(-0.05)
    assert np.max(mesh.p[0]) == pytest.approx(0.15)
    assert mesh.domain_spec.to_public_dict()["axes"][0]["name"] == "r"
    assert {"r_min", "r_max"} <= set(mesh.boundaries)


def test_create_mesh_accepts_numpy_bound_pair_extents() -> None:
    """Explicit bound-pair domains may come from numerical NumPy arrays."""

    mesh, _ = create_mesh(np.array([[0.25, 1.75]]), refine=1)

    assert np.min(mesh.p[0]) == pytest.approx(0.25)
    assert np.max(mesh.p[0]) == pytest.approx(1.75)
    assert mesh.domain_spec.axes[0].lower == pytest.approx(0.25)
    assert mesh.domain_spec.axes[0].upper == pytest.approx(1.75)
    assert {"s_min", "s_max"} <= set(mesh.boundaries)


def test_dirichlet_bc_materializes_generators_and_rejects_bad_facets() -> None:
    """Boundary collections are consumed once and validated before enforcement."""

    space = _black_scholes_space()
    values = space.dirichlet(0.5)
    bc = DirichletBC(name for name in ["s_min", "s_max"])

    first_A, first_b = bc.apply(space, space.mass.copy(), values.copy(), 0.5)
    second_A, second_b = bc.apply(space, space.mass.copy(), values.copy(), 0.5)

    assert first_A.shape == second_A.shape == space.mass.shape
    np.testing.assert_allclose(first_b, second_b)
    assert bc.boundaries == ("s_min", "s_max")

    with pytest.raises(ValueError, match="duplicate boundary"):
        DirichletBC(["s_min", "s_min"])

    with pytest.raises(ValueError, match="Unknown boundary facet"):
        DirichletBC(["not_a_facet"]).apply(space, space.mass.copy(), values.copy(), 0.5)

    single = DirichletBC("s_min")
    assert single.boundaries == ("s_min",)
    single_A, single_b = single.apply(space, space.mass.copy(), values.copy(), 0.5)
    assert single_A.shape == space.mass.shape
    assert single_b.shape == values.shape


def test_black_scholes_dirichlet_facets_include_carry_strike_and_maturity() -> None:
    """Named endpoint facets must use the full Black-Scholes oracle semantics."""

    space = _black_scholes_space(upper=250.0)
    maturity = 1.25
    values = space.dirichlet(maturity)
    variance = space.dynamics.sig**2
    payoff = space.payoff

    left = space.Vh.get_dofs("s_min").all()
    right = space.Vh.get_dofs("s_max").all()
    left_spot = space.Vh.doflocs[0, left]
    right_spot = space.Vh.doflocs[0, right]

    np.testing.assert_allclose(
        values[left], payoff.call(maturity, left_spot, variance), atol=1e-10
    )
    np.testing.assert_allclose(
        values[right], payoff.call(maturity, right_spot, variance), atol=1e-10
    )
    assert values[right][0] == pytest.approx(
        float(payoff.call(maturity, float(right_spot[0]), variance))
    )


def test_log_transformed_boundary_values_match_physical_boundaries() -> None:
    """State-coordinate transforms must preserve boundary value semantics."""

    dynamics = DynamicsParametersBlackScholes(r=0.03, q=0.01, sig=0.2)
    payoff = EuropeanOptionBs(k=100.0, q=dynamics.q, mkt=Market(r=dynamics.r))
    physical_domain = DomainSpec((DomainAxis("s", 20.0, 220.0),))
    transformed_domain = physical_domain.transform(CoordinateTransform(price=LogPrice()))
    mesh, config = create_mesh(transformed_domain, refine=2)
    space = SpaceSolver(
        mesh,
        dynamics,
        payoff,
        is_call=False,
        transform=CoordinateTransform(price=LogPrice()),
        config=config,
    )

    maturity = 0.75
    values = space.dirichlet(maturity)
    variance = dynamics.sig**2
    for facet in ("s_min", "s_max"):
        dofs = space.Vh.get_dofs(facet).all()
        physical_spot = np.exp(space.Vh.doflocs[0, dofs])
        np.testing.assert_allclose(
            values[dofs], payoff.put(maturity, physical_spot, variance), atol=1e-10
        )


def test_domain_diagnostics_are_attached_to_time_step_solves() -> None:
    """Every solve records domain and tail diagnostics for downstream evidence."""

    dynamics = DynamicsParametersHeston(
        r=0.03, q=0.01, kappa=1.5, theta=0.04, sig=0.2, rho=-0.3
    )
    payoff = EuropeanOptionBs(k=100.0, q=dynamics.q, mkt=Market(r=dynamics.r))
    domain = DomainSpec((DomainAxis("s", 0.0, 220.0), DomainAxis("v", 0.0, 0.5)))
    mesh, config = create_mesh(domain, refine=1)
    space = SpaceSolver(mesh, dynamics, payoff, is_call=True, config=config)
    stepper = ThetaScheme(theta=0.5)

    result = stepper.solve([0.0, 0.25, 1.0], space, boundary_condition=DirichletBC(["s_min", "s_max"]))

    diagnostics = stepper.last_domain_diagnostics
    assert result.shape[0] == 3
    assert diagnostics["horizon"] == pytest.approx(1.0)
    assert diagnostics["state_domain"][1]["name"] == "v"
    assert diagnostics["variance_domain"]["policy"] == "cir-chebyshev-tail-bound"
    assert "estimated_omitted_mass" in diagnostics["variance_domain"]
    assert {"s_min", "s_max", "v_min", "v_max"} <= set(diagnostics["boundary_facets"])
