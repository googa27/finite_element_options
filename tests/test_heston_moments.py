"""Heston/CIR conditional-moment and domain-diagnostic regressions."""

from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import ValidationError

from finite_element_options.core.cir import (
    cir_conditional_mean,
    cir_conditional_variance,
    cir_time_average_mean,
    cir_variance_domain_diagnostics,
)
from finite_element_options.core.dynamics_heston import DynamicsParametersHeston
from finite_element_options.core.dynamics_heston_3d import DynamicsParametersHeston3D
from finite_element_options.core.market import Market
from finite_element_options.core.vanilla_bs import EuropeanOptionBs
from finite_element_options.space.mesh import create_mesh
from finite_element_options.space.solver import SpaceSolver


def _heston_2d(**overrides) -> DynamicsParametersHeston:
    params = dict(r=0.03, q=0.01, kappa=1.7, theta=0.04, sig=0.35, rho=-0.25)
    params.update(overrides)
    return DynamicsParametersHeston(**params)


def _heston_3d(**overrides) -> DynamicsParametersHeston3D:
    params = dict(
        r=0.03,
        q=0.01,
        kappa=1.7,
        theta=0.04,
        sig_v=0.35,
        rho=-0.25,
        kappa_r=0.4,
        theta_r=0.03,
        sig_r=0.01,
    )
    params.update(overrides)
    return DynamicsParametersHeston3D(**params)


@pytest.mark.parametrize("factory", [_heston_2d, _heston_3d])
def test_heston_mean_variance_uses_exact_cir_time_average(factory) -> None:
    """Boundary effective variance is the CIR time-average, not terminal mean."""

    dynamics = factory()
    tau = 0.75
    variance_seed = np.array([0.01, 0.04, 0.12])

    expected_average = cir_time_average_mean(
        kappa=dynamics.kappa,
        theta=dynamics.theta,
        horizon=tau,
        initial_variance=variance_seed,
    )
    actual = dynamics.mean_variance(tau, variance_seed)

    np.testing.assert_allclose(actual, expected_average, rtol=1e-13, atol=1e-15)

    terminal_mean = cir_conditional_mean(
        kappa=dynamics.kappa,
        theta=dynamics.theta,
        horizon=tau,
        initial_variance=variance_seed,
    )
    np.testing.assert_allclose(
        dynamics.terminal_mean_variance(tau, variance_seed),
        terminal_mean,
        rtol=1e-13,
        atol=1e-15,
    )
    assert not np.allclose(expected_average, terminal_mean, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("factory", [_heston_2d, _heston_3d])
def test_heston_mean_variance_limits_are_stable(factory) -> None:
    variance_seed = np.array([0.0025, 0.04, 0.2])

    zero_horizon = factory().mean_variance(0.0, variance_seed)
    np.testing.assert_allclose(zero_horizon, variance_seed, rtol=0.0, atol=0.0)

    zero_kappa = factory(kappa=0.0).mean_variance(10.0, variance_seed)
    np.testing.assert_allclose(zero_kappa, variance_seed, rtol=0.0, atol=0.0)

    long_horizon_model = factory(kappa=2.0, theta=0.09)
    long_horizon = long_horizon_model.mean_variance(60.0, variance_seed)
    np.testing.assert_allclose(
        long_horizon,
        cir_time_average_mean(
            kappa=2.0,
            theta=0.09,
            horizon=60.0,
            initial_variance=variance_seed,
        ),
        rtol=1e-13,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        long_horizon_model.terminal_mean_variance(60.0, variance_seed),
        np.full_like(variance_seed, 0.09),
        atol=1e-14,
    )

    tiny_horizon = 1.0e-12
    tiny_actual = factory(kappa=1.0e-10, theta=0.07).mean_variance(
        tiny_horizon, variance_seed
    )
    np.testing.assert_allclose(tiny_actual, variance_seed, rtol=1e-15, atol=1e-15)


def test_cir_conditional_variance_matches_closed_form_and_limits() -> None:
    kappa = 1.4
    theta = 0.05
    sigma = 0.3
    tau = 0.8
    variance_seed = np.array([0.01, 0.05, 0.2])
    decay = math.exp(-kappa * tau)
    expected = (
        variance_seed * sigma**2 * decay * (1.0 - decay) / kappa
        + theta * sigma**2 * (1.0 - decay) ** 2 / (2.0 * kappa)
    )

    actual = cir_conditional_variance(
        kappa=kappa,
        theta=theta,
        volatility_of_variance=sigma,
        horizon=tau,
        initial_variance=variance_seed,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(
        cir_conditional_variance(
            kappa=0.0,
            theta=theta,
            volatility_of_variance=sigma,
            horizon=tau,
            initial_variance=variance_seed,
        ),
        variance_seed * sigma**2 * tau,
        rtol=1e-13,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        cir_conditional_variance(
            kappa=kappa,
            theta=theta,
            volatility_of_variance=sigma,
            horizon=0.0,
            initial_variance=variance_seed,
        ),
        np.zeros_like(variance_seed),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "factory,overrides",
    [
        (_heston_2d, {"kappa": -1.0}),
        (_heston_2d, {"theta": -0.01}),
        (_heston_2d, {"sig": 0.0}),
        (_heston_2d, {"rho": 1.5}),
        (_heston_3d, {"kappa": -1.0}),
        (_heston_3d, {"theta": -0.01}),
        (_heston_3d, {"sig_v": 0.0}),
        (_heston_3d, {"rho": -1.5}),
        (_heston_3d, {"kappa_r": -1.0}),
        (_heston_3d, {"theta_r": math.inf}),
        (_heston_3d, {"sig_r": -0.01}),
    ],
)
def test_heston_invalid_variance_parameters_fail_closed(factory, overrides) -> None:
    with pytest.raises((ValueError, ValidationError)):
        factory(**overrides)


def test_heston_3d_accepts_deterministic_short_rate_volatility_limit() -> None:
    dynamics = _heston_3d(sig_r=0.0)

    assert dynamics.sig_r == 0.0


@pytest.mark.parametrize("factory", [_heston_2d, _heston_3d])
def test_heston_domain_diagnostics_report_tail_bound_and_feller_state(factory) -> None:
    if factory is _heston_2d:
        dynamics = factory(kappa=1.2, theta=0.05, sig=0.25)
    else:
        dynamics = factory(kappa=1.2, theta=0.05, sig_v=0.25)
    variance_seed = np.array([0.01, 0.12])

    diagnostics = dynamics.variance_domain_diagnostics(
        horizon=2.0,
        initial_variance=variance_seed,
        tail_mass=1.0e-4,
    )

    assert diagnostics["policy"] == "cir-chebyshev-tail-bound"
    assert diagnostics["domain_lower"] == 0.0
    assert diagnostics["domain_upper"] > diagnostics["mean_variance_max"]
    assert diagnostics["estimated_omitted_mass"] <= 1.0e-4
    assert diagnostics["variance_of_variance_max"] >= 0.0
    assert diagnostics["feller_ratio"] == pytest.approx(
        2.0 * dynamics.kappa * dynamics.theta / dynamics.variance_volatility**2
    )
    assert diagnostics["feller_condition_satisfied"] == (
        diagnostics["feller_ratio"] >= 1.0
    )


def test_variance_domain_upper_bound_contains_initial_variance() -> None:
    """Sharp mean reversion must not truncate the starting variance state."""

    diagnostics = cir_variance_domain_diagnostics(
        kappa=10.0,
        theta=0.01,
        volatility_of_variance=0.01,
        horizon=1.0,
        initial_variance=np.array([1.0]),
    )

    assert diagnostics["mean_variance_max"] < diagnostics["initial_variance_max"]
    assert diagnostics["domain_upper"] >= diagnostics["initial_variance_max"]


def test_variance_domain_diagnostics_reject_empty_variance_samples() -> None:
    with pytest.raises(ValueError, match="initial_variance must be non-empty"):
        cir_variance_domain_diagnostics(
            kappa=1.0,
            theta=0.04,
            volatility_of_variance=0.2,
            horizon=1.0,
            initial_variance=np.array([]),
        )


def test_space_solver_includes_variance_domain_diagnostics_in_evidence() -> None:
    dynamics = _heston_2d(kappa=1.2, theta=0.05, sig=0.25)
    market = Market(r=dynamics.r)
    payoff = EuropeanOptionBs(k=1.0, q=dynamics.q, mkt=market)
    mesh, config = create_mesh([2.0, 0.5], 1)
    space = SpaceSolver(mesh, dynamics, payoff, is_call=True, config=config)

    diagnostics = space.variance_domain_diagnostics(horizon=1.5, tail_mass=1.0e-4)

    assert diagnostics["policy"] == "cir-chebyshev-tail-bound"
    assert diagnostics["mesh_dimension"] == 2
    assert diagnostics["mesh_elements"] == mesh.nelements
    assert diagnostics["mesh_variance_min"] == pytest.approx(0.0)
    assert diagnostics["mesh_variance_max"] == pytest.approx(0.5)
    assert isinstance(diagnostics["mesh_contains_tail_bound"], bool)
    assert diagnostics["domain_upper"] > diagnostics["mean_variance_max"]


def test_space_solver_tail_coverage_requires_lower_and_upper_variance_bounds() -> None:
    dynamics = _heston_2d(kappa=1.2, theta=0.05, sig=0.25)
    market = Market(r=dynamics.r)
    payoff = EuropeanOptionBs(k=1.0, q=dynamics.q, mkt=market)
    mesh, config = create_mesh([2.0, 0.5], 1)
    mesh.p[1] += 0.1
    space = SpaceSolver(mesh, dynamics, payoff, is_call=True, config=config)

    diagnostics = space.variance_domain_diagnostics(horizon=0.1, tail_mass=0.99)

    assert diagnostics["domain_lower"] == 0.0
    assert diagnostics["mesh_variance_min"] > diagnostics["domain_lower"]
    assert diagnostics["domain_upper"] <= diagnostics["mesh_variance_max"]
    assert diagnostics["mesh_contains_tail_bound"] is False
