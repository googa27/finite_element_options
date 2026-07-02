"""Regression tests for the spatial solver interfaces.

```mermaid
flowchart TD
    X[Transformed coordinates] --> U[untransform_state]
    U --> S[Spot grid]
    U --> V[Variance seed]
    S --> B{Call option?}
    B -->|Yes| C[call/payoff branch]
    B -->|No| P[put/payoff branch]
    C --> R[Projected value]
    P --> R
```
"""

import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from finite_element_options.core.dynamics_heston import DynamicsParametersHeston  # noqa: E402
from finite_element_options.core.market import Market  # noqa: E402
from finite_element_options.core.vanilla_bs import EuropeanOptionBs  # noqa: E402
from finite_element_options.space.mesh import create_mesh  # noqa: E402
from finite_element_options.space.solver import SpaceSolver  # noqa: E402
from finite_element_options.space.boundary import DirichletBC  # noqa: E402
from finite_element_options.time_integration.stepper import ThetaScheme  # noqa: E402


def _build_space_solver(is_call: bool):
    dh = DynamicsParametersHeston(
        r=0.03, q=0.03, kappa=0.5, theta=0.5, sig=0.2, rho=0.5
    )
    mkt = Market(r=dh.r)
    bsopt = EuropeanOptionBs(k=0.4, q=dh.q, mkt=mkt)
    mesh, cfg = create_mesh([1.0, 1.0], 1)
    space = SpaceSolver(mesh, dh, bsopt, is_call=is_call, config=cfg)
    return space, dh, bsopt


def test_solver_runs():
    space, _, _ = _build_space_solver(is_call=True)
    t = np.linspace(0.0, 1.0, 3)
    stepper = ThetaScheme(theta=0.5)
    bc = DirichletBC([])
    v_tsv = stepper.solve(t, space, boundary_condition=bc)
    assert v_tsv.shape[0] == 3


@pytest.mark.parametrize(
    ("is_call", "payoff_attr"),
    [(True, "call_payoff"), (False, "put_payoff")],
)
def test_initial_condition_matches_intrinsic(is_call, payoff_attr):
    space, _, payoff = _build_space_solver(is_call=is_call)
    initial = space.initial_condition()
    expected = space.Vh.project(
        lambda x: getattr(payoff, payoff_attr)(
            space.transform.untransform_state(x)[0]
        )
    )
    np.testing.assert_allclose(initial, expected)


@pytest.mark.parametrize(("is_call", "price_attr"), [(True, "call"), (False, "put")])
def test_dirichlet_matches_model_prices(is_call, price_attr):
    space, dynamics, payoff = _build_space_solver(is_call=is_call)
    th = 0.25
    values = space.dirichlet(th)
    th_phys = space.transform.untransform_time(th)
    supports_config = "config" in dynamics.mean_variance.__code__.co_varnames

    def _expected(x):
        state = space.transform.untransform_state(x)
        spots = state[0]
        variance_seed = state[1] if state.shape[0] > 1 else np.zeros_like(spots)
        kwargs = {"config": space.config} if supports_config else {}
        mean_var = dynamics.mean_variance(th_phys, variance_seed, **kwargs)
        return getattr(payoff, price_attr)(th_phys, spots, mean_var)

    expected = space.Vh.project(_expected)
    np.testing.assert_allclose(values, expected)
