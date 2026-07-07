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
from dataclasses import dataclass, field

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


@dataclass
class _ConfigRecordingMeanVariance:
    """Callable mean-variance oracle deliberately lacking ``__code__`` metadata."""

    calls: list[object] = field(default_factory=list)

    def __call__(self, th, v, *, config=None):  # pylint: disable=unused-argument
        self.calls.append(config)
        return np.full_like(v, 0.04, dtype=float)


@dataclass
class _CallableMeanVarianceDynamics:
    """Minimal dynamics whose explicit callable contract replaces signature probes."""

    r: float = 0.03
    q: float = 0.01
    sig: float = 0.2
    mean_variance: _ConfigRecordingMeanVariance = field(
        default_factory=_ConfigRecordingMeanVariance
    )

    def A(self, s):
        return [[self.sig**2 * s**2]]

    def dA(self, s):
        return [2 * self.sig**2 * s]

    def b(self, s):
        return [(self.r - self.q) * s]

    def discount(self, state, time):  # pylint: disable=unused-argument
        return self.r

    def source(self, state, time):  # pylint: disable=unused-argument
        return 0.0


def _build_space_solver(is_call: bool, *, adaptive_criterion: str | None = None):
    dh = DynamicsParametersHeston(
        r=0.03, q=0.03, kappa=0.5, theta=0.5, sig=0.2, rho=0.5
    )
    mkt = Market(r=dh.r)
    bsopt = EuropeanOptionBs(k=0.4, q=dh.q, mkt=mkt)
    mesh, cfg = create_mesh([1.0, 1.0], 1)
    space = SpaceSolver(
        mesh,
        dh,
        bsopt,
        is_call=is_call,
        config=cfg,
        adaptive_criterion=adaptive_criterion,
    )
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
    th_phys = float(np.asarray(space.transform.untransform_time(th)))
    state = space.transform.untransform_state(space.Vh.doflocs)
    spots = state[0]
    variance_seed = state[1] if state.shape[0] > 1 else np.zeros_like(spots)
    mean_var = dynamics.mean_variance(th_phys, variance_seed, config=space.config)
    expected = getattr(payoff, price_attr)(th_phys, spots, mean_var)
    np.testing.assert_allclose(values, expected)


def test_dirichlet_uses_explicit_mean_variance_protocol_without_code_introspection():
    dynamics = _CallableMeanVarianceDynamics()
    payoff = EuropeanOptionBs(k=0.4, q=dynamics.q, mkt=Market(r=dynamics.r))
    mesh, cfg = create_mesh([1.0], 1)
    space = SpaceSolver(mesh, dynamics, payoff, is_call=True, config=cfg)

    assert not hasattr(dynamics.mean_variance, "__code__")

    values = space.dirichlet(0.25)

    assert values.shape == (space.Vh.N,)
    assert dynamics.mean_variance.calls == [space.config]


def test_refine_with_transfer_rebuilds_basis_and_returns_values():
    space, _, _ = _build_space_solver(is_call=True, adaptive_criterion="gradient")
    old_elements = space.mesh.nelements
    old_domain = space.mesh.domain_spec
    values = space.initial_condition()

    result = space.refine_with_transfer(values)

    assert space.mesh.nelements > old_elements
    assert result.mesh is space.mesh
    assert result.values.shape == (space.Vh.N,)
    assert result.diagnostics.old_elements == old_elements
    assert result.diagnostics.new_elements == space.mesh.nelements
    assert space.last_adaptive_diagnostics == result.diagnostics
    assert space.mesh.domain_spec == old_domain
    assert set(space.mesh.boundaries) == {"s_min", "s_max", "v_min", "v_max"}
    assert space.Vh.get_dofs(["s_min"]).nodal["u"].size > 0
