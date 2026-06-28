"""Public-synthetic Black--Scholes parity fixtures for FEM routing."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np
import scipy.stats as spst

from src.core.dynamics_black_scholes import DynamicsParametersBlackScholes
from src.core.market import Market
from src.core.vanilla_bs import EuropeanOptionBs
from src.space.boundary import DirichletBC
from src.space.mesh import create_mesh
from src.space.solver import SpaceSolver
from src.time.stepper import ThetaScheme

PUBLIC_SYNTHETIC_BLACK_SCHOLES_BENCHMARK_ID = "fem-bs-001"
PUBLIC_SYNTHETIC_PROBLEM_ID = "public-synthetic-vanilla-call-v0"
EXPECTED_BLACK_SCHOLES_CALL_PRICE = 10.450583572185565
DEFAULT_TOLERANCE_ABSOLUTE = 2e-3
DEFAULT_TOLERANCE_RELATIVE = 5e-4


@dataclass(frozen=True)
class FEMParityConvergenceRow:
    """One mesh/time refinement row in the public parity report."""

    refinement_level: int
    time_steps: int
    degrees_of_freedom: int
    observed_price: float
    expected_price: float
    absolute_error: float
    relative_error: float
    observed_delta: float
    expected_delta: float
    delta_absolute_error: float
    observed_gamma: float
    expected_gamma: float
    gamma_absolute_error: float

    def to_public_dict(self) -> dict[str, float | int]:
        """Return a deterministic public-synthetic evidence record."""

        return {
            "refinement_level": self.refinement_level,
            "time_steps": self.time_steps,
            "degrees_of_freedom": self.degrees_of_freedom,
            "observed_price": self.observed_price,
            "expected_price": self.expected_price,
            "absolute_error": self.absolute_error,
            "relative_error": self.relative_error,
            "observed_delta": self.observed_delta,
            "expected_delta": self.expected_delta,
            "delta_absolute_error": self.delta_absolute_error,
            "observed_gamma": self.observed_gamma,
            "expected_gamma": self.expected_gamma,
            "gamma_absolute_error": self.gamma_absolute_error,
        }


@dataclass(frozen=True)
class FEMParityReport:
    """Executable public-synthetic FEM parity report."""

    benchmark_id: str
    problem_id: str
    privacy_class: str
    expected_price: float
    observed_price: float
    price_absolute_error: float
    price_relative_error: float
    expected_delta: float
    observed_delta: float
    delta_absolute_error: float
    delta_tolerance_absolute: float
    expected_gamma: float
    observed_gamma: float
    gamma_absolute_error: float
    gamma_tolerance_absolute: float
    tolerance_absolute: float
    tolerance_relative: float
    convergence_rows: tuple[FEMParityConvergenceRow, ...]
    diagnostics: dict[str, str | int | float]

    def to_public_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable evidence payload with no private data."""

        return {
            "benchmark_id": self.benchmark_id,
            "problem_id": self.problem_id,
            "privacy_class": self.privacy_class,
            "expected_price": self.expected_price,
            "observed_price": self.observed_price,
            "price_absolute_error": self.price_absolute_error,
            "price_relative_error": self.price_relative_error,
            "expected_delta": self.expected_delta,
            "observed_delta": self.observed_delta,
            "delta_absolute_error": self.delta_absolute_error,
            "delta_tolerance_absolute": self.delta_tolerance_absolute,
            "expected_gamma": self.expected_gamma,
            "observed_gamma": self.observed_gamma,
            "gamma_absolute_error": self.gamma_absolute_error,
            "gamma_tolerance_absolute": self.gamma_tolerance_absolute,
            "tolerance_absolute": self.tolerance_absolute,
            "tolerance_relative": self.tolerance_relative,
            "convergence_rows": [row.to_public_dict() for row in self.convergence_rows],
            "diagnostics": dict(self.diagnostics),
        }


def run_public_black_scholes_parity_fixture(
    *,
    refinement_levels: tuple[int, ...] = (4, 5, 6),
    time_steps: int = 80,
) -> FEMParityReport:
    """Run the public-synthetic Black-Scholes call FEM parity fixture.

    The existing FEM implementation is formulated in normalized spot/strike
    coordinates.  Because the Black-Scholes PDE is homogeneous in spot and
    strike, the fixture solves with ``S/K = 1`` and scales the value back by the
    public synthetic strike ``K=100``.
    """

    if not refinement_levels:
        raise ValueError("at least one refinement level is required")
    if time_steps <= 0:
        raise ValueError("time_steps must be positive")

    rows = tuple(_run_row(refinement_level=level, time_steps=time_steps) for level in refinement_levels)
    final = rows[-1]
    diagnostics: dict[str, str | int | float] = {
        "mesh_family": "line_uniform",
        "element_family": "lagrange_p2",
        "time_integrator": "theta_crank_nicolson",
        "linear_solver": "scikit_fem_scipy_direct",
        "boundary_conditions": "named_endpoint_dirichlet_projection; upper endpoint uses exact Black-Scholes value as the finite-domain linear-growth proxy",
        "boundary_nodes_enforced": 2,
        "weak_form_sign_convention": "existing_forward_tau_identity_transform_black_scholes_forms",
        "coordinate_transform": "identity",
        "spot": 100.0,
        "strike": 100.0,
        "rate": 0.05,
        "volatility": 0.2,
        "maturity": 1.0,
        "deterministic_seed": 1729,
        "source_issue": "googa27/finite_element_options#64",
        "oracle_issue": "googa27/haircut-engine#108",
        "greek_policy": "central finite-element stencil at S=K with explicit delta/gamma oracle errors; payoff-kink production maturity remains separate evidence",
    }
    return FEMParityReport(
        benchmark_id=PUBLIC_SYNTHETIC_BLACK_SCHOLES_BENCHMARK_ID,
        problem_id=PUBLIC_SYNTHETIC_PROBLEM_ID,
        privacy_class="public_synthetic",
        expected_price=EXPECTED_BLACK_SCHOLES_CALL_PRICE,
        observed_price=final.observed_price,
        price_absolute_error=final.absolute_error,
        price_relative_error=final.relative_error,
        expected_delta=final.expected_delta,
        observed_delta=final.observed_delta,
        delta_absolute_error=final.delta_absolute_error,
        delta_tolerance_absolute=1e-3,
        expected_gamma=final.expected_gamma,
        observed_gamma=final.observed_gamma,
        gamma_absolute_error=final.gamma_absolute_error,
        gamma_tolerance_absolute=2e-5,
        tolerance_absolute=DEFAULT_TOLERANCE_ABSOLUTE,
        tolerance_relative=DEFAULT_TOLERANCE_RELATIVE,
        convergence_rows=rows,
        diagnostics=diagnostics,
    )


def _run_row(*, refinement_level: int, time_steps: int) -> FEMParityConvergenceRow:
    if refinement_level < 1:
        raise ValueError("refinement_level must be positive")

    strike = 100.0
    dynamics = DynamicsParametersBlackScholes(r=0.05, q=0.0, sig=0.2)
    market = Market(r=dynamics.r)
    option = EuropeanOptionBs(k=1.0, q=dynamics.q, mkt=market)
    times = np.linspace(0.0, 1.0, time_steps + 1)
    mesh, config = create_mesh([4.0], refinement_level)
    mesh = mesh.with_boundaries(
        {
            "left": lambda x: np.isclose(x[0], 0.0),
            "right": lambda x: np.isclose(x[0], 4.0),
        }
    )
    space = SpaceSolver(mesh, dynamics, option, is_call=True, config=config)
    with np.errstate(divide="ignore", invalid="ignore"):
        solution = ThetaScheme(theta=0.5).solve(times, space, boundary_condition=DirichletBC(["left", "right"]))
    spot_node = int(np.argmin(np.abs(space.Vh.doflocs[0] - 1.0)))
    normalized_price = float(solution[-1, spot_node])
    observed_delta, observed_gamma = _central_difference_greeks(space.Vh.doflocs[0], solution[-1], strike)
    observed_price = strike * normalized_price
    expected_price = EXPECTED_BLACK_SCHOLES_CALL_PRICE
    expected_delta = float(option.call_delta(1.0, 1.0, dynamics.sig**2))
    expected_gamma = float(spst.norm.pdf(option.d1(1.0, 1.0, dynamics.sig**2)) / (strike * dynamics.sig))
    absolute_error = abs(observed_price - expected_price)
    relative_error = absolute_error / max(abs(expected_price), 1.0)
    delta_absolute_error = abs(observed_delta - expected_delta)
    gamma_absolute_error = abs(observed_gamma - expected_gamma)
    values = (observed_price, absolute_error, relative_error, observed_delta, observed_gamma)
    if any(not isfinite(value) for value in values):
        raise FloatingPointError("non-finite FEM parity fixture result")
    return FEMParityConvergenceRow(
        refinement_level=refinement_level,
        time_steps=time_steps,
        degrees_of_freedom=int(space.Vh.N),
        observed_price=observed_price,
        expected_price=expected_price,
        absolute_error=absolute_error,
        relative_error=relative_error,
        observed_delta=observed_delta,
        expected_delta=expected_delta,
        delta_absolute_error=delta_absolute_error,
        observed_gamma=observed_gamma,
        expected_gamma=expected_gamma,
        gamma_absolute_error=gamma_absolute_error,
    )


def _central_difference_greeks(dof_locations: np.ndarray, values: np.ndarray, strike: float) -> tuple[float, float]:
    """Return central finite-element Delta and Gamma at normalized spot one."""

    order = np.argsort(dof_locations)
    coordinates = dof_locations[order]
    ordered_values = values[order]
    center = int(np.argmin(np.abs(coordinates - 1.0)))
    if center == 0 or center == len(coordinates) - 1:
        raise ValueError("spot=1.0 must have neighboring FEM nodes for Greek extraction")
    left_spacing = coordinates[center] - coordinates[center - 1]
    right_spacing = coordinates[center + 1] - coordinates[center]
    delta = (ordered_values[center + 1] - ordered_values[center - 1]) / (
        coordinates[center + 1] - coordinates[center - 1]
    )
    gamma_normalized = 2.0 * (
        (ordered_values[center + 1] - ordered_values[center]) / right_spacing
        - (ordered_values[center] - ordered_values[center - 1]) / left_spacing
    ) / (left_spacing + right_spacing)
    return float(delta), float(gamma_normalized / strike)


__all__ = [
    "DEFAULT_TOLERANCE_ABSOLUTE",
    "DEFAULT_TOLERANCE_RELATIVE",
    "EXPECTED_BLACK_SCHOLES_CALL_PRICE",
    "PUBLIC_SYNTHETIC_BLACK_SCHOLES_BENCHMARK_ID",
    "PUBLIC_SYNTHETIC_PROBLEM_ID",
    "FEMParityConvergenceRow",
    "FEMParityReport",
    "run_public_black_scholes_parity_fixture",
]
