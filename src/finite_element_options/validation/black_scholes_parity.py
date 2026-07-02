"""Public-synthetic Black--Scholes parity fixtures for FEM routing.

The module exposes deterministic fixture generators for the PIELM parity workstream.
It includes both executable report structures and lightweight public-file contracts
that a downstream consumer (for example arxiv-lab) can read without importing
`src` internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path
import json
from typing import Any

import numpy as np
import scipy.stats as spst

from ..core.dynamics_black_scholes import DynamicsParametersBlackScholes
from ..core.market import Market
from ..core.vanilla_bs import EuropeanOptionBs
from ..space.boundary import DirichletBC
from ..space.mesh import create_mesh
from ..space.solver import SpaceSolver
from ..time_integration.stepper import ThetaScheme


PUBLIC_SYNTHETIC_BLACK_SCHOLES_BENCHMARK_ID = "fem-bs-001"
PUBLIC_SYNTHETIC_PROBLEM_ID = "public-synthetic-vanilla-call-v0"
EXPECTED_BLACK_SCHOLES_CALL_PRICE = 10.450583572185565
DEFAULT_TOLERANCE_ABSOLUTE = 2e-3
DEFAULT_TOLERANCE_RELATIVE = 5e-4
DEFAULT_DELTA_TOLERANCE_ABSOLUTE = 1e-3
DEFAULT_GAMMA_TOLERANCE_ABSOLUTE = 2e-5


FIXTURE_ROOT = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "fem_bs_001"
FEM_BS_001_PROBLEM_SPEC_PATH = FIXTURE_ROOT / "problem_spec.json"
FEM_BS_001_RESULT_EXPORT_PATH = FIXTURE_ROOT / "result_export.json"


@dataclass(frozen=True)
class BoundaryMetadata:
    """Typed boundary condition metadata for FEM oracle contracts."""

    location: str
    condition_type: str
    expression: str
    enforced_nodes: int = 1

    def to_public_dict(self) -> dict[str, float | int | str]:
        """Return a JSON-safe boundary specification dictionary."""

        return {
            "location": self.location,
            "condition_type": self.condition_type,
            "expression": self.expression,
            "enforced_nodes": self.enforced_nodes,
        }


@dataclass(frozen=True)
class WeakFormMetadata:
    """Weak-form metadata that must be consistent between code and fixture."""

    equation_id: str
    sign_convention: str
    time_transformation: str
    coordinate_transform: str

    def to_public_dict(self) -> dict[str, str]:
        """Return a JSON-safe weak-form metadata dictionary."""

        return {
            "equation_id": self.equation_id,
            "sign_convention": self.sign_convention,
            "time_transformation": self.time_transformation,
            "coordinate_transform": self.coordinate_transform,
        }


@dataclass(frozen=True)
class MeshMetadata:
    """Deterministic mesh metadata used by the public fixture contract."""

    mesh_family: str
    element_family: str
    domain_min: float
    domain_max: float
    spatial_domain: str
    min_refinement_level: int
    max_refinement_level: int
    refinement_levels: tuple[int, ...]
    solver_backing: str

    def to_public_dict(self) -> dict[str, str | float | int | list[int]]:
        """Return a JSON-safe mesh configuration payload."""

        return {
            "mesh_family": self.mesh_family,
            "element_family": self.element_family,
            "domain_min": self.domain_min,
            "domain_max": self.domain_max,
            "spatial_domain": self.spatial_domain,
            "min_refinement_level": self.min_refinement_level,
            "max_refinement_level": self.max_refinement_level,
            "refinement_levels": list(self.refinement_levels),
            "solver_backing": self.solver_backing,
        }


@dataclass(frozen=True)
class TimeMetadata:
    """Time-stepping metadata."""

    integrator: str
    theta: float
    time_steps: int
    start_time: float
    end_time: float
    time_domain: str

    def to_public_dict(self) -> dict[str, float | int | str]:
        """Return a JSON-safe time discretization metadata dictionary."""

        return {
            "integrator": self.integrator,
            "theta": self.theta,
            "time_steps": self.time_steps,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "time_domain": self.time_domain,
        }


@dataclass(frozen=True)
class SensitivityReferencePolicy:
    """Policy for Delta/Gamma checks in the public oracle comparison."""

    policy_id: str
    delta_reference: str
    gamma_reference: str
    extraction_method: str
    allowed_fallback: str

    def to_public_dict(self) -> dict[str, str]:
        """Return a JSON-safe sensitivity-policy dictionary."""

        return {
            "policy_id": self.policy_id,
            "delta_reference": self.delta_reference,
            "gamma_reference": self.gamma_reference,
            "extraction_method": self.extraction_method,
            "allowed_fallback": self.allowed_fallback,
        }


@dataclass(frozen=True)
class ComparisonPolicy:
    """Documented cross-solver parity policy for arxiv-lab."""

    policy_id: str
    mode: str
    primary_metric: str
    tolerance: float
    note: str
    metric_tolerances: tuple[tuple[str, float], ...]

    def to_public_dict(self) -> dict[str, str | float | dict[str, float]]:
        """Return a JSON-safe comparison-policy payload."""

        return {
            "policy_id": self.policy_id,
            "mode": self.mode,
            "primary_metric": self.primary_metric,
            "tolerance": self.tolerance,
            "metric_tolerances": dict(self.metric_tolerances),
            "note": self.note,
        }


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
    weak_form: WeakFormMetadata
    mesh_metadata: MeshMetadata
    time_metadata: TimeMetadata
    boundaries: tuple[BoundaryMetadata, ...]
    sensitivity_reference_policy: SensitivityReferencePolicy
    comparison_policy: ComparisonPolicy
    config_hash: str

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
            "config_hash": self.config_hash,
            "weak_form": self.weak_form.to_public_dict(),
            "mesh_metadata": self.mesh_metadata.to_public_dict(),
            "time_metadata": self.time_metadata.to_public_dict(),
            "boundaries": [item.to_public_dict() for item in self.boundaries],
            "sensitivity_reference_policy": self.sensitivity_reference_policy.to_public_dict(),
            "comparison_policy": self.comparison_policy.to_public_dict(),
            "convergence_rows": [row.to_public_dict() for row in self.convergence_rows],
            "diagnostics": dict(self.diagnostics),
        }

    def export_payload(self) -> dict[str, Any]:
        """Return a result-export shape suitable for direct public consumption."""

        comparison_policy = {
            **self.comparison_policy.to_public_dict(),
            "metric_tolerances": {
                "price_absolute": self.tolerance_absolute,
                "price_relative": self.tolerance_relative,
                "delta_absolute": self.delta_tolerance_absolute,
                "gamma_absolute": self.gamma_tolerance_absolute,
            },
        }
        return {
            "format_version": "fem-bs-oracle-result-v1",
            "benchmark_id": self.benchmark_id,
            "problem_id": self.problem_id,
            "privacy_class": self.privacy_class,
            "config_hash": self.config_hash,
            "comparison_policy": comparison_policy,
            "weak_form": self.weak_form.to_public_dict(),
            "mesh_metadata": self.mesh_metadata.to_public_dict(),
            "time_metadata": self.time_metadata.to_public_dict(),
            "sensitivity_reference_policy": self.sensitivity_reference_policy.to_public_dict(),
            "rows": [row.to_public_dict() for row in self.convergence_rows],
            "summary": {
                "expected_price": self.expected_price,
                "observed_price": self.observed_price,
                "price_absolute_error": self.price_absolute_error,
                "price_relative_error": self.price_relative_error,
                "price_tolerance_absolute": self.tolerance_absolute,
                "price_tolerance_relative": self.tolerance_relative,
                "expected_delta": self.expected_delta,
                "observed_delta": self.observed_delta,
                "delta_absolute_error": self.delta_absolute_error,
                "delta_tolerance_absolute": self.delta_tolerance_absolute,
                "expected_gamma": self.expected_gamma,
                "observed_gamma": self.observed_gamma,
                "gamma_absolute_error": self.gamma_absolute_error,
                "gamma_tolerance_absolute": self.gamma_tolerance_absolute,
            },
        }


def build_public_fem_bs_oracle_problem_spec(
    *,
    refinement_levels: tuple[int, ...] = (4, 5, 6),
    time_steps: int = 80,
) -> dict[str, Any]:
    """Return the deterministic fixture/problem-spec contract used by arxiv-lab."""

    if not refinement_levels:
        raise ValueError("at least one refinement level is required")
    if time_steps <= 0:
        raise ValueError("time_steps must be positive")

    return {
        "contract_version": "fem-parity-contract/v1",
        "fixture_id": PUBLIC_SYNTHETIC_BLACK_SCHOLES_BENCHMARK_ID,
        "problem_id": PUBLIC_SYNTHETIC_PROBLEM_ID,
        "title": "FEM oracle fixture for public-synthetic European call",
        "privacy_class": "public_synthetic",
        "problem": {
            "name": "European call Black-Scholes",
            "spot": 100.0,
            "strike": 100.0,
            "rate": 0.05,
            "volatility": 0.2,
            "maturity": 1.0,
            "units": {
                "spot": "CLP",
                "strike": "CLP",
                "time": "years",
                "volatility": "annualized_decimal",
            },
        },
        "weak_form": _public_weak_form_metadata().to_public_dict(),
        "boundaries": [
            {
                "location": "S=0",
                "condition_type": "dirichlet",
                "expression": "0",
                "enforced_nodes": 1,
            },
            {
                "location": "S=S_max",
                "condition_type": "dirichlet",
                "expression": "linear_growth",
                "enforced_nodes": 1,
            },
        ],
        "mesh_metadata": {
            "family": "line_uniform",
            "element_family": "lagrange_p2",
            "spatial_domain": "[0, 4.0] normalized spot",
            "solver_backing": "scikit-fem+sparse-direct",
            "mesh_refinement_levels": list(refinement_levels),
            "default_time_steps": time_steps,
        },
        "time_metadata": {
            "integrator": "theta_crank_nicolson",
            "theta": 0.5,
            "time_domain": "[0, 1.0]",
        },
        "sensitivity_reference_policy": {
            "policy_id": "finite_difference_central_stencil_v1",
            "delta_reference": "Black-Scholes analytical delta at valuation time, S=K",
            "gamma_reference": "Black-Scholes analytical gamma at valuation time, S=K",
            "extraction_method": "central finite-difference in normalized strike space",
            "allowed_fallback": "separate kink-aware evidence for production Greek policy",
        },
        "comparison_policy": {
            "policy_id": "equal-error-budget-v1",
            "mode": "equal_error",
            "primary_metric": "abs(price,delta,gamma errors)@policy_tolerance",
            "tolerance": DEFAULT_TOLERANCE_ABSOLUTE,
            "metric_tolerances": {
                "price_absolute": DEFAULT_TOLERANCE_ABSOLUTE,
                "price_relative": DEFAULT_TOLERANCE_RELATIVE,
                "delta_absolute": DEFAULT_DELTA_TOLERANCE_ABSOLUTE,
                "gamma_absolute": DEFAULT_GAMMA_TOLERANCE_ABSOLUTE,
            },
            "note": "Compare by matching error budget, not by raw grid size alone.",
        },
        "result_export_uri": "tests/fixtures/fem_bs_001/result_export.json",
    }


def build_fixture_config_hash(payload: dict[str, Any]) -> str:
    """Compute a deterministic hash for fixture contracts and export control."""

    payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(payload_bytes).hexdigest()


def write_public_fem_bs_oracle_spec(
    path: Path | str = FEM_BS_001_PROBLEM_SPEC_PATH,
    *,
    report: FEMParityReport | None = None,
) -> Path:
    """Write the deterministic problem spec to a public JSON fixture file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if report is None:
        payload = build_public_fem_bs_oracle_problem_spec()
    else:
        payload = build_public_fem_bs_oracle_problem_spec(
            refinement_levels=report.mesh_metadata.refinement_levels,
            time_steps=report.time_metadata.time_steps,
        )
    payload["contract_id"] = build_fixture_config_hash(payload)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def write_public_fem_bs_result_export(
    path: Path | str = FEM_BS_001_RESULT_EXPORT_PATH,
    *,
    refresh: bool = False,
    report: FEMParityReport | None = None,
) -> Path:
    """Run and write the FEM result export in a stable public artifact shape."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if (not target.exists()) or refresh:
        if report is None:
            report = run_public_black_scholes_parity_fixture()
        payload = report.export_payload()
        payload["config_id"] = report.config_hash
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def run_public_black_scholes_parity_fixture(
    *,
    refinement_levels: tuple[int, ...] = (4, 5, 6),
    time_steps: int = 80,
    refresh_exports: bool = False,
) -> FEMParityReport:
    """Run the public-synthetic Black-Scholes FEM parity fixture.

    The existing FEM implementation is formulated in normalized spot/strike coordinates.
    The fixture keeps spot=strike=1 in solver coordinates and scales the final value by
    strike=100 for the public fixture contract.
    """

    if not refinement_levels:
        raise ValueError("at least one refinement level is required")
    if any(level < 1 for level in refinement_levels):
        raise ValueError("all refinement levels must be positive")
    if time_steps <= 0:
        raise ValueError("time_steps must be positive")

    rows = tuple(
        _run_row(refinement_level=level, time_steps=time_steps) for level in refinement_levels
    )
    final = rows[-1]

    weak_form = _public_weak_form_metadata()
    boundaries = _public_boundary_metadata()
    mesh_metadata = MeshMetadata(
        mesh_family="line_uniform",
        element_family="lagrange_p2",
        domain_min=0.0,
        domain_max=4.0,
        spatial_domain="[0, 4.0] normalized spot",
        min_refinement_level=min(refinement_levels),
        max_refinement_level=max(refinement_levels),
        refinement_levels=tuple(refinement_levels),
        solver_backing="scikit-fem+sparse-direct",
    )
    time_metadata = TimeMetadata(
        integrator="theta_crank_nicolson",
        theta=0.5,
        time_steps=time_steps,
        start_time=0.0,
        end_time=1.0,
        time_domain="[0, 1]",
    )
    sensitivity_reference_policy = SensitivityReferencePolicy(
        policy_id="finite_difference_central_stencil_v1",
        delta_reference="analytical Black-Scholes delta at valuation time, S=K",
        gamma_reference="analytical Black-Scholes gamma at valuation time, S=K",
        extraction_method="central finite-difference in normalized strike space",
        allowed_fallback="explicit kink-aware boundary derivative remains separate evidence",
    )
    comparison_policy = ComparisonPolicy(
        policy_id="equal-error-budget-v1",
        mode="equal_error",
        primary_metric="price/Delta/Gamma absolute error",
        tolerance=DEFAULT_TOLERANCE_ABSOLUTE,
        note="Meshes and time steps are compared at matching error budgets (default absolute policy tolerances).",
        metric_tolerances=(
            ("price_absolute", DEFAULT_TOLERANCE_ABSOLUTE),
            ("price_relative", DEFAULT_TOLERANCE_RELATIVE),
            ("delta_absolute", DEFAULT_DELTA_TOLERANCE_ABSOLUTE),
            ("gamma_absolute", DEFAULT_GAMMA_TOLERANCE_ABSOLUTE),
        ),
    )

    diagnostics: dict[str, str | int | float] = {
        "mesh_family": "line_uniform",
        "element_family": "lagrange_p2",
        "time_integrator": "theta_crank_nicolson",
        "linear_solver": "scikit_fem_scipy_direct",
        "boundary_conditions": "named_endpoint_dirichlet_projection; upper endpoint uses exact Black-Scholes value as far-field linear-growth proxy",
        "boundary_nodes_enforced": len(boundaries),
        "weak_form_sign_convention": weak_form.sign_convention,
        "coordinate_transform": weak_form.coordinate_transform,
        "spot": 100.0,
        "strike": 100.0,
        "rate": 0.05,
        "volatility": 0.2,
        "maturity": 1.0,
        "deterministic_seed": 1729,
        "source_issue": "googa27/finite_element_options#64",
        "oracle_issue": "googa27/haircut-engine#108",
        "greek_policy": sensitivity_reference_policy.policy_id,
    }

    report = FEMParityReport(
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
        delta_tolerance_absolute=DEFAULT_DELTA_TOLERANCE_ABSOLUTE,
        expected_gamma=final.expected_gamma,
        observed_gamma=final.observed_gamma,
        gamma_absolute_error=final.gamma_absolute_error,
        gamma_tolerance_absolute=DEFAULT_GAMMA_TOLERANCE_ABSOLUTE,
        tolerance_absolute=DEFAULT_TOLERANCE_ABSOLUTE,
        tolerance_relative=DEFAULT_TOLERANCE_RELATIVE,
        convergence_rows=rows,
        diagnostics=diagnostics,
        weak_form=weak_form,
        mesh_metadata=mesh_metadata,
        time_metadata=time_metadata,
        boundaries=boundaries,
        sensitivity_reference_policy=sensitivity_reference_policy,
        comparison_policy=comparison_policy,
        config_hash="",  # filled below
    )
    config_hash = _config_hash(report)
    report = FEMParityReport(
        **{
            **report.__dict__,
            "config_hash": config_hash,
        }
    )

    if refresh_exports:
        write_public_fem_bs_oracle_spec(path=FEM_BS_001_PROBLEM_SPEC_PATH, report=report)
        write_public_fem_bs_result_export(
            path=FEM_BS_001_RESULT_EXPORT_PATH, refresh=True, report=report
        )

    return report


def _config_hash(report: FEMParityReport) -> str:
    payload = {
        "benchmark_id": report.benchmark_id,
        "problem_id": report.problem_id,
        "privacy_class": report.privacy_class,
        "weak_form": report.weak_form.to_public_dict(),
        "mesh_metadata": report.mesh_metadata.to_public_dict(),
        "refinement_levels": list(report.mesh_metadata.refinement_levels),
        "time_metadata": report.time_metadata.to_public_dict(),
        "boundaries": [boundary.to_public_dict() for boundary in report.boundaries],
        "sensitivity_reference_policy": report.sensitivity_reference_policy.to_public_dict(),
        "comparison_policy": report.comparison_policy.to_public_dict(),
        "tolerances": {
            "absolute": report.tolerance_absolute,
            "relative": report.tolerance_relative,
            "delta": report.delta_tolerance_absolute,
            "gamma": report.gamma_tolerance_absolute,
        },
    }
    return build_fixture_config_hash(payload)


def _public_weak_form_metadata() -> WeakFormMetadata:
    return WeakFormMetadata(
        equation_id="black_scholes_weak_form_transformed",
        sign_convention="existing_forward_tau_identity_transform_black_scholes_forms",
        time_transformation="tau = T - t",
        coordinate_transform="identity",
    )


def _public_boundary_metadata() -> tuple[BoundaryMetadata, ...]:
    return (
        BoundaryMetadata(
            location="S=0", condition_type="dirichlet", expression="0", enforced_nodes=1
        ),
        BoundaryMetadata(
            location="S=S_max",
            condition_type="dirichlet",
            expression="linear_growth",
            enforced_nodes=1,
        ),
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
        solution = ThetaScheme(theta=0.5).solve(
            times, space, boundary_condition=DirichletBC(["left", "right"])
        )
    spot_node = int(np.argmin(np.abs(space.Vh.doflocs[0] - 1.0)))
    normalized_price = float(solution[-1, spot_node])
    observed_delta, observed_gamma = _central_difference_greeks(
        space.Vh.doflocs[0], solution[-1], strike
    )
    observed_price = strike * normalized_price
    expected_price = EXPECTED_BLACK_SCHOLES_CALL_PRICE
    expected_delta = float(option.call_delta(1.0, 1.0, dynamics.sig**2))
    expected_gamma = float(
        spst.norm.pdf(option.d1(1.0, 1.0, dynamics.sig**2)) / (strike * dynamics.sig)
    )
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


def _central_difference_greeks(
    dof_locations: np.ndarray, values: np.ndarray, strike: float
) -> tuple[float, float]:
    """Return central finite-element Delta and Gamma at normalized spot one."""

    order = np.argsort(dof_locations)
    coordinates = dof_locations[order]
    ordered_values = values[order]
    center = int(np.argmin(np.abs(coordinates - 1.0)))
    if center == 0 or center == len(coordinates) - 1:
        raise ValueError("spot=1.0 must have neighboring FEM nodes for Greek extraction")
    delta = (ordered_values[center + 1] - ordered_values[center - 1]) / (
        coordinates[center + 1] - coordinates[center - 1]
    )
    gamma_normalized = (
        2.0
        * (
            (ordered_values[center + 1] - ordered_values[center])
            / (coordinates[center + 1] - coordinates[center])
            - (ordered_values[center] - ordered_values[center - 1])
            / (coordinates[center] - coordinates[center - 1])
        )
        / (coordinates[center + 1] - coordinates[center - 1])
    )
    return float(delta), float(gamma_normalized / strike)


__all__ = [
    "DEFAULT_TOLERANCE_ABSOLUTE",
    "DEFAULT_TOLERANCE_RELATIVE",
    "DEFAULT_DELTA_TOLERANCE_ABSOLUTE",
    "DEFAULT_GAMMA_TOLERANCE_ABSOLUTE",
    "EXPECTED_BLACK_SCHOLES_CALL_PRICE",
    "PUBLIC_SYNTHETIC_BLACK_SCHOLES_BENCHMARK_ID",
    "PUBLIC_SYNTHETIC_PROBLEM_ID",
    "FEMParityConvergenceRow",
    "FEMParityReport",
    "BoundaryMetadata",
    "WeakFormMetadata",
    "MeshMetadata",
    "TimeMetadata",
    "SensitivityReferencePolicy",
    "ComparisonPolicy",
    "build_public_fem_bs_oracle_problem_spec",
    "build_fixture_config_hash",
    "write_public_fem_bs_oracle_spec",
    "write_public_fem_bs_result_export",
    "run_public_black_scholes_parity_fixture",
    "FEM_BS_001_PROBLEM_SPEC_PATH",
    "FEM_BS_001_RESULT_EXPORT_PATH",
]
