"""Public-synthetic solver-cache and factorization benchmark evidence.

This module exercises the released one-dimensional line-uniform/Lagrange-P2
FEM route with SciPy sparse-direct factorization reuse.  AMG, PETSc and banded
routes are represented by the capability manifest as unsupported/fail-closed
until their own residual and equal-error evidence exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from time import perf_counter
from typing import Any

import numpy as np
import scipy.stats as spst  # type: ignore[import-untyped]

from finite_element_options.contracts import DEFAULT_FEM_CAPABILITY_MANIFEST
from finite_element_options.core.dynamics_black_scholes import (
    DynamicsParametersBlackScholes,
)
from finite_element_options.core.market import Market
from finite_element_options.core.vanilla_bs import EuropeanOptionBs
from finite_element_options.space.boundary import DirichletBC
from finite_element_options.space.mesh import create_mesh
from finite_element_options.space.solver import SpaceSolver
from finite_element_options.time_integration.stepper import LinearSolveDiagnostics, ThetaScheme

SOLVER_CACHE_BENCHMARK_ID = "FEM-SOLVER-CACHE-001"
SOLVER_CACHE_CONTRACT_VERSION = "solver-cache-benchmark/v1"


@dataclass(frozen=True)
class SolverCacheBenchmarkCase:
    """Public-synthetic benchmark controls for repeated 1D line-uniform solves."""

    benchmark_id: str = SOLVER_CACHE_BENCHMARK_ID
    problem_id: str = "public-synthetic-solver-cache-black-scholes-v0"
    privacy_class: str = "public_synthetic"
    refinement_level: int = 5
    time_steps: int = 40
    repeats: int = 3
    spot: float = 1.0
    strike: float = 1.0
    domain_max: float = 4.0
    rate: float = 0.05
    volatility: float = 0.2
    maturity: float = 1.0
    price_abs_tolerance: float = 4.0e-3
    residual_abs_tolerance: float = 1.0e-10

    def __post_init__(self) -> None:
        """Validate benchmark controls before mesh or assembly allocation."""

        if self.refinement_level < 1:
            raise ValueError("refinement_level must be positive")
        if self.time_steps <= 0:
            raise ValueError("time_steps must be positive")
        if self.repeats <= 0:
            raise ValueError("repeats must be positive")
        if min(self.spot, self.strike, self.domain_max, self.volatility, self.maturity) <= 0.0:
            raise ValueError("spot, strike, domain_max, volatility and maturity must be positive")


@dataclass(frozen=True)
class SolverCacheBenchmarkRow:
    """One repeated solve row with factorization-cache diagnostics."""

    repeat_index: int
    degrees_of_freedom: int
    nonzeros: int
    assembly_count: int
    factorization_count: int
    factorization_reuse_count: int
    solve_count: int
    max_linear_residual_abs: float
    observed_price: float
    expected_price: float
    price_absolute_error: float
    assembly_cache_key: str
    factorization_cache_key: str
    stage_timings_sec: dict[str, float]

    @property
    def accepted(self) -> bool:
        """Whether the row demonstrates factorization reuse and finite residual."""

        return (
            self.assembly_count == 1
            and self.factorization_count == 1
            and self.factorization_reuse_count == self.solve_count - 1
            and self.max_linear_residual_abs < 1.0e-10
        )

    def to_public_dict(self) -> dict[str, Any]:
        """Return a JSON-safe row payload."""

        return {
            "repeat_index": self.repeat_index,
            "degrees_of_freedom": self.degrees_of_freedom,
            "nonzeros": self.nonzeros,
            "assembly_count": self.assembly_count,
            "factorization_count": self.factorization_count,
            "factorization_reuse_count": self.factorization_reuse_count,
            "solve_count": self.solve_count,
            "max_linear_residual_abs": self.max_linear_residual_abs,
            "observed_price": self.observed_price,
            "expected_price": self.expected_price,
            "price_absolute_error": self.price_absolute_error,
            "assembly_cache_key": self.assembly_cache_key,
            "factorization_cache_key": self.factorization_cache_key,
            "stage_timings_sec": dict(self.stage_timings_sec),
            "accepted": self.accepted,
        }


@dataclass(frozen=True)
class SolverCacheBenchmarkReport:
    """Repeated-solve benchmark report and residual acceptance evidence."""

    case: SolverCacheBenchmarkCase
    rows: tuple[SolverCacheBenchmarkRow, ...]
    unsupported_solver_routes: tuple[dict[str, str | bool | None], ...]
    config_hash: str

    @property
    def accepted(self) -> bool:
        """Return true when all repeated solves satisfy acceptance budgets."""

        return all(row.accepted for row in self.rows) and all(
            row.price_absolute_error <= self.case.price_abs_tolerance for row in self.rows
        )

    def to_public_dict(self) -> dict[str, Any]:
        """Return a public-synthetic benchmark evidence payload."""

        return {
            "contract_version": SOLVER_CACHE_CONTRACT_VERSION,
            "benchmark_id": self.case.benchmark_id,
            "problem_id": self.case.problem_id,
            "privacy_class": self.case.privacy_class,
            "config_hash": self.config_hash,
            "accepted": self.accepted,
            "route": {
                "mesh_family": "line_uniform",
                "element_family": "lagrange_p2",
                "time_integrator": "theta_crank_nicolson",
                "linear_solver": "scipy_direct",
                "factorization_cache": "per invariant theta-system splu reuse",
            },
            "case": {
                "refinement_level": self.case.refinement_level,
                "time_steps": self.case.time_steps,
                "repeats": self.case.repeats,
                "spot": self.case.spot,
                "strike": self.case.strike,
                "domain_max": self.case.domain_max,
                "rate": self.case.rate,
                "volatility": self.case.volatility,
                "maturity": self.case.maturity,
                "price_abs_tolerance": self.case.price_abs_tolerance,
                "residual_abs_tolerance": self.case.residual_abs_tolerance,
            },
            "rows": [row.to_public_dict() for row in self.rows],
            "unsupported_solver_routes": list(self.unsupported_solver_routes),
        }


def run_solver_cache_benchmark(
    *, case: SolverCacheBenchmarkCase | None = None
) -> SolverCacheBenchmarkReport:
    """Run repeated 1D FEM solves and capture cache/factorization evidence."""

    case = case or SolverCacheBenchmarkCase()
    rows = tuple(_run_row(case, repeat_index=index) for index in range(case.repeats))
    unsupported_routes = tuple(
        backend.to_public_dict()
        for backend in DEFAULT_FEM_CAPABILITY_MANIFEST.solver_backends
        if backend.name != "scipy_direct"
    )
    report = SolverCacheBenchmarkReport(
        case=case,
        rows=rows,
        unsupported_solver_routes=unsupported_routes,
        config_hash="",
    )
    return SolverCacheBenchmarkReport(**{**report.__dict__, "config_hash": _config_hash(report)})


def _run_row(case: SolverCacheBenchmarkCase, *, repeat_index: int) -> SolverCacheBenchmarkRow:
    started_assembly = perf_counter()
    dynamics = DynamicsParametersBlackScholes(r=case.rate, q=0.0, sig=case.volatility)
    market = Market(r=dynamics.r)
    option = EuropeanOptionBs(k=case.strike, q=dynamics.q, mkt=market)
    mesh, config = create_mesh([case.domain_max], case.refinement_level)
    mesh = mesh.with_boundaries(
        {
            "left": lambda x: np.isclose(x[0], 0.0),
            "right": lambda x: np.isclose(x[0], case.domain_max),
        }
    )
    space = SpaceSolver(mesh, dynamics, option, is_call=True, config=config)
    assembly_time = perf_counter() - started_assembly
    times = np.linspace(0.0, case.maturity, case.time_steps + 1)
    stepper = ThetaScheme(theta=0.5, linear_solver="scipy_direct", reuse_factorization=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        solution = stepper.solve(times, space, boundary_condition=DirichletBC(["left", "right"]))
    diagnostics = stepper.last_solve_diagnostics
    spot_node = int(np.argmin(np.abs(space.Vh.doflocs[0] - case.spot)))
    observed_price = float(solution[-1, spot_node])
    expected_price = float(option.call(case.maturity, case.spot, dynamics.sig**2))
    stage_timings = _stage_timings(assembly_time, diagnostics)
    return SolverCacheBenchmarkRow(
        repeat_index=repeat_index,
        degrees_of_freedom=int(space.Vh.N),
        nonzeros=int(space.stiffness.nnz),
        assembly_count=1,
        factorization_count=diagnostics.factorization_count,
        factorization_reuse_count=diagnostics.factorization_reuse_count,
        solve_count=diagnostics.solve_count,
        max_linear_residual_abs=diagnostics.max_linear_residual_abs,
        observed_price=observed_price,
        expected_price=expected_price,
        price_absolute_error=abs(observed_price - expected_price),
        assembly_cache_key=diagnostics.assembly_cache_key,
        factorization_cache_key=diagnostics.factorization_cache_key,
        stage_timings_sec=stage_timings,
    )


def _stage_timings(assembly_time: float, diagnostics: LinearSolveDiagnostics) -> dict[str, float]:
    return {
        "assembly": assembly_time,
        "factorization": diagnostics.stage_timings_sec["factorization"],
        "solve": diagnostics.stage_timings_sec["solve"],
    }


def _config_hash(report: SolverCacheBenchmarkReport) -> str:
    payload = {
        "benchmark_id": report.case.benchmark_id,
        "problem_id": report.case.problem_id,
        "refinement_level": report.case.refinement_level,
        "time_steps": report.case.time_steps,
        "repeats": report.case.repeats,
        "route": "line_uniform/lagrange_p2/theta/scipy_direct",
        "row_keys": [
            {
                "dofs": row.degrees_of_freedom,
                "nnz": row.nonzeros,
                "assembly_cache_key": row.assembly_cache_key,
                "factorization_cache_key": row.factorization_cache_key,
            }
            for row in report.rows
        ],
    }
    return sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def analytical_gamma(case: SolverCacheBenchmarkCase) -> float:
    """Return the analytical Black-Scholes Gamma at the benchmark spot."""

    dynamics = DynamicsParametersBlackScholes(r=case.rate, q=0.0, sig=case.volatility)
    market = Market(r=dynamics.r)
    option = EuropeanOptionBs(k=case.strike, q=dynamics.q, mkt=market)
    return float(
        spst.norm.pdf(option.d1(case.maturity, case.spot, dynamics.sig**2))
        / (case.spot * dynamics.sig)
    )
