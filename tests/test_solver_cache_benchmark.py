"""Solver-cache and residual benchmark acceptance evidence tests."""

from __future__ import annotations

import pytest

from finite_element_options.time_integration.stepper import ThetaScheme
from finite_element_options.validation.solver_cache_benchmark import (
    SOLVER_CACHE_BENCHMARK_ID,
    SolverCacheBenchmarkCase,
    run_solver_cache_benchmark,
)


def test_theta_scheme_rejects_unreleased_solver_routes() -> None:
    with pytest.raises(ValueError, match="AMG/PETSc/banded policies fail closed"):
        ThetaScheme(linear_solver="petsc")


def test_solver_cache_benchmark_reuses_one_factorization_per_repeated_solve() -> None:
    case = SolverCacheBenchmarkCase(refinement_level=5, time_steps=30, repeats=2)

    report = run_solver_cache_benchmark(case=case)

    assert report.case.benchmark_id == SOLVER_CACHE_BENCHMARK_ID
    assert report.accepted
    assert len(report.rows) == 2
    assert {row.assembly_cache_key for row in report.rows}
    assert len({row.assembly_cache_key for row in report.rows}) == 1
    assert len({row.factorization_cache_key for row in report.rows}) == 1
    for row in report.rows:
        assert row.assembly_count == 1
        assert row.factorization_count == 1
        assert row.factorization_reuse_count == case.time_steps - 1
        assert row.solve_count == case.time_steps
        assert row.max_linear_residual_abs <= case.residual_abs_tolerance
        assert row.price_absolute_error <= case.price_abs_tolerance
        assert row.stage_timings_sec["assembly"] >= 0.0
        assert row.stage_timings_sec["factorization"] >= 0.0
        assert row.stage_timings_sec["solve"] >= 0.0


def test_solver_cache_report_records_unsupported_optional_routes() -> None:
    report = run_solver_cache_benchmark(
        case=SolverCacheBenchmarkCase(refinement_level=4, time_steps=20, repeats=1)
    )
    payload = report.to_public_dict()
    unsupported_names = {route["name"] for route in payload["unsupported_solver_routes"]}

    assert payload["contract_version"] == "solver-cache-benchmark/v1"
    assert payload["route"]["linear_solver"] == "scipy_direct"
    assert {"scipy_banded", "amg", "petsc"} <= unsupported_names
    assert all(route["status"] == "unsupported" for route in payload["unsupported_solver_routes"])
