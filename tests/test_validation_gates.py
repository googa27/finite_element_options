"""Numerical validation gates for Project #5 issue #42."""

from __future__ import annotations

from dataclasses import replace

import pytest

from finite_element_options.contracts.capability_matrix import (
    DEFAULT_CAPABILITY_RECORDS,
    CapabilityStatus,
)
from finite_element_options.time_integration import LCPDiagnostics
from finite_element_options.validation.verification_gates import (
    BackendValidationReport,
    BenchmarkSpec,
    ConvergenceRow,
    ConvergenceStudy,
    OptionSurfacePoint,
    ValidationGateError,
    audit_capability_benchmark_coverage,
    compare_backend_reports,
    default_benchmark_registry,
    evaluate_american_lcp_gate,
    evaluate_call_arbitrage,
    manufactured_solution_cases,
)


def test_default_validation_registry_covers_all_validated_benchmark_ids() -> None:
    registry = default_benchmark_registry()
    records = tuple(
        record
        for record in DEFAULT_CAPABILITY_RECORDS
        if record.status in {CapabilityStatus.VALIDATED, CapabilityStatus.PRODUCTION}
    )

    audit = audit_capability_benchmark_coverage(records, registry)

    assert audit.accepted
    assert audit.missing_benchmark_ids == ()
    assert audit.production_without_benchmark_ids == ()
    assert audit.validated_without_benchmark_ids == ()
    for benchmark_id, spec in registry.items():
        payload = spec.to_public_dict()
        assert payload["benchmark_id"] == benchmark_id
        assert payload["model"]
        assert payload["instrument"]
        assert payload["state_convention"]
        assert payload["domain"]
        assert payload["grid"]
        assert payload["time_schedule"]
        assert payload["oracle"]
        assert payload["norm"]
        assert set(payload["tolerance_components"]) == {
            "discretization",
            "oracle",
            "floating_point",
        }


def test_capability_audit_fails_closed_for_missing_production_benchmark_metadata() -> None:
    incomplete = BenchmarkSpec(
        benchmark_id="MISSING-META",
        model="Black-Scholes",
        instrument="European call",
        state_convention="forward tau",
        domain="[0, 4K]",
        grid="line_uniform",
        time_schedule="theta",
        oracle="analytical",
        norm="linf",
        expected_order=2.0,
        tolerance_components={"discretization": 1.0e-3},
    )

    with pytest.raises(ValidationGateError, match="tolerance components"):
        incomplete.validate()

    validated_without = replace(
        DEFAULT_CAPABILITY_RECORDS[0],
        status=CapabilityStatus.VALIDATED,
        benchmark_ids=(),
    )
    audit = audit_capability_benchmark_coverage(
        (validated_without,),
        default_benchmark_registry(),
    )

    assert not audit.accepted
    assert audit.validated_without_benchmark_ids == (validated_without.capability_id,)


def test_manufactured_solution_cases_cover_required_operator_families() -> None:
    cases = manufactured_solution_cases()
    assert set(cases) >= {
        "diffusion",
        "convection_diffusion",
        "mixed_derivative",
        "state_dependent_reaction",
    }

    for case in cases.values():
        for point in case.sample_points:
            assert abs(case.residual(point)) <= case.residual_tolerance, case.operator_family

    unsupported = next(iter(cases.values())).__class__(
        operator_family="not_a_declared_operator",
        equation="nonsense",
        sample_points=((0.5, 0.25),),
        residual_tolerance=1.0e-12,
    )
    with pytest.raises(ValidationGateError, match="unsupported manufactured operator"):
        unsupported.residual((0.5, 0.25))


def test_convergence_study_separates_spatial_temporal_and_domain_error_budgets() -> None:
    spatial = ConvergenceStudy(
        benchmark_id="FEM-VALIDATION-GATES-V0",
        dimension="spatial",
        expected_order=2.0,
        order_tolerance=0.1,
        rows=(
            ConvergenceRow(resolution=16, step=1.0 / 16.0, error=4.0e-3),
            ConvergenceRow(resolution=32, step=1.0 / 32.0, error=1.0e-3),
            ConvergenceRow(resolution=64, step=1.0 / 64.0, error=2.5e-4),
        ),
        tolerance_components={
            "discretization": 5.0e-4,
            "oracle": 1.0e-8,
            "floating_point": 1.0e-10,
        },
    )
    temporal = ConvergenceStudy(
        benchmark_id="FEM-VALIDATION-GATES-V0",
        dimension="temporal",
        expected_order=1.0,
        order_tolerance=0.1,
        rows=(
            ConvergenceRow(resolution=20, step=1.0 / 20.0, error=2.0e-3),
            ConvergenceRow(resolution=40, step=1.0 / 40.0, error=1.0e-3),
            ConvergenceRow(resolution=80, step=1.0 / 80.0, error=5.0e-4),
        ),
        tolerance_components={
            "discretization": 7.0e-4,
            "oracle": 1.0e-8,
            "floating_point": 1.0e-10,
        },
    )
    domain = ConvergenceStudy(
        benchmark_id="FEM-VALIDATION-GATES-V0",
        dimension="domain_truncation",
        expected_order=0.0,
        order_tolerance=0.0,
        rows=(
            ConvergenceRow(resolution=4, step=4.0, error=8.0e-4),
            ConvergenceRow(resolution=6, step=6.0, error=2.0e-4),
            ConvergenceRow(resolution=8, step=8.0, error=5.0e-5),
        ),
        tolerance_components={
            "discretization": 1.0e-4,
            "oracle": 1.0e-8,
            "floating_point": 1.0e-10,
        },
    )

    for study in (spatial, temporal, domain):
        report = study.evaluate()
        assert report.accepted
        assert report.dimension == study.dimension
        assert report.final_error <= report.error_budget
        assert report.rows

    failed = ConvergenceStudy(
        benchmark_id="FEM-VALIDATION-GATES-V0",
        dimension="spatial",
        expected_order=2.0,
        order_tolerance=0.1,
        rows=(
            ConvergenceRow(resolution=16, step=1.0 / 16.0, error=4.0e-3),
            ConvergenceRow(resolution=32, step=1.0 / 32.0, error=3.0e-3),
            ConvergenceRow(resolution=64, step=1.0 / 64.0, error=2.0e-3),
        ),
        tolerance_components={
            "discretization": 5.0e-4,
            "oracle": 1.0e-8,
            "floating_point": 1.0e-10,
        },
    )

    with pytest.raises(ValidationGateError) as excinfo:
        failed.require_passed()
    assert "spatial" in str(excinfo.value)
    assert failed.evaluate().actionable_table[0]["observed_order"] is None

    infinite_budget = ConvergenceStudy(
        benchmark_id="FEM-VALIDATION-GATES-V0",
        dimension="spatial",
        expected_order=0.0,
        order_tolerance=0.0,
        rows=(
            ConvergenceRow(resolution=16, step=1.0 / 16.0, error=1.0e8),
            ConvergenceRow(resolution=32, step=1.0 / 32.0, error=1.0e8),
        ),
        tolerance_components={
            "discretization": float("inf"),
            "oracle": 1.0e-8,
            "floating_point": 1.0e-10,
        },
    )
    with pytest.raises(ValidationGateError, match="invalid tolerance component"):
        infinite_budget.evaluate()

    invalid_order = replace(spatial, order_tolerance=float("nan"))
    with pytest.raises(ValidationGateError, match="invalid convergence order controls"):
        invalid_order.evaluate()


def test_arbitrage_gate_accepts_valid_call_surface_and_blocks_violations() -> None:
    valid = (
        OptionSurfacePoint(
            spot=90.0,
            strike=100.0,
            rate=0.03,
            maturity=1.0,
            price=5.0,
            delta=0.35,
            gamma=0.02,
        ),
        OptionSurfacePoint(
            spot=100.0,
            strike=100.0,
            rate=0.03,
            maturity=1.0,
            price=10.0,
            delta=0.55,
            gamma=0.018,
        ),
        OptionSurfacePoint(
            spot=110.0,
            strike=100.0,
            rate=0.03,
            maturity=1.0,
            price=17.0,
            delta=0.72,
            gamma=0.015,
        ),
    )

    report = evaluate_call_arbitrage(valid)

    assert report.accepted
    assert report.failures == ()

    invalid = (
        valid[0],
        OptionSurfacePoint(
            spot=100.0,
            strike=100.0,
            rate=0.03,
            maturity=1.0,
            price=4.0,
            delta=1.2,
            gamma=-0.01,
        ),
    )
    with pytest.raises(ValidationGateError, match="arbitrage"):
        evaluate_call_arbitrage(invalid, fail_on_error=True)

    duplicate_spot = (
        valid[0],
        OptionSurfacePoint(
            spot=90.0,
            strike=100.0,
            rate=0.03,
            maturity=1.0,
            price=5.2,
            delta=0.36,
            gamma=0.02,
        ),
    )
    duplicate_report = evaluate_call_arbitrage(duplicate_spot)
    assert not duplicate_report.accepted
    assert any("duplicate spot" in failure for failure in duplicate_report.failures)

    mixed_contract = (
        valid[0],
        OptionSurfacePoint(
            spot=100.0,
            strike=105.0,
            rate=0.03,
            maturity=1.0,
            price=8.0,
            delta=0.45,
            gamma=0.02,
        ),
    )
    with pytest.raises(ValidationGateError, match="same strike/rate/maturity"):
        evaluate_call_arbitrage(mixed_contract, fail_on_error=True)


def test_cross_backend_gate_requires_identical_conventions_and_toleranced_values() -> None:
    scikit = BackendValidationReport(
        benchmark_id="fem-bs-001",
        backend_id="scikit-fem+sparse-direct",
        pde_convention_hash="same-convention",
        grid_hash="same-grid",
        time_schedule_hash="same-time",
        values={"price": 10.4505, "delta": 0.6368, "gamma": 0.0187},
    )
    fenicsx = BackendValidationReport(
        benchmark_id="fem-bs-001",
        backend_id="fenicsx+PETSc",
        pde_convention_hash="same-convention",
        grid_hash="same-grid",
        time_schedule_hash="same-time",
        values={"price": 10.4507, "delta": 0.6367, "gamma": 0.01872},
    )

    report = compare_backend_reports(
        scikit, fenicsx, tolerances={"price": 5.0e-4, "delta": 5.0e-4, "gamma": 5.0e-5}
    )

    assert report.accepted
    assert report.max_abs_difference <= 5.0e-4

    mismatched = BackendValidationReport(
        benchmark_id="fem-bs-001",
        backend_id="fenicsx+PETSc",
        pde_convention_hash="wrong-time-direction",
        grid_hash="same-grid",
        time_schedule_hash="same-time",
        values=fenicsx.values,
    )
    with pytest.raises(ValidationGateError, match="identical PDE conventions"):
        compare_backend_reports(scikit, mismatched, tolerances={"price": 1.0})

    extra_metric = BackendValidationReport(
        benchmark_id="fem-bs-001",
        backend_id="fenicsx+PETSc",
        pde_convention_hash="same-convention",
        grid_hash="same-grid",
        time_schedule_hash="same-time",
        values={**fenicsx.values, "theta": 0.5},
    )
    with pytest.raises(ValidationGateError, match="identical metric sets"):
        compare_backend_reports(
            scikit, extra_metric, tolerances={"price": 1.0, "delta": 1.0, "gamma": 1.0}
        )

    with pytest.raises(ValidationGateError, match="invalid tolerance"):
        compare_backend_reports(
            scikit, fenicsx, tolerances={"price": float("nan"), "delta": 1.0, "gamma": 1.0}
        )


def test_american_lcp_gate_requires_complementarity_and_exercise_front_diagnostics() -> None:
    diagnostics = LCPDiagnostics(
        success=True,
        iterations=12,
        tolerance=1.0e-8,
        relaxation=0.5,
        primal_violation_max=0.0,
        dual_violation_max=0.0,
        complementarity_max=1.0e-10,
        projected_residual_max=1.0e-10,
        max_update=1.0e-10,
        exercise_count=2,
        exercise_set=(True, False, True),
        message="projected SOR converged",
        solve_time_sec=0.001,
    )

    report = evaluate_american_lcp_gate("FEM-AMERICAN-LCP", (diagnostics,))

    assert report.accepted
    assert report.exercise_front_observed

    bad = LCPDiagnostics(
        **{
            **diagnostics.__dict__,
            "complementarity_max": 1.0e-3,
            "exercise_count": 0,
            "exercise_set": (False, False, False),
        }
    )
    with pytest.raises(ValidationGateError, match="complementarity"):
        evaluate_american_lcp_gate("FEM-AMERICAN-LCP", (bad,), fail_on_error=True)

    all_exercised = LCPDiagnostics(
        **{
            **diagnostics.__dict__,
            "exercise_count": 3,
            "exercise_set": (True, True, True),
        }
    )
    with pytest.raises(ValidationGateError, match="exercise front"):
        evaluate_american_lcp_gate("FEM-AMERICAN-LCP", (all_exercised,), fail_on_error=True)

    nonfinite = LCPDiagnostics(
        **{
            **diagnostics.__dict__,
            "tolerance": float("nan"),
            "primal_violation_max": float("nan"),
            "dual_violation_max": float("nan"),
            "complementarity_max": float("nan"),
            "projected_residual_max": float("nan"),
        }
    )
    with pytest.raises(ValidationGateError, match="non-finite"):
        evaluate_american_lcp_gate("FEM-AMERICAN-LCP", (nonfinite,), fail_on_error=True)
