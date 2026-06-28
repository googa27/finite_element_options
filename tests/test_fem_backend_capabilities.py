"""FEM capability-manifest, QuantProblemSpec mapping and parity tests."""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from src.contracts import (  # noqa: E402
    DEFAULT_FEM_CAPABILITY_MANIFEST,
    FEMRouteRequest,
    UnsupportedReason,
    UnsupportedRouteError,
    diagnose_unsupported_route,
    ensure_route_supported,
)
from src.validation.black_scholes_parity import (  # noqa: E402
    PUBLIC_SYNTHETIC_BLACK_SCHOLES_BENCHMARK_ID,
    run_public_black_scholes_parity_fixture,
)

FIXTURE_DIR = pathlib.Path(__file__).parent / "fixtures" / "quant_problem_specs"


def _supported_payload() -> dict[str, object]:
    return {
        "schema_version": "quant-problem-spec/v0",
        "valuation_context": {
            "measure": "risk_neutral",
            "numeraire": "money_market_account",
            "units": {"underlying": "USD", "time": "ACT/365F"},
            "valuation_date": "2026-01-02",
            "maturity_date": "2027-01-02",
        },
        "mathematical_problem": {
            "dimension": 1,
            "pde_terms": ["drift", "diffusion", "reaction"],
            "boundary_conditions": ["dirichlet"],
            "exercise_style": "european",
        },
        "solver_plan": {
            "mesh_family": "line_uniform",
            "element_family": "lagrange_p2",
            "requested_outputs": ["value", "delta", "gamma"],
            "stability_controls": ["theta"],
            "linear_solver": "scipy_direct",
        },
    }


def test_default_manifest_declares_fem_support_without_claiming_unvalidated_routes() -> None:
    manifest = DEFAULT_FEM_CAPABILITY_MANIFEST

    assert manifest.backend_id == "finite_element_options.fem_backend.v0"
    assert manifest.contract_version == "0.1.0"
    assert manifest.supported_dimensions == (1,)
    assert manifest.element_families == ("lagrange_p2",)
    assert "american" not in manifest.exercise_styles
    assert "hjb_control" not in manifest.pde_terms
    assert {"measure", "numeraire", "units", "valuation_date", "maturity_or_time_domain"} <= set(
        manifest.required_conventions
    )


def test_quant_problem_spec_mapping_preserves_conventions_mesh_and_outputs() -> None:
    request = FEMRouteRequest.from_quant_problem_spec(_supported_payload())

    assert request.dimension == 1
    assert request.mesh_family == "line_uniform"
    assert request.element_family == "lagrange_p2"
    assert request.pde_terms == ("drift", "diffusion", "reaction")
    assert request.boundary_conditions == ("dirichlet",)
    assert request.requested_outputs == ("value", "delta", "gamma")
    assert request.measure == "risk_neutral"
    assert request.numeraire == "money_market_account"
    assert request.units == {"underlying": "USD", "time": "ACT/365F"}
    assert request.valuation_date == "2026-01-02"
    assert request.maturity_date == "2027-01-02"
    assert diagnose_unsupported_route(request) == ()


def test_haircut_engine_vanilla_call_fixture_maps_to_supported_fem_route() -> None:
    """Consume the same public QuantProblemSpec fixture that Haircut Engine validates."""

    payload = json.loads((FIXTURE_DIR / "vanilla_call.json").read_text())

    request = FEMRouteRequest.from_quant_problem_spec(payload)

    assert request.source_schema_version == "quant-problem-spec/v0"
    assert request.dimension == 1
    assert request.mesh_family == "line_uniform"
    assert request.element_family == "lagrange_p2"
    assert request.pde_terms == ("drift", "diffusion", "reaction")
    assert request.boundary_conditions == ("dirichlet",)
    assert request.boundary_details == {"S=0": "0", "S=S_max": "linear growth"}
    assert request.requested_outputs == ("value", "delta", "gamma")
    assert request.measure == "risk_neutral_money_market"
    assert request.numeraire == "money_market_account_CLP"
    assert request.units["S"] == "CLP"
    assert request.valuation_date == "2026-06-30"
    assert request.time_domain == "[0, 1]"
    assert diagnose_unsupported_route(request) == ()


def test_unsupported_variational_terms_dimensions_boundaries_and_exercise_fail_closed() -> None:
    payload = _supported_payload()
    payload["mathematical_problem"] = {
        "dimension": 2,
        "pde_terms": ["drift", "hjb_control", "jump_integral"],
        "boundary_conditions": ["free_boundary"],
        "exercise_style": "american",
    }
    request = FEMRouteRequest.from_quant_problem_spec(payload)

    diagnostics = diagnose_unsupported_route(request)
    reasons = {diagnostic.reason for diagnostic in diagnostics}

    assert UnsupportedReason.UNSUPPORTED_DIMENSION in reasons
    assert UnsupportedReason.UNSUPPORTED_TERM in reasons
    assert UnsupportedReason.UNSUPPORTED_BOUNDARY in reasons
    assert UnsupportedReason.UNSUPPORTED_EXERCISE in reasons
    assert {diagnostic.field for diagnostic in diagnostics} >= {
        "dimension",
        "pde_terms",
        "boundary_conditions",
        "exercise_style",
    }
    with pytest.raises(UnsupportedRouteError, match="FEM backend supports dimensions") as exc_info:
        ensure_route_supported(request)
    assert exc_info.value.diagnostics == diagnostics


def test_missing_measure_numeraire_units_and_dates_are_actionable_diagnostics() -> None:
    payload = _supported_payload()
    payload["valuation_context"] = {}
    request = FEMRouteRequest.from_quant_problem_spec(payload)

    diagnostics = diagnose_unsupported_route(request)
    missing = {diagnostic.field for diagnostic in diagnostics if diagnostic.reason == UnsupportedReason.MISSING_CONVENTION}

    assert missing == {"measure", "numeraire", "units", "valuation_date", "maturity_or_time_domain"}
    assert all("missing or empty" in diagnostic.message for diagnostic in diagnostics)


def test_unsupported_outputs_mesh_element_and_solver_controls_do_not_silently_downgrade() -> None:
    payload = _supported_payload()
    payload["solver_plan"] = {
        "mesh_family": "adaptive_unstructured",
        "element_family": "discontinuous_galerkin",
        "requested_outputs": ["value", "vega", "mesh_error_indicator"],
        "stability_controls": ["adaptive_time", "rannacher"],
        "linear_solver": "petsc_amg",
    }
    request = FEMRouteRequest.from_quant_problem_spec(payload)

    diagnostics = diagnose_unsupported_route(request)
    by_field = {diagnostic.field: diagnostic for diagnostic in diagnostics}

    assert by_field["mesh_family"].reason == UnsupportedReason.UNSUPPORTED_MESH
    assert by_field["element_family"].reason == UnsupportedReason.UNSUPPORTED_ELEMENT
    assert by_field["linear_solver"].reason == UnsupportedReason.UNSUPPORTED_LINEAR_SOLVER
    assert {diagnostic.value for diagnostic in diagnostics if diagnostic.field == "requested_outputs"} == {
        "vega",
        "mesh_error_indicator",
    }
    assert {diagnostic.value for diagnostic in diagnostics if diagnostic.field == "stability_controls"} == {
        "adaptive_time",
        "rannacher",
    }


def test_public_black_scholes_parity_fixture_matches_analytical_oracle_with_evidence() -> None:
    report = run_public_black_scholes_parity_fixture()

    assert report.benchmark_id == PUBLIC_SYNTHETIC_BLACK_SCHOLES_BENCHMARK_ID
    assert report.problem_id == "public-synthetic-vanilla-call-v0"
    assert report.privacy_class == "public_synthetic"
    assert report.price_absolute_error <= report.tolerance_absolute
    assert report.price_relative_error <= report.tolerance_relative
    assert report.observed_price == pytest.approx(report.expected_price, abs=report.tolerance_absolute)
    assert report.delta_absolute_error <= report.delta_tolerance_absolute
    assert report.gamma_absolute_error <= report.gamma_tolerance_absolute
    assert report.observed_delta == pytest.approx(report.expected_delta, abs=report.delta_tolerance_absolute)
    assert report.observed_gamma == pytest.approx(report.expected_gamma, abs=report.gamma_tolerance_absolute)
    assert report.convergence_rows
    assert [row.absolute_error for row in report.convergence_rows] == sorted(
        (row.absolute_error for row in report.convergence_rows),
        reverse=True,
    )
    assert report.diagnostics["mesh_family"] == "line_uniform"
    assert report.diagnostics["element_family"] == "lagrange_p2"
    assert report.diagnostics["boundary_nodes_enforced"] == 2
    assert report.diagnostics["weak_form_sign_convention"] == "existing_forward_tau_identity_transform_black_scholes_forms"


def test_public_black_scholes_parity_fixture_is_deterministic() -> None:
    first = run_public_black_scholes_parity_fixture()
    second = run_public_black_scholes_parity_fixture()

    assert first.to_public_dict() == second.to_public_dict()
