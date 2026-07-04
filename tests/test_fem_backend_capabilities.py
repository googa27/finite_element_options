"""FEM capability-manifest, QuantProblemSpec mapping and parity tests."""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from finite_element_options.contracts import (  # noqa: E402
    DEFAULT_FEM_CAPABILITY_MANIFEST,
    DEFAULT_RELEASED_FEM_SOLVER_CONTRACT,
    CapabilityStatus,
    FEMRouteRequest,
    UnsupportedReason,
    UnsupportedRouteError,
    diagnose_unsupported_route,
    ensure_route_supported,
)
from finite_element_options.validation import black_scholes_parity as parity_module  # noqa: E402
from finite_element_options.validation.black_scholes_parity import (  # noqa: E402
    FEM_BS_001_PROBLEM_SPEC_PATH,
    FEM_BS_001_RESULT_EXPORT_PATH,
    PUBLIC_SYNTHETIC_BLACK_SCHOLES_BENCHMARK_ID,
    build_public_fem_bs_oracle_problem_spec,
    build_fixture_config_hash,
    run_public_black_scholes_parity_fixture,
)

FIXTURE_DIR = pathlib.Path(__file__).parent / "fixtures" / "quant_problem_specs"
FEM_FIXTURE_DIR = pathlib.Path(__file__).parent / "fixtures" / "fem_bs_001"


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
    assert any("Pinares fixed-price proxy" in note for note in manifest.notes)
    assert any("obstacles/free boundaries" in note for note in manifest.notes)


def test_released_solver_contract_exposes_pinares_public_proxy_and_fail_closed_routes() -> None:
    contract = DEFAULT_RELEASED_FEM_SOLVER_CONTRACT
    manifest_payload = contract.to_public_dict()["capability_manifest"]
    solver_statuses = {
        backend.name: backend.status for backend in contract.manifest.solver_backends
    }

    assert contract.backend_id == DEFAULT_FEM_CAPABILITY_MANIFEST.backend_id
    assert contract.privacy_class == "public_synthetic"
    assert "PINARES-FEM-FIXED-PRICE-PROXY-V0" in contract.public_fixture_ids
    assert "PINARES-FEM-FAIL-CLOSED-V0" in contract.public_fixture_ids
    assert all("private" not in path for path in contract.public_fixture_paths)
    assert any("Pinares private modules" in item for item in contract.forbidden_dependencies)
    assert solver_statuses["scipy_direct"] == CapabilityStatus.VALIDATED
    assert solver_statuses["scipy_banded"] == CapabilityStatus.UNSUPPORTED
    assert solver_statuses["amg"] == CapabilityStatus.UNSUPPORTED
    assert solver_statuses["petsc"] == CapabilityStatus.UNSUPPORTED
    assert manifest_payload["solver_backends"][0]["factorization_reuse"] is True


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


def test_pinares_fixed_price_problem_fixture_maps_to_supported_fem_route() -> None:
    """Pinares publishes the problem; FEM only chooses mesh/weak-form controls."""

    payload = json.loads((FIXTURE_DIR / "pinares_fixed_price_proxy.json").read_text())

    request = FEMRouteRequest.from_quant_problem_spec(payload)

    assert payload["problem_id"] == "pinares.fixed_price_option_proxy.v1"
    assert payload["problem_hash"] == "publicsyntheticpinares001"
    assert request.source_schema_version == "quant-problem-spec/v0"
    assert request.backend_id == "finite_element_options.fem_backend.v0"
    assert request.dimension == 1
    assert request.mesh_family == "line_uniform"
    assert request.element_family == "lagrange_p2"
    assert request.pde_terms == ("drift", "diffusion", "reaction")
    assert request.boundary_conditions == ("dirichlet",)
    assert request.boundary_details == {"S=0": "dirichlet", "S=S_max": "linear_growth"}
    assert request.requested_outputs == ("value", "delta", "gamma")
    assert request.measure == "Q*"
    assert request.numeraire == "UF_money_market_account_proxy"
    assert request.units["S"] == "UF"
    assert request.valuation_date == "2026-06-30"
    assert request.time_domain == "[0, 1]"
    assert diagnose_unsupported_route(request) == ()


def test_wrong_backend_id_fails_closed_for_fem_route() -> None:
    payload = json.loads((FIXTURE_DIR / "pinares_fixed_price_proxy.json").read_text())
    payload["solver_plan"] = {
        **payload["solver_plan"],
        "backend_id": "finite_difference_options.fd_backend.v0",
    }
    request = FEMRouteRequest.from_quant_problem_spec(payload)

    diagnostics = diagnose_unsupported_route(request)
    assert any(
        diagnostic.reason == UnsupportedReason.UNSUPPORTED_BACKEND
        and diagnostic.field == "backend_id"
        and diagnostic.value == "finite_difference_options.fd_backend.v0"
        for diagnostic in diagnostics
    )


def test_empty_boundary_conditions_fail_closed_instead_of_defaulting_to_dirichlet() -> None:
    payload = _supported_payload()
    math_problem = payload["mathematical_problem"]
    assert isinstance(math_problem, dict)
    payload["mathematical_problem"] = {**math_problem, "boundary_conditions": []}

    request = FEMRouteRequest.from_quant_problem_spec(payload)
    diagnostics = diagnose_unsupported_route(request)

    assert request.boundary_conditions == ()
    assert any(
        diagnostic.reason == UnsupportedReason.UNSUPPORTED_BOUNDARY
        and diagnostic.field == "boundary_conditions"
        and diagnostic.value == "<missing>"
        for diagnostic in diagnostics
    )


def test_endpoint_linear_growth_is_classified_as_dirichlet_only_for_endpoint_locations() -> None:
    payload = _supported_payload()
    math_problem = payload["mathematical_problem"]
    assert isinstance(math_problem, dict)
    payload["mathematical_problem"] = {
        **math_problem,
        "boundary_conditions": {"S=0": "linear growth"},
    }

    request = FEMRouteRequest.from_quant_problem_spec(payload)

    assert request.boundary_conditions == ("dirichlet",)
    assert diagnose_unsupported_route(request) == ()


def test_free_boundary_text_is_not_misclassified_as_dirichlet() -> None:
    payload = _supported_payload()
    math_problem = payload["mathematical_problem"]
    assert isinstance(math_problem, dict)
    payload["mathematical_problem"] = {
        **math_problem,
        "boundary_conditions": {"S=S_star": "free boundary with linear continuation"},
    }

    request = FEMRouteRequest.from_quant_problem_spec(payload)
    diagnostics = diagnose_unsupported_route(request)

    assert request.boundary_conditions == ("free_boundary",)
    assert any(
        diagnostic.reason == UnsupportedReason.UNSUPPORTED_BOUNDARY for diagnostic in diagnostics
    )


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
    missing = {
        diagnostic.field
        for diagnostic in diagnostics
        if diagnostic.reason == UnsupportedReason.MISSING_CONVENTION
    }

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
    assert {
        diagnostic.value for diagnostic in diagnostics if diagnostic.field == "requested_outputs"
    } == {
        "vega",
        "mesh_error_indicator",
    }
    assert {
        diagnostic.value for diagnostic in diagnostics if diagnostic.field == "stability_controls"
    } == {
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
    assert report.observed_price == pytest.approx(
        report.expected_price, abs=report.tolerance_absolute
    )
    assert report.delta_absolute_error <= report.delta_tolerance_absolute
    assert report.gamma_absolute_error <= report.gamma_tolerance_absolute
    assert report.observed_delta == pytest.approx(
        report.expected_delta, abs=report.delta_tolerance_absolute
    )
    assert report.observed_gamma == pytest.approx(
        report.expected_gamma, abs=report.gamma_tolerance_absolute
    )
    assert report.convergence_rows
    assert [row.absolute_error for row in report.convergence_rows] == sorted(
        (row.absolute_error for row in report.convergence_rows),
        reverse=True,
    )
    assert report.diagnostics["mesh_family"] == "line_uniform"
    assert report.diagnostics["element_family"] == "lagrange_p2"
    assert report.diagnostics["boundary_nodes_enforced"] == 2
    assert (
        report.diagnostics["weak_form_sign_convention"]
        == "existing_forward_tau_identity_transform_black_scholes_forms"
    )


def test_public_black_scholes_parity_fixture_is_deterministic() -> None:
    first = run_public_black_scholes_parity_fixture()
    second = run_public_black_scholes_parity_fixture()

    assert first.to_public_dict() == second.to_public_dict()


def test_fem_bs_001_public_problem_spec_is_stable_and_consumable() -> None:
    report = run_public_black_scholes_parity_fixture()
    spec_payload = json.loads(FEM_BS_001_PROBLEM_SPEC_PATH.read_text())
    generated = build_public_fem_bs_oracle_problem_spec()
    regenerated_id = build_fixture_config_hash(generated)

    assert spec_payload["contract_version"] == "fem-parity-contract/v1"
    assert spec_payload["fixture_id"] == PUBLIC_SYNTHETIC_BLACK_SCHOLES_BENCHMARK_ID
    assert spec_payload["problem_id"] == report.problem_id
    assert spec_payload["privacy_class"] == "public_synthetic"
    assert spec_payload["weak_form"]["sign_convention"] == report.weak_form.sign_convention
    assert spec_payload["weak_form"]["equation_id"] == report.weak_form.equation_id
    assert spec_payload["weak_form"]["time_transformation"] == report.weak_form.time_transformation
    assert spec_payload["comparison_policy"]["mode"] == report.comparison_policy.mode
    assert spec_payload["comparison_policy"]["policy_id"] == report.comparison_policy.policy_id
    assert spec_payload["comparison_policy"]["metric_tolerances"] == dict(
        report.comparison_policy.metric_tolerances
    )
    assert (
        spec_payload["sensitivity_reference_policy"]["policy_id"]
        == report.sensitivity_reference_policy.policy_id
    )
    assert spec_payload["boundaries"] == [item.to_public_dict() for item in report.boundaries]
    assert spec_payload["contract_id"] == regenerated_id
    assert FEM_FIXTURE_DIR.joinpath("problem_spec.json").exists()


def test_fem_bs_001_result_export_is_public_mesh_time_and_result_payload() -> None:
    report = run_public_black_scholes_parity_fixture()
    payload = json.loads(FEM_BS_001_RESULT_EXPORT_PATH.read_text())
    spec_payload = json.loads(FEM_BS_001_PROBLEM_SPEC_PATH.read_text())

    assert payload["format_version"] == "fem-bs-oracle-result-v1"
    assert payload["benchmark_id"] == report.benchmark_id
    assert payload["problem_id"] == report.problem_id
    assert payload["config_hash"] == report.config_hash
    assert payload["weak_form"]["sign_convention"] == report.weak_form.sign_convention
    assert payload["comparison_policy"]["mode"] == "equal_error"
    assert payload["comparison_policy"]["metric_tolerances"] == {
        "price_absolute": report.tolerance_absolute,
        "price_relative": report.tolerance_relative,
        "delta_absolute": report.delta_tolerance_absolute,
        "gamma_absolute": report.gamma_tolerance_absolute,
    }
    assert payload["mesh_metadata"]["mesh_family"] == report.mesh_metadata.mesh_family
    assert (
        payload["mesh_metadata"]["solver_backing"]
        == spec_payload["mesh_metadata"]["solver_backing"]
    )
    assert payload["mesh_metadata"]["refinement_levels"] == list(
        report.mesh_metadata.refinement_levels
    )
    assert payload["time_metadata"]["time_steps"] == report.time_metadata.time_steps
    assert (
        payload["sensitivity_reference_policy"]["policy_id"]
        == report.sensitivity_reference_policy.policy_id
    )
    assert [row["refinement_level"] for row in payload["rows"]] == [
        row.refinement_level for row in report.convergence_rows
    ]
    assert payload["rows"][0]["absolute_error"] >= payload["rows"][1]["absolute_error"]
    assert payload["summary"]["observed_price"] == pytest.approx(report.observed_price)
    assert payload["summary"]["price_absolute_error"] == pytest.approx(report.price_absolute_error)
    assert payload["summary"]["price_tolerance_absolute"] == report.tolerance_absolute
    assert payload["summary"]["price_tolerance_relative"] == report.tolerance_relative
    assert payload["summary"]["delta_tolerance_absolute"] == report.delta_tolerance_absolute
    assert payload["summary"]["gamma_tolerance_absolute"] == report.gamma_tolerance_absolute


def test_config_hash_distinguishes_sparse_refinement_schedule() -> None:
    default_report = run_public_black_scholes_parity_fixture(
        refinement_levels=(4, 5, 6), time_steps=40
    )
    sparse_report = run_public_black_scholes_parity_fixture(refinement_levels=(4, 6), time_steps=40)

    assert default_report.mesh_metadata.refinement_levels == (4, 5, 6)
    assert sparse_report.mesh_metadata.refinement_levels == (4, 6)
    assert default_report.config_hash != sparse_report.config_hash


def test_refresh_exports_serializes_current_non_default_report(tmp_path, monkeypatch) -> None:
    result_path = tmp_path / "result_export.json"
    spec_path = tmp_path / "problem_spec.json"
    monkeypatch.setattr(parity_module, "FEM_BS_001_RESULT_EXPORT_PATH", result_path)
    monkeypatch.setattr(parity_module, "FEM_BS_001_PROBLEM_SPEC_PATH", spec_path)

    report = parity_module.run_public_black_scholes_parity_fixture(
        refinement_levels=(4, 5), time_steps=40, refresh_exports=True
    )
    payload = json.loads(result_path.read_text())
    spec_payload = json.loads(spec_path.read_text())

    assert spec_path.exists()
    assert payload["config_hash"] == report.config_hash
    assert payload["time_metadata"]["time_steps"] == 40
    assert payload["mesh_metadata"]["refinement_levels"] == [4, 5]
    assert spec_payload["mesh_metadata"]["mesh_refinement_levels"] == [4, 5]
    assert spec_payload["mesh_metadata"]["default_time_steps"] == 40
    assert spec_payload["contract_id"] == build_fixture_config_hash(
        {key: value for key, value in spec_payload.items() if key != "contract_id"}
    )
    assert [row["time_steps"] for row in payload["rows"]] == [40, 40]
    assert payload["summary"]["observed_price"] == pytest.approx(report.observed_price)
    assert payload["summary"]["price_absolute_error"] == pytest.approx(report.price_absolute_error)
