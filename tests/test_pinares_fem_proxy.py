"""Pinares fixed-price proxy weak-form FEM compatibility tests."""

from __future__ import annotations

from typing import Any
import json

import pytest

from finite_element_options.contracts import (
    DEFAULT_FEM_CAPABILITY_MANIFEST,
    FEMRouteRequest,
    UnsupportedReason,
    diagnose_unsupported_route,
    ensure_route_supported,
    validate_fpf_solver_result_evidence_payload,
)
from finite_element_options.validation.pinares_fixed_price_proxy import (
    PINARES_FEM_FAIL_CLOSED_BENCHMARK_ID,
    PINARES_FEM_FIXED_PRICE_PROXY_BENCHMARK_ID,
    PINARES_FEM_FIXED_PRICE_PROXY_PROBLEM_HASH,
    PINARES_FEM_FIXED_PRICE_PROXY_PROBLEM_ID,
    PINARES_FEM_PROVIDER_EVIDENCE_MANIFEST_PATH,
    PINARES_FEM_PROXY_PROBLEM_SPEC_PATH,
    PINARES_FEM_PROXY_RESULT_EXPORT_PATH,
    PINARES_FEM_PROXY_UNSUPPORTED_SPEC_PATH,
    PINARES_QPS_FIXTURE_PATH,
    PinaresFEMProxyReport,
    PinaresFixedPriceProxyCase,
    build_pinares_fem_provider_evidence_manifest,
    build_pinares_fem_proxy_hash,
    public_pinares_fixed_price_problem_spec,
    public_pinares_full_deal_unsupported_problem_spec,
    run_public_pinares_fixed_price_proxy_fixture,
)


@pytest.fixture(scope="module")
def pinares_report() -> PinaresFEMProxyReport:
    """Run the deterministic public-synthetic Pinares FEM fixture once."""

    return run_public_pinares_fixed_price_proxy_fixture()


def _load_json(path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_pinares_fixed_price_proxy_runs_against_analytical_survival_scaled_oracle(
    pinares_report: PinaresFEMProxyReport,
) -> None:
    report = pinares_report

    assert report.converged
    assert report.case.problem_id == PINARES_FEM_FIXED_PRICE_PROXY_PROBLEM_ID
    assert report.case.problem_hash == PINARES_FEM_FIXED_PRICE_PROXY_PROBLEM_HASH
    assert report.price_absolute_error_uf <= report.case.price_abs_tolerance_uf
    assert report.delta_absolute_error <= report.case.delta_abs_tolerance
    assert report.gamma_absolute_error <= report.case.gamma_abs_tolerance
    assert report.observed_price_uf == pytest.approx(
        report.expected_price_uf, abs=report.case.price_abs_tolerance_uf
    )
    assert report.observed_delta == pytest.approx(
        report.expected_delta, abs=report.case.delta_abs_tolerance
    )
    assert report.observed_gamma == pytest.approx(
        report.expected_gamma, abs=report.case.gamma_abs_tolerance
    )
    assert report.rows
    assert [row.absolute_error_uf for row in report.rows] == sorted(
        (row.absolute_error_uf for row in report.rows), reverse=True
    )
    assert all(row.degrees_of_freedom > 0 for row in report.rows)
    assert report.no_arbitrage["value_bound_ok"]
    assert report.no_arbitrage["upper_bound_ok"]
    assert report.no_arbitrage["delta_upper_bound_ok"]
    assert report.no_arbitrage["gamma_non_negative_ok"]


def test_pinares_problem_spec_maps_to_supported_fem_route_and_preserves_conventions() -> None:
    payload = public_pinares_fixed_price_problem_spec()
    request = FEMRouteRequest.from_quant_problem_spec(payload)

    assert payload["privacy_class"] == "public_synthetic"
    assert payload["problem_id"] == PINARES_FEM_FIXED_PRICE_PROXY_PROBLEM_ID
    assert payload["problem_hash"] == PINARES_FEM_FIXED_PRICE_PROXY_PROBLEM_HASH
    assert payload["mathematical_problem"]["weak_form"]["equation_id"] == (
        "pinares_fixed_price_proxy_black_scholes_weak_form"
    )
    assert request.source_schema_version == "quant-problem-spec/v0"
    assert request.backend_id == DEFAULT_FEM_CAPABILITY_MANIFEST.backend_id
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
    assert request.maturity_date == "2027-06-30"
    assert request.time_domain == "[0, 1]"
    ensure_route_supported(request)
    assert diagnose_unsupported_route(request) == ()


def test_static_pinares_exports_are_current_public_and_hash_stable(
    pinares_report: PinaresFEMProxyReport,
) -> None:
    problem_spec = _load_json(PINARES_FEM_PROXY_PROBLEM_SPEC_PATH)
    qps_spec = _load_json(PINARES_QPS_FIXTURE_PATH)
    result_export = _load_json(PINARES_FEM_PROXY_RESULT_EXPORT_PATH)
    unsupported_spec = _load_json(PINARES_FEM_PROXY_UNSUPPORTED_SPEC_PATH)
    provider_manifest = _load_json(PINARES_FEM_PROVIDER_EVIDENCE_MANIFEST_PATH)

    generated_spec = public_pinares_fixed_price_problem_spec(case=pinares_report.case)
    generated_unsupported = public_pinares_full_deal_unsupported_problem_spec()

    assert problem_spec == generated_spec
    assert qps_spec == generated_spec
    assert unsupported_spec == generated_unsupported
    assert result_export == pinares_report.export_payload()
    assert provider_manifest == build_pinares_fem_provider_evidence_manifest(pinares_report)
    assert problem_spec["contract_id"] == build_pinares_fem_proxy_hash(
        {key: value for key, value in problem_spec.items() if key != "contract_id"}
    )
    assert unsupported_spec["contract_id"] == build_pinares_fem_proxy_hash(
        {key: value for key, value in unsupported_spec.items() if key != "contract_id"}
    )
    assert result_export["benchmark_id"] == PINARES_FEM_FIXED_PRICE_PROXY_BENCHMARK_ID
    assert result_export["format_version"] == "fem-pinares-fixed-price-proxy-result-v1"
    assert validate_fpf_solver_result_evidence_payload(result_export) == ()
    assert result_export["problem_id"] == PINARES_FEM_FIXED_PRICE_PROXY_PROBLEM_ID
    assert result_export["problem_hash"] == PINARES_FEM_FIXED_PRICE_PROXY_PROBLEM_HASH
    assert result_export["status"] == "converged"
    assert result_export["measure"] == "Q*"
    assert result_export["numeraire"] == "UF_money_market_account_proxy"
    assert result_export["units"] == pinares_report.case.normalized_units()
    assert result_export["backend_capability_status"]["backend_id"] == DEFAULT_FEM_CAPABILITY_MANIFEST.backend_id
    assert result_export["diagnostics"]["route_id"] == pinares_report.case.route_id
    assert result_export["privacy_class"] == "public_synthetic"
    assert (
        result_export["summary"]["price_absolute_error_uf"]
        <= result_export["summary"]["price_tolerance_absolute_uf"]
    )


def test_pinares_fem_provider_evidence_manifest_reports_cache_and_performance_fields(
    pinares_report: PinaresFEMProxyReport,
) -> None:
    manifest = build_pinares_fem_provider_evidence_manifest(pinares_report)

    assert manifest["schema"] == "pinares.provider_evidence_manifest.v1"
    assert manifest["producer"] == "finite_element_options"
    assert manifest["privacy_class"] == "public_synthetic"
    assert manifest["issue_refs"] == ["googa27/finite_element_options#104"]
    assert manifest["evidence_class"] == "deterministic_proxy_not_full_family_contract_valuation"
    assert (
        manifest["capability_manifest"]["backend_id"] == DEFAULT_FEM_CAPABILITY_MANIFEST.backend_id
    )
    assert (
        manifest["capability_manifest"]["contract_version"]
        == DEFAULT_FEM_CAPABILITY_MANIFEST.contract_version
    )
    assert manifest["route"]["method_kind"] == "finite_element"
    assert (
        manifest["route"]["weak_form"]["equation_id"]
        == "pinares_fixed_price_proxy_black_scholes_weak_form"
    )
    assert manifest["cache_factorization_policy"]["linear_solver"] == "scipy_direct"
    assert manifest["cache_factorization_policy"]["factorization_reuse"] is True
    assert manifest["performance_sidecar"]["runtime"]["seconds"] is None
    assert (
        manifest["performance_sidecar"]["degrees_of_freedom"]
        == pinares_report.rows[-1].degrees_of_freedom
    )
    assert manifest["parity_metrics"]["price_abs_uf"] <= manifest["error_budgets"]["price_abs_uf"]
    assert manifest["unsupported_routes"]["full_family_contract"] == "fail_closed"


def test_unsupported_full_deal_route_fails_closed_with_actionable_diagnostics() -> None:
    payload = public_pinares_full_deal_unsupported_problem_spec()
    request = FEMRouteRequest.from_quant_problem_spec(payload)
    diagnostics = diagnose_unsupported_route(request)
    reasons = {diagnostic.reason for diagnostic in diagnostics}
    values = {diagnostic.value for diagnostic in diagnostics}

    assert payload["benchmark_ids"] == [PINARES_FEM_FAIL_CLOSED_BENCHMARK_ID]
    assert UnsupportedReason.UNSUPPORTED_DIMENSION in reasons
    assert UnsupportedReason.UNSUPPORTED_TERM in reasons
    assert UnsupportedReason.UNSUPPORTED_BOUNDARY in reasons
    assert UnsupportedReason.UNSUPPORTED_EXERCISE in reasons
    assert UnsupportedReason.UNSUPPORTED_OUTPUT in reasons
    assert {"hazard_killing", "liquidity_jump", "hjb_control", "obstacle"} <= values
    assert "legal_tax_conclusion" in values
    assert "free_boundary" in values
    assert request.dimension == 4
    with pytest.raises(ValueError, match="FEM backend supports dimensions"):
        ensure_route_supported(request)


def test_zero_survival_probability_edge_case_stays_finite() -> None:
    report = run_public_pinares_fixed_price_proxy_fixture(
        case=PinaresFixedPriceProxyCase(
            survival_probability=0.0, refinement_levels=(5,), time_steps=80
        )
    )

    assert report.converged
    assert report.expected_price_uf == pytest.approx(0.0)
    assert report.observed_price_uf == pytest.approx(0.0)
    assert report.expected_delta == pytest.approx(0.0)
    assert report.observed_delta == pytest.approx(0.0)
    assert report.expected_gamma == pytest.approx(0.0)
    assert report.observed_gamma == pytest.approx(0.0)
    assert report.no_arbitrage["survival_scale_ok"]
