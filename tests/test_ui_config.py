"""Validation and capability gates for Streamlit/UI-bound configuration."""

from __future__ import annotations

import pytest

from finite_element_options.ui_config import (
    UiGridSpec,
    UiModelSpec,
    UiResourceLimits,
    UiSolverOptions,
    UiValidationError,
    ui_problem_from_shareable,
    validate_ui_problem,
)


def _valid_black_scholes_model(
    *,
    strike: float = 0.4,
    maturity: float = 1.0,
    rate: float = 0.03,
    carry: float = 0.01,
    volatility: float = 0.20,
) -> UiModelSpec:
    return UiModelSpec(
        model="black_scholes",
        strike=strike,
        maturity=maturity,
        rate=rate,
        carry=carry,
        volatility=volatility,
    )


def test_supported_black_scholes_ui_problem_builds_shareable_contract() -> None:
    limits = UiResourceLimits(max_dofs=10_000, max_time_steps=100, max_solves=100)
    validated = validate_ui_problem(
        model=_valid_black_scholes_model(),
        grid=UiGridSpec(mesh_refine=4, time_steps=64),
        solver=UiSolverOptions(theta=0.5, exercise_style="european"),
        limits=limits,
        strict=True,
    )

    assert validated.can_run
    assert validated.route_status == "supported"
    assert validated.domain_axes[0].name == "s"
    assert validated.domain_axes[0].upper > validated.model.strike
    assert validated.work_estimate.estimated_dofs > 0
    assert validated.work_estimate.estimated_matrix_bytes > 0
    assert not validated.error_diagnostics

    payload = validated.to_shareable_dict()
    assert payload["schema_version"] == "finite-element-options-ui-config-v1"
    assert payload["cache_key"] == validated.cache_key
    assert payload["model"]["model"] == "black_scholes"
    assert payload["solver"]["exercise_style"] == "european"
    assert "secret" not in str(payload).lower()

    round_tripped = ui_problem_from_shareable(payload)
    assert round_tripped.cache_key == validated.cache_key
    assert round_tripped.limits == limits
    assert round_tripped.to_shareable_dict()["model"] == payload["model"]

    status = validated.to_status_dict(solver_diagnostics={"steps": 63})
    assert status["backend"]["capability_maturity"] == "validated"
    assert status["benchmark_ids"]
    assert status["approximation_status"]["estimated_dofs"] > 0
    assert status["convergence_status"] == "benchmark_evidenced_not_reestimated_in_ui_run"
    assert status["solver_diagnostics"] == {"steps": 63}


def test_ui_validation_rejects_bad_cross_field_inputs_before_mesh_allocation() -> None:
    result = validate_ui_problem(
        model=_valid_black_scholes_model(maturity=1.0, volatility=0.2),
        grid=UiGridSpec(mesh_refine=4, time_steps=1),
        solver=UiSolverOptions(theta=1.4, exercise_style="european"),
    )

    assert not result.can_run
    assert {diag.field for diag in result.error_diagnostics} == {"time_steps", "theta"}
    with pytest.raises(UiValidationError) as excinfo:
        result.raise_for_errors()
    assert "time_steps" in str(excinfo.value)


def test_ui_rejects_boundary_facets_not_present_on_validated_domain() -> None:
    result = validate_ui_problem(
        model=_valid_black_scholes_model(),
        grid=UiGridSpec(mesh_refine=4, time_steps=32),
        solver=UiSolverOptions(theta=0.5, dirichlet_boundaries=("v_min",)),
    )

    assert not result.can_run
    assert result.domain_axes[0].name == "s"
    assert any(
        diag.code == "unsupported_boundary_for_domain"
        and diag.supported == ("s_max", "s_min")
        for diag in result.error_diagnostics
    )


def test_invalid_public_scalars_return_diagnostics_instead_of_crashing() -> None:
    result = validate_ui_problem(
        model=_valid_black_scholes_model(strike=-0.4),
        grid=UiGridSpec(mesh_refine=4, time_steps=32),
        solver=UiSolverOptions(theta=0.5),
    )

    assert not result.can_run
    assert {diag.field for diag in result.error_diagnostics} >= {"strike", "domain"}
    with pytest.raises(UiValidationError):
        result.raise_for_errors()


def test_tail_probability_changes_domain_width_monotonically() -> None:
    narrow_tail = validate_ui_problem(
        model=_valid_black_scholes_model(volatility=0.6, maturity=2.0),
        grid=UiGridSpec(mesh_refine=4, time_steps=32, alpha_tail=0.01),
        solver=UiSolverOptions(theta=0.5),
        strict=True,
    )
    wide_tail = validate_ui_problem(
        model=_valid_black_scholes_model(volatility=0.6, maturity=2.0),
        grid=UiGridSpec(mesh_refine=4, time_steps=32, alpha_tail=0.30),
        solver=UiSolverOptions(theta=0.5),
        strict=True,
    )

    assert narrow_tail.domain_axes[0].upper > wide_tail.domain_axes[0].upper
    assert narrow_tail.domain_axes[0].tail_mass == 0.01
    assert wide_tail.domain_axes[0].tail_mass == 0.30


def test_zero_maturity_and_zero_volatility_are_explicit_analytical_limits() -> None:
    analytical_models = (
        _valid_black_scholes_model(maturity=0.0),
        _valid_black_scholes_model(volatility=0.0),
        UiModelSpec(
            model="heston",
            strike=0.4,
            maturity=1.0,
            rate=0.03,
            carry=0.01,
            volatility=0.0,
            kappa=0.0,
            long_run_variance=0.0,
            vol_of_variance=0.0,
            correlation=0.5,
        ),
    )
    for model in analytical_models:
        result = validate_ui_problem(
            model=model,
            grid=UiGridSpec(mesh_refine=20, time_steps=1),
            solver=UiSolverOptions(theta=1.4, exercise_style="european"),
            strict=True,
        )
        assert result.route_status == "analytical_limit"
        assert result.can_run
        assert result.requires_numerical_solve is False
        assert any(diag.code == "analytical_limit" for diag in result.diagnostics)


def test_american_and_heston_routes_fail_closed_with_capability_diagnostics() -> None:
    american = validate_ui_problem(
        model=_valid_black_scholes_model(),
        grid=UiGridSpec(mesh_refine=4, time_steps=32),
        solver=UiSolverOptions(theta=0.5, exercise_style="american"),
    )
    assert not american.can_run
    assert any(diag.field == "exercise_style" for diag in american.error_diagnostics)

    heston = validate_ui_problem(
        model=UiModelSpec(
            model="heston",
            strike=0.4,
            maturity=1.0,
            rate=0.03,
            carry=0.01,
            volatility=0.2,
            kappa=0.5,
            long_run_variance=0.04,
            vol_of_variance=0.2,
            correlation=0.5,
        ),
        grid=UiGridSpec(mesh_refine=4, time_steps=32),
        solver=UiSolverOptions(theta=0.5, exercise_style="european"),
    )
    assert not heston.can_run
    assert {diag.field for diag in heston.error_diagnostics} >= {
        "dimension",
        "mesh_family",
    }


def test_heston_cross_field_validation_catches_singular_and_zero_cir_inputs() -> None:
    result = validate_ui_problem(
        model=UiModelSpec(
            model="heston",
            strike=0.4,
            maturity=1.0,
            rate=0.03,
            carry=0.01,
            volatility=0.2,
            kappa=0.0,
            long_run_variance=0.0,
            vol_of_variance=0.2,
            correlation=1.0,
        ),
        grid=UiGridSpec(mesh_refine=4, time_steps=32),
        solver=UiSolverOptions(theta=0.5, exercise_style="european"),
    )

    assert not result.can_run
    assert {diag.field for diag in result.error_diagnostics} >= {
        "kappa",
        "long_run_variance",
        "correlation",
    }


def test_work_estimate_blocks_oversized_meshes_before_allocation() -> None:
    result = validate_ui_problem(
        model=_valid_black_scholes_model(),
        grid=UiGridSpec(mesh_refine=20, time_steps=2000),
        solver=UiSolverOptions(theta=0.5, exercise_style="european"),
    )

    assert not result.can_run
    assert result.work_estimate.estimated_dofs > result.limits.max_dofs
    assert any(diag.code == "work_limit_exceeded" for diag in result.error_diagnostics)
