"""Tests for calibration utilities."""

from __future__ import annotations

import builtins
import hashlib
import importlib
import importlib.util
import sys
import warnings
from collections.abc import Mapping, Sequence as StringSequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from finite_element_options.estimation import (
    CalibrationObjective,
    CalibrationPricingError,
    CalibrationResult,
    HestonCalibrator,
    HestonMCMCDiagnosticThresholds,
    HestonPricingCalibrator,
    PricingCalibrationDataset,
    PricingModelCalibrator,
    PyMCCalibrator,
    StatsmodelsCalibrator,
    SyntheticSurfaceCalibrator,
    build_heston_bayesian_calibration_result,
    evaluate_heston_mcmc_diagnostics,
    validate_heston_posterior_draws,
)


def _surface() -> tuple[pd.DataFrame, np.ndarray]:
    s = np.linspace(80, 120, 5)
    t = np.linspace(0.1, 1.0, 5)
    strikes_grid, maturities_grid = np.meshgrid(s, t)
    strikes = strikes_grid.ravel()
    maturities = maturities_grid.ravel()
    true_params = np.array([0.04, 1.0, 0.04, 0.3, -0.7])
    prices = SyntheticSurfaceCalibrator.price_formula(strikes, maturities, true_params)
    data = pd.DataFrame({"strike": strikes, "maturity": maturities, "price": prices})
    return data, true_params


def test_synthetic_calibration_returns_full_diagnostics() -> None:
    data, true_params = _surface()
    calibrator = SyntheticSurfaceCalibrator(data)
    initial = true_params + np.array([0.01, -0.1, 0.02, -0.05, 0.1])

    result = calibrator.calibrate(initial)

    assert isinstance(result, CalibrationResult)
    assert result.success is True
    assert result.parameters.shape == true_params.shape
    assert np.allclose(result.parameters, true_params, atol=1e-2)
    assert result.residuals.shape == data["price"].shape
    assert np.linalg.norm(result.residuals) < 1e-8
    assert result.jacobian_rank == len(true_params)
    assert np.isfinite(result.jacobian_condition)
    assert result.bounds[0].shape == true_params.shape
    assert result.bounds[1].shape == true_params.shape
    assert result.nfev > 0
    assert result.status != 0
    assert result.message
    assert result.method == "scipy.least_squares"


def test_calibration_reports_rank_deficiency_without_claiming_full_identification() -> None:
    true_params = np.array([0.04, 1.0, 0.04, 0.3, -0.7])
    strikes = np.full(8, 100.0)
    maturities = np.full(8, 0.5)
    prices = SyntheticSurfaceCalibrator.price_formula(strikes, maturities, true_params)
    data = pd.DataFrame({"strike": strikes, "maturity": maturities, "price": prices})
    calibrator = SyntheticSurfaceCalibrator(data)

    result = calibrator.calibrate(true_params + np.array([0.1, -0.2, 0.1, -0.1, 0.2]))

    assert isinstance(result, CalibrationResult)
    assert result.success is True
    assert result.jacobian_rank < len(true_params)
    assert result.jacobian_condition == np.inf


def test_statsmodels_calibrator_is_deprecated_scipy_shim_without_global_monkeypatch() -> None:
    data, true_params = _surface()
    calibrator = StatsmodelsCalibrator(data)
    initial = true_params + np.array([0.01, -0.1, 0.02, -0.05, 0.1])

    regression = None
    had_regression_results = False
    prior_regression_results = None
    if importlib.util.find_spec("statsmodels") is not None:
        import statsmodels.regression as regression  # type: ignore[import-untyped,no-redef]

        had_regression_results = hasattr(regression, "RegressionResults")
        prior_regression_results = getattr(regression, "RegressionResults", None)

    with pytest.warns(DeprecationWarning, match="delegates to SciPy"):
        result = calibrator.calibrate(initial)

    assert isinstance(result, CalibrationResult)
    assert result.success is True
    assert np.allclose(result.parameters, true_params, atol=1e-2)
    if regression is not None:
        assert hasattr(regression, "RegressionResults") is had_regression_results
        assert getattr(regression, "RegressionResults", None) is prior_regression_results


def test_heston_named_calibrator_fails_closed_until_real_heston_engine_exists() -> None:
    data, true_params = _surface()
    calibrator = HestonCalibrator(data)

    with pytest.raises(NotImplementedError, match="real Heston pricing engine"):
        calibrator.calibrate(true_params)


def test_core_import_does_not_require_statsmodels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def blocked_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: StringSequence[str] = (),
        level: int = 0,
    ) -> object:
        if name.startswith("statsmodels"):
            raise ModuleNotFoundError(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    sys.modules.pop("finite_element_options", None)

    module = importlib.import_module("finite_element_options")

    assert module.__name__ == "finite_element_options"


def test_estimation_source_has_no_private_statsmodels_nonlinls_import() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "finite_element_options"
    offenders = [
        path
        for path in src_root.rglob("*.py")
        if "statsmodels.miscmodels.nonlinls" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_pymc_calibration_recovers_parameters() -> None:
    data, true_params = _surface()
    calibrator = PyMCCalibrator(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = calibrator.calibrate(draws=200, chains=2, random_seed=123)
    assert isinstance(result, CalibrationResult)
    assert result.success is True
    assert np.allclose(result.parameters, true_params, atol=0.15)


def _valid_heston_draws() -> dict[str, np.ndarray]:
    return {
        "v0": np.array([0.040, 0.045, 0.050]),
        "kappa": np.array([2.20, 2.35, 2.50]),
        "theta": np.array([0.040, 0.042, 0.044]),
        "sigma": np.array([0.25, 0.26, 0.27]),
        "rho": np.array([-0.45, -0.40, -0.35]),
    }


def test_heston_posterior_draws_enforce_constraints_and_report_feller_policy() -> None:
    report = validate_heston_posterior_draws(
        _valid_heston_draws(),
        feller_policy="report",
    )

    assert report.feller_policy == "report"
    assert report.parameter_names == ("v0", "kappa", "theta", "sigma", "rho")
    assert report.draw_count == 3
    assert report.feller_ratio_min > 1.0
    assert report.feller_condition_satisfied is True

    invalid_rho = _valid_heston_draws()
    invalid_rho["rho"] = np.array([-0.1, 1.0, 0.2])
    with pytest.raises(ValueError, match="rho"):
        validate_heston_posterior_draws(invalid_rho)

    feller_violation = _valid_heston_draws()
    feller_violation["sigma"] = np.array([1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="Feller"):
        validate_heston_posterior_draws(
            feller_violation,
            feller_policy="enforce",
        )


def test_heston_mcmc_diagnostic_gate_rejects_weak_or_divergent_runs() -> None:
    thresholds = HestonMCMCDiagnosticThresholds(
        max_r_hat=1.01,
        min_bulk_ess=400.0,
        min_tail_ess=400.0,
        max_divergences=0,
        max_tree_depth_hits=0,
    )
    weak_summary = pd.DataFrame(
        {
            "r_hat": [1.02, 1.0, 1.0, 1.0, 1.0],
            "ess_bulk": [390.0, 500.0, 500.0, 500.0, 500.0],
            "ess_tail": [500.0, 500.0, 350.0, 500.0, 500.0],
        },
        index=pd.Index(["v0", "kappa", "theta", "sigma", "rho"]),
    )

    report = evaluate_heston_mcmc_diagnostics(
        weak_summary,
        divergences=1,
        tree_depth_hits=1,
        thresholds=thresholds,
    )

    assert report.accepted is False
    assert any("r_hat" in failure for failure in report.failures)
    assert any("bulk ESS" in failure for failure in report.failures)
    assert any("tail ESS" in failure for failure in report.failures)
    assert any("divergence" in failure for failure in report.failures)
    assert any("tree depth" in failure for failure in report.failures)

    with pytest.raises(ValueError, match="heldout_rmse"):
        evaluate_heston_mcmc_diagnostics(
            weak_summary,
            heldout_rmse=float("nan"),
            thresholds=thresholds,
        )
    with pytest.raises(ValueError, match="heldout_rmse"):
        evaluate_heston_mcmc_diagnostics(
            weak_summary,
            heldout_rmse=-0.01,
            thresholds=thresholds,
        )


def test_heston_bayesian_result_requires_validated_engine_and_retains_provenance(
    tmp_path: Path,
) -> None:
    draws = _valid_heston_draws()
    diagnostic_summary = pd.DataFrame(
        {
            "r_hat": [1.0, 1.0, 1.0, 1.0, 1.0],
            "ess_bulk": [800.0, 820.0, 810.0, 805.0, 815.0],
            "ess_tail": [700.0, 710.0, 705.0, 715.0, 720.0],
        },
        index=pd.Index(["v0", "kappa", "theta", "sigma", "rho"]),
    )
    observed = np.array([[1.0, 1.5], [2.0, 2.5]])
    fitted = np.array([[1.01, 1.49], [2.02, 2.48]])
    validation_artifact = tmp_path / "fourier-heston-validation.json"
    validation_artifact.write_text(
        '{"engine_family":"heston","validation_grid":"smoke"}\n',
        encoding="utf-8",
    )
    validation_artifact_sha256 = hashlib.sha256(validation_artifact.read_bytes()).hexdigest()

    result = build_heston_bayesian_calibration_result(
        posterior_draws=draws,
        diagnostic_summary=diagnostic_summary,
        observed_values=observed,
        fitted_values=fitted,
        inference_data_artifact="artifacts/heston-idata.nc",
        pricing_engine="validated-fourier-heston",
        pricing_engine_validation={
            "engine_family": "heston",
            "validated": True,
            "validation_artifact": str(validation_artifact),
            "validation_artifact_sha256": validation_artifact_sha256,
            "version": "2026.7",
        },
        likelihood_units="price",
        observation_noise=0.02,
        random_seed=123,
        thresholds=HestonMCMCDiagnosticThresholds(max_heldout_rmse=0.05),
        heldout_rmse=0.02,
    )

    assert isinstance(result, CalibrationResult)
    assert result.success is True
    assert result.method == "pymc.heston.diagnostics"
    assert result.parameter_names == ("v0", "kappa", "theta", "sigma", "rho")
    assert np.all(result.parameters[:4] > 0.0)
    assert -1.0 < result.parameters[4] < 1.0
    assert result.bounds[0].tolist() == [0.0, 0.0, 0.0, 0.0, -1.0]
    assert result.bounds[1][-1] == 1.0
    assert result.active_mask.tolist() == [0, 0, 0, 0, 0]
    assert result.cost == pytest.approx(0.5 * float(np.sum((fitted - observed).ravel() ** 2)))
    assert result.artifacts == ("artifacts/heston-idata.nc",)
    assert result.provenance["pricing_engine"] == "validated-fourier-heston"
    validation_metadata = result.provenance["pricing_engine_validation"]
    assert isinstance(validation_metadata, Mapping)
    assert validation_metadata["validation_artifact"] == str(validation_artifact)
    assert validation_metadata["validation_artifact_sha256"] == validation_artifact_sha256
    assert result.provenance["likelihood_units"] == "price"
    mcmc_diagnostics = result.diagnostics["mcmc"]
    constraint_diagnostics = result.diagnostics["constraints"]
    likelihood_diagnostics = result.diagnostics["likelihood"]
    assert isinstance(mcmc_diagnostics, Mapping)
    assert isinstance(constraint_diagnostics, Mapping)
    assert isinstance(likelihood_diagnostics, Mapping)
    assert mcmc_diagnostics["accepted"] is True
    assert constraint_diagnostics["feller_condition_satisfied"] is True
    assert likelihood_diagnostics["fit_rmse"] == pytest.approx(
        float(np.sqrt(np.mean((fitted - observed).ravel() ** 2)))
    )
    assert likelihood_diagnostics["heldout_rmse"] == 0.02

    with pytest.raises(ValueError, match="validated Heston pricing engine"):
        build_heston_bayesian_calibration_result(
            posterior_draws=draws,
            diagnostic_summary=diagnostic_summary,
            observed_values=observed,
            fitted_values=fitted,
            inference_data_artifact="artifacts/heston-idata.nc",
            pricing_engine="synthetic toy polynomial",
            pricing_engine_validation={
                "engine_family": "heston",
                "validated": True,
                "validation_artifact": "artifacts/fourier-heston-validation.json",
                "validation_artifact_sha256": "a" * 64,
                "version": "2026.7",
            },
            likelihood_units="price",
            observation_noise=0.02,
            random_seed=123,
        )

    with pytest.raises(ValueError, match="validated Heston pricing engine"):
        build_heston_bayesian_calibration_result(
            posterior_draws=draws,
            diagnostic_summary=diagnostic_summary,
            observed_values=observed,
            fitted_values=fitted,
            inference_data_artifact="artifacts/heston-idata.nc",
            pricing_engine="validated-fourier-engine",
            pricing_engine_validation={
                "engine_family": "heston",
                "validated": True,
                "validation_artifact": "artifacts/fourier-heston-validation.json",
                "validation_artifact_sha256": "a" * 64,
                "version": "2026.7",
            },
            likelihood_units="price",
            observation_noise=0.02,
            random_seed=123,
        )

    with pytest.raises(ValueError, match="validation_artifact_sha256"):
        build_heston_bayesian_calibration_result(
            posterior_draws=draws,
            diagnostic_summary=diagnostic_summary,
            observed_values=observed,
            fitted_values=fitted,
            inference_data_artifact="artifacts/heston-idata.nc",
            pricing_engine="validated-fourier-heston",
            pricing_engine_validation={
                "engine_family": "heston",
                "validated": True,
                "validation_artifact": "artifacts/fourier-heston-validation.json",
                "version": "2026.7",
            },
            likelihood_units="price",
            observation_noise=0.02,
            random_seed=123,
        )

    with pytest.raises(ValueError, match="validation_artifact_sha256"):
        build_heston_bayesian_calibration_result(
            posterior_draws=draws,
            diagnostic_summary=diagnostic_summary,
            observed_values=observed,
            fitted_values=fitted,
            inference_data_artifact="artifacts/heston-idata.nc",
            pricing_engine="validated-fourier-heston",
            pricing_engine_validation={
                "engine_family": "heston",
                "validated": True,
                "validation_artifact": str(validation_artifact),
                "validation_artifact_sha256": "b" * 64,
                "version": "2026.7",
            },
            likelihood_units="price",
            observation_noise=0.02,
            random_seed=123,
        )

    with pytest.raises(ValueError, match="validation_artifact"):
        build_heston_bayesian_calibration_result(
            posterior_draws=draws,
            diagnostic_summary=diagnostic_summary,
            observed_values=observed,
            fitted_values=fitted,
            inference_data_artifact="artifacts/heston-idata.nc",
            pricing_engine="validated-fourier-heston",
            pricing_engine_validation={
                "engine_family": "heston",
                "validated": True,
                "validation_artifact": str(tmp_path / "missing-validation.json"),
                "validation_artifact_sha256": validation_artifact_sha256,
                "version": "2026.7",
            },
            likelihood_units="price",
            observation_noise=0.02,
            random_seed=123,
        )


def _pricing_frame() -> pd.DataFrame:
    spot = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    strike = np.array([80.0, 90.0, 100.0, 110.0, 120.0, 105.0])
    maturity = np.array([0.25, 0.50, 0.75, 1.00, 1.25, 1.50])
    rate = np.full_like(strike, 0.03)
    carry = np.full_like(strike, 0.01)
    true_params = np.array([0.20, -0.06, 0.03])
    quote = _linear_smile_prices(true_params, spot, strike, maturity, rate, carry)
    bid_ask = np.array([0.04, 0.05, 0.05, 0.06, 0.07, 0.08])
    return pd.DataFrame(
        {
            "spot": spot,
            "strike": strike,
            "maturity": maturity,
            "rate": rate,
            "carry": carry,
            "quote": quote,
            "bid": quote - bid_ask / 2.0,
            "ask": quote + bid_ask / 2.0,
            "vega": np.linspace(8.0, 14.0, quote.size),
            "split": ["train", "train", "train", "train", "train", "holdout"],
        }
    )


def _linear_smile_prices(
    params: np.ndarray,
    spot: np.ndarray,
    strike: np.ndarray,
    maturity: np.ndarray,
    rate: np.ndarray,
    carry: np.ndarray,
) -> np.ndarray:
    level, skew, term = params
    log_moneyness = np.log(strike / spot)
    discount = np.exp(-rate * maturity)
    forward_adjustment = np.exp(-carry * maturity)
    return discount * (level + skew * log_moneyness + term * maturity) * forward_adjustment


def _linear_smile_engine(
    params: np.ndarray,
    dataset: PricingCalibrationDataset,
) -> np.ndarray:
    return _linear_smile_prices(
        params,
        dataset.spot,
        dataset.strike,
        dataset.maturity,
        dataset.rate,
        dataset.carry,
    )


def test_pricing_model_calibrator_enforces_bounds_weights_and_holdout_diagnostics() -> None:
    frame = _pricing_frame()
    dataset = PricingCalibrationDataset.from_frame(frame, quote_units="price")
    calibrator = PricingModelCalibrator(
        dataset=dataset,
        pricing_function=_linear_smile_engine,
        parameter_names=("level", "skew", "term"),
        bounds=([0.01, -0.30, -0.10], [0.50, 0.20, 0.20]),
    )
    objective = CalibrationObjective(
        residual_units="price",
        weight_policy="bid_ask",
        robust_loss="soft_l1",
    )

    result = calibrator.calibrate(
        initial_guess=np.array([0.11, 0.08, -0.02]),
        objective=objective,
        holdout_mask=frame["split"].eq("holdout"),
        candidate_initial_guesses=(np.array([0.35, -0.20, 0.12]),),
    )

    assert isinstance(result, CalibrationResult)
    assert result.success is True
    assert result.parameter_names == ("level", "skew", "term")
    np.testing.assert_allclose(result.parameters, [0.20, -0.06, 0.03], atol=2.0e-6)
    assert np.all(result.parameters >= result.bounds[0])
    assert np.all(result.parameters <= result.bounds[1])
    assert np.linalg.norm(result.residuals) < 1.0e-8
    diagnostics = result.diagnostics
    assert diagnostics["objective"]["residual_units"] == "price"
    assert diagnostics["objective"]["weight_policy"] == "bid_ask"
    assert diagnostics["objective"]["robust_loss"] == "soft_l1"
    assert diagnostics["observations"]["training_count"] == 5
    assert diagnostics["observations"]["holdout_count"] == 1
    assert diagnostics["fit"]["holdout_rmse"] < 1.0e-8
    assert diagnostics["pricing_engine"]["pricing_failures"] == 0
    assert diagnostics["pricing_engine"]["pricing_evaluations"] >= 1
    assert diagnostics["weights"]["min"] > 0.0
    assert diagnostics["optimizer"]["start_count"] == 2
    assert len(diagnostics["optimizer"]["multi_start"]) == 2


def test_pricing_model_calibrator_fails_closed_for_bad_market_data_and_prices() -> None:
    frame = _pricing_frame()
    broken_quotes = frame.copy()
    broken_quotes.loc[0, "quote"] = np.nan
    with pytest.raises(ValueError, match="quote"):
        PricingCalibrationDataset.from_frame(broken_quotes)

    inverted_market = frame.copy()
    inverted_market.loc[0, "bid"] = inverted_market.loc[0, "ask"] + 0.01
    dataset = PricingCalibrationDataset.from_frame(inverted_market)
    calibrator = PricingModelCalibrator(
        dataset=dataset,
        pricing_function=_linear_smile_engine,
        parameter_names=("level", "skew", "term"),
    )
    with pytest.raises(ValueError, match="bid.*ask"):
        calibrator.calibrate(
            initial_guess=np.array([0.20, -0.06, 0.03]),
            objective=CalibrationObjective(weight_policy="bid_ask"),
        )

    good_dataset = PricingCalibrationDataset.from_frame(frame)

    def nonfinite_engine(
        params: np.ndarray,
        dataset: PricingCalibrationDataset,
    ) -> np.ndarray:
        del params
        prices = np.ones(dataset.quote.shape)
        prices[0] = np.nan
        return prices

    bad_calibrator = PricingModelCalibrator(
        dataset=good_dataset,
        pricing_function=nonfinite_engine,
        parameter_names=("level", "skew", "term"),
    )
    with pytest.raises(CalibrationPricingError, match="nonfinite"):
        bad_calibrator.calibrate(initial_guess=np.array([0.20, -0.06, 0.03]))


def test_pricing_model_calibrator_supports_vega_scaled_implied_volatility_objective() -> None:
    frame = _pricing_frame()
    implied_vol = frame.copy()
    true_params = np.array([0.20, -0.06, 0.03])
    implied_vol["quote"] = _linear_smile_prices(
        true_params,
        implied_vol["spot"].to_numpy(),
        implied_vol["strike"].to_numpy(),
        implied_vol["maturity"].to_numpy(),
        implied_vol["rate"].to_numpy(),
        implied_vol["carry"].to_numpy(),
    )
    dataset = PricingCalibrationDataset.from_frame(implied_vol, quote_units="implied_volatility")
    calibrator = PricingModelCalibrator(
        dataset=dataset,
        pricing_function=_linear_smile_engine,
        parameter_names=("level", "skew", "term"),
    )

    result = calibrator.calibrate(
        initial_guess=np.array([0.18, -0.03, 0.01]),
        objective=CalibrationObjective(
            residual_units="implied_volatility",
            weight_policy="vega",
        ),
    )

    assert result.success is True
    np.testing.assert_allclose(result.parameters, true_params, atol=2.0e-6)
    assert result.diagnostics["objective"]["residual_units"] == "implied_volatility"
    assert result.diagnostics["objective"]["weight_policy"] == "vega"
    assert result.diagnostics["weights"]["source"] == "vega"


def _heston_smoke_prices(params: np.ndarray, dataset: PricingCalibrationDataset) -> np.ndarray:
    v0, kappa, theta, sigma, rho = params
    if min(v0, kappa, theta, sigma) <= 0.0 or abs(rho) >= 1.0:
        raise ValueError("invalid Heston parameters")
    average_variance = theta + (v0 - theta) * (1.0 - np.exp(-kappa * dataset.maturity)) / (
        kappa * dataset.maturity
    )
    log_moneyness = np.log(dataset.strike / dataset.spot)
    skew_adjustment = 1.0 + 0.15 * rho * log_moneyness + 0.03 * sigma * dataset.maturity
    return np.sqrt(average_variance) * skew_adjustment


def test_heston_pricing_calibrator_recovers_parameters_with_validated_injected_engine(
    tmp_path: Path,
) -> None:
    true_params = np.array([0.040, 2.2, 0.050, 0.30, -0.35])
    frame = _pricing_frame().iloc[:5].copy()
    frame["quote"] = _heston_smoke_prices(
        true_params,
        PricingCalibrationDataset.from_frame(frame),
    )
    validation_artifact = tmp_path / "heston-moment-oracle-validation.json"
    validation_artifact.write_text(
        '{"engine_family":"heston","oracle":"moment-smoke"}\n',
        encoding="utf-8",
    )
    validation_sha = hashlib.sha256(validation_artifact.read_bytes()).hexdigest()

    calibrator = HestonPricingCalibrator(
        dataset=PricingCalibrationDataset.from_frame(frame),
        pricing_function=_heston_smoke_prices,
        pricing_engine="validated-moment-heston-smoke",
        pricing_engine_validation={
            "engine_family": "heston",
            "validated": True,
            "validation_artifact": str(validation_artifact),
            "validation_artifact_sha256": validation_sha,
            "version": "2026.7",
        },
        feller_policy="report",
    )

    result = calibrator.calibrate(
        initial_guess=np.array([0.035, 1.8, 0.045, 0.25, -0.25]),
        objective=CalibrationObjective(weight_policy="none"),
    )

    assert result.success is True
    assert result.parameter_names == ("v0", "kappa", "theta", "sigma", "rho")
    assert np.all(result.parameters[:4] > 0.0)
    assert -1.0 < result.parameters[4] < 1.0
    assert np.all(
        np.abs(result.parameters - true_params) <= np.array([5e-3, 0.8, 5e-3, 0.12, 0.25])
    )
    assert result.provenance["pricing_engine"] == "validated-moment-heston-smoke"
    assert result.diagnostics["constraints"]["feller_policy"] == "report"
    assert result.diagnostics["fit"]["training_rmse"] < 1.0e-6

    with pytest.raises(ValueError, match="validated Heston pricing engine"):
        HestonPricingCalibrator(
            dataset=PricingCalibrationDataset.from_frame(frame),
            pricing_function=_heston_smoke_prices,
            pricing_engine="synthetic toy polynomial",
            pricing_engine_validation={
                "engine_family": "heston",
                "validated": True,
                "validation_artifact": str(validation_artifact),
                "validation_artifact_sha256": validation_sha,
                "version": "2026.7",
            },
        )
