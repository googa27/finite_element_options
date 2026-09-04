"""Research calibration tests for the regime-switching quanto example."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm

from finite_element_options.examples.regime_switching_quanto import (
    DataQualityConfig,
    discrete_to_continuous_generator,
    fit_markov_switching_joint_diffusion,
    fit_regime_candidates,
    prepare_joint_log_returns,
)


def test_prepare_joint_log_returns_quarantines_bad_levels_and_bridges() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=6, freq="D"),
            "sp500": [1000.0, 1010.0, np.nan, 1030.0, 1040.0, 90.0],
            "usdclp": [800.0, 804.0, 806.0, -1.0, 812.0, 816.0],
        }
    )
    config = DataQualityConfig(min_return_rows=2)

    returns, report = prepare_joint_log_returns(frame, config)

    assert list(returns.columns) == ["date", "sp500", "usdclp"]
    assert returns["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-02",
        "2024-01-05",
    ]
    np.testing.assert_allclose(
        returns[["sp500", "usdclp"]].to_numpy(),
        np.array(
            [
                [np.log(1010.0 / 1000.0), np.log(804.0 / 800.0)],
                [np.log(1040.0 / 1010.0), np.log(812.0 / 804.0)],
            ]
        ),
    )
    assert report["input_rows"] == 6
    assert report["valid_level_rows"] == 3
    assert report["return_rows"] == 2
    reasons = {reason for row in report["quarantined_rows"] for reason in row["reasons"]}
    assert reasons == {"sp500_nonfinite", "usdclp_nonpositive", "sp500_out_of_bounds"}
    assert report["bridged_return_gaps"] == [
        {
            "previous_valid_date": "2024-01-02",
            "return_date": "2024-01-05",
            "calendar_gap_days": 3,
            "quarantined_dates": ["2024-01-03", "2024-01-04"],
        }
    ]
    json.dumps(report)


def test_discrete_to_continuous_generator_recovers_known_three_state_generator() -> None:
    q_true = np.array(
        [
            [-4.0, 1.5, 2.5],
            [0.75, -1.0, 0.25],
            [1.25, 0.5, -1.75],
        ]
    )
    p_daily = expm(q_true / 252.0)

    q_est, residual = discrete_to_continuous_generator(p_daily, periods_per_year=252)

    assert residual < 1.0e-9
    np.testing.assert_allclose(q_est, q_true, atol=2.0e-5, rtol=2.0e-5)
    np.testing.assert_allclose(q_est.sum(axis=1), 0.0, atol=1.0e-10)
    assert np.all(q_est[~np.eye(3, dtype=bool)] >= -1.0e-12)


def test_fit_markov_switching_joint_diffusion_orders_vol_and_orients_transition() -> None:
    rng = np.random.default_rng(1234)
    n_obs = 260
    p_true = np.array([[0.96, 0.04], [0.10, 0.90]])
    regimes = np.zeros(n_obs, dtype=int)
    for idx in range(1, n_obs):
        regimes[idx] = rng.choice(2, p=p_true[regimes[idx - 1]])
    covariances = [
        np.array([[0.004**2, 0.25 * 0.004 * 0.003], [0.25 * 0.004 * 0.003, 0.003**2]]),
        np.array([[0.018**2, -0.35 * 0.018 * 0.014], [-0.35 * 0.018 * 0.014, 0.014**2]]),
    ]
    means = [np.array([0.00015, -0.00005]), np.array([-0.00025, 0.00015])]
    rows = [rng.multivariate_normal(means[state], covariances[state]) for state in regimes]
    returns = pd.DataFrame(rows, columns=["sp500", "usdclp"])
    returns.insert(0, "date", pd.date_range("2025-01-01", periods=n_obs, freq="B"))

    result = fit_markov_switching_joint_diffusion(
        returns,
        k_regimes=2,
        autoregressive_order=1,
        seed=7,
        search_reps=4,
        search_iter=5,
        maxiter=120,
        fit_attempts=2,
    )
    payload = result.to_dict()

    vols = [regime["composite_vol"] for regime in payload["regimes"]]
    assert vols[0] < vols[1]
    p_est = np.array(payload["transition_matrix"])
    assert p_est.shape == (2, 2)
    np.testing.assert_allclose(p_est.sum(axis=1), 1.0, atol=1.0e-10)
    assert p_est[0, 0] > p_est[0, 1]
    assert p_est[1, 1] > p_est[1, 0]
    current = np.array(payload["current_probabilities"])
    np.testing.assert_allclose(current.sum(), 1.0, atol=1.0e-10)
    assert current.shape == (2,)
    assert payload["generator_residual"] < 1.0e-5
    assert payload["autoregressive_order"] == 1
    assert np.asarray(payload["ar_coefficients"]).shape == (2, 1)
    assert np.all(np.abs(np.asarray(payload["ar_coefficients"])) < 1.0)
    assert len(payload["fit_attempt_diagnostics"]) == 2
    assert payload["llf"] == max(item["llf"] for item in payload["fit_attempt_diagnostics"])
    json.dumps(payload)


def test_fit_regime_result_serializes_without_numpy_scalars() -> None:
    rng = np.random.default_rng(5678)
    dates = pd.date_range("2025-06-01", periods=90, freq="B")
    values = rng.normal(0.0, [0.007, 0.005], size=(90, 2))
    returns = pd.DataFrame(values, columns=["sp500", "usdclp"])
    returns.insert(0, "date", dates)

    result = fit_markov_switching_joint_diffusion(
        returns,
        k_regimes=1,
        autoregressive_order=2,
        seed=11,
        search_reps=0,
    )

    payload = result.to_dict()
    dumped = json.dumps(payload, sort_keys=True, allow_nan=False)
    assert "ndarray" not in dumped
    assert payload["k_regimes"] == 1
    assert payload["transition_matrix"] == [[1.0]]
    assert payload["expected_durations"] == [None]
    assert payload["autoregressive_order"] == 2
    assert np.asarray(payload["ar_coefficients"]).shape == (1, 2)

    candidates = fit_regime_candidates(returns, candidate_ks=(1,), autoregressive_order=2, seed=11)
    candidate_payload = [candidate.to_dict() for candidate in candidates]
    assert candidate_payload[0]["k_regimes"] == 1
    assert candidate_payload[0]["autoregressive_order"] == 2
    json.dumps(candidate_payload, sort_keys=True, allow_nan=False)


def _run_import_probe(code: str) -> subprocess.CompletedProcess[str]:
    root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_pricing_imports_without_optional_calibration_stack() -> None:
    probe = _run_import_probe(
        """
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'pandas', 'statsmodels'}:
        raise ModuleNotFoundError(f'blocked optional dependency: {name}', name=name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from finite_element_options.examples.regime_switching_quanto import (
    ContractSpec, FEMGridSpec, TwoFactorRegimeModel, price_contract_fem
)
assert ContractSpec and FEMGridSpec and TwoFactorRegimeModel and price_contract_fem
"""
    )
    assert probe.returncode == 0, probe.stderr


def test_missing_statsmodels_names_calibration_extra() -> None:
    probe = _run_import_probe(
        """
import builtins
import numpy as np
import pandas as pd
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] == 'statsmodels':
        raise ModuleNotFoundError(f'blocked optional dependency: {name}', name=name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from finite_element_options.examples.regime_switching_quanto import fit_markov_switching_joint_diffusion
returns = pd.DataFrame({
    'date': pd.date_range('2025-01-01', periods=40, freq='B'),
    'sp500': np.zeros(40),
    'usdclp': np.zeros(40),
})
for autoregressive_order in (0, 1):
    try:
        fit_markov_switching_joint_diffusion(
            returns, k_regimes=1, autoregressive_order=autoregressive_order
        )
    except ImportError as exc:
        message = str(exc)
        assert 'finite-element-options[calibration]' in message, message
    else:
        raise AssertionError(
            f'expected an actionable ImportError for AR({autoregressive_order})'
        )
"""
    )
    assert probe.returncode == 0, probe.stderr
