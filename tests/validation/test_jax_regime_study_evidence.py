"""Hash-bound validation for the JAX regime research evidence bundle."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.validation
ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/evidence/jax_regime_study_2026-09-07.json"
SIDECAR = EVIDENCE.with_suffix(EVIDENCE.suffix + ".sha256")
LOCK = ROOT / "environments/jax-regime-py312/requirements.lock"
TEST_LOCK = ROOT / "environments/jax-regime-py312/test-requirements.lock"
CI_LOCK = ROOT / "environments/jax-regime-py312/ci-requirements.lock"
VISUAL_LOCK = ROOT / "environments/jax-regime-visual-py312/requirements.lock"
IMAGE_DIR = ROOT / "docs/images"
ARTIFACT_MANIFEST = IMAGE_DIR / "jax_regime_study_2026-09-07.sha256"
EXPECTED_EVIDENCE_SHA256 = "2faf09c5316d59ebdeec31e26c85483c87cee5f92be786f563923d6e11a9a854"
EXPECTED_LOCK_SHA256 = "42f83eb5da5716b7f228bdb94338beb5b552d9fe0fdb866449e5cb31b8c46a7c"
EXPECTED_TEST_LOCK_SHA256 = "ab7d270889b7d1b74e7723668d972173b86e2e5d763d6385ad6566d5ac418af0"
EXPECTED_CI_LOCK_SHA256 = "5dbd4f3f15dce41e455b4cde0cb453c23782379cc4b37fef0db526ec75e0580b"
EXPECTED_VISUAL_LOCK_SHA256 = "8110cfc79dcaffaf734730272ae5db84174a25a3304241a964422de2988891b6"
EXPECTED_ARTIFACT_HASHES = {
    "jax_regime_study_2026-09-07.png": (
        "46e3e795c5bf693d117550a0e1b57b5bc77c7e05792120d9e750ede56df1df24"
    ),
    "jax_regime_study_2026-09-07.pdf": (
        "06d0e5d298e00cdbd3aee83248baec0f53c8ca4c4ca8882dc7a16a7d66585646"
    ),
}


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _locked_versions(path: Path) -> dict[str, str]:
    pattern = re.compile(r"^([a-z0-9-]+)==([^ ;\\]+)")
    return {
        match.group(1): match.group(2)
        for line in path.read_text(encoding="utf-8").splitlines()
        if (match := pattern.match(line)) is not None
    }


def test_jax_regime_locks_agree_on_every_shared_distribution() -> None:
    locks = [_locked_versions(path) for path in (LOCK, TEST_LOCK, CI_LOCK)]
    for left_index, left in enumerate(locks):
        for right in locks[left_index + 1 :]:
            shared = set(left) & set(right)
            assert {name: left[name] for name in shared} == {name: right[name] for name in shared}


def _strings(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [item for nested in value.values() for item in _strings(nested)]
    if isinstance(value, list):
        return [item for nested in value for item in _strings(nested)]
    return [value] if isinstance(value, str) else []


def test_jax_regime_evidence_is_hash_bound_and_promotably_honest() -> None:
    assert _digest(EVIDENCE) == EXPECTED_EVIDENCE_SHA256
    assert SIDECAR.read_text(encoding="utf-8").split()[0] == EXPECTED_EVIDENCE_SHA256
    assert _digest(LOCK) == EXPECTED_LOCK_SHA256
    assert _digest(TEST_LOCK) == EXPECTED_TEST_LOCK_SHA256
    assert _digest(CI_LOCK) == EXPECTED_CI_LOCK_SHA256
    payload = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "feo-jax-regime-study-v2"
    assert payload["status"] == "passed"
    assert {
        key: payload["config"][key]
        for key in (
            "seed",
            "num_states",
            "em_iters",
            "holdout_fraction",
            "warmup",
            "posterior_samples",
            "chains",
            "pricing_paths",
        )
    } == {
        "seed": 20260907,
        "num_states": 4,
        "em_iters": 250,
        "holdout_fraction": 0.15,
        "warmup": 300,
        "posterior_samples": 300,
        "chains": 2,
        "pricing_paths": 4096,
    }
    em_iterations = payload["config"]["em_iters"]
    assert payload["hmm"]["dynamax_full_fit"]["em_iterations"] == em_iterations
    assert all(
        row["em_iterations"] == em_iterations
        for row in payload["hmm"]["candidate_comparison"]
        if row["engine"] == "dynamax"
    )
    assert f"{em_iterations}-iteration starts" in payload["hmm"]["selection"]["rule"]
    posterior_filtered = np.asarray(
        payload["hmm"]["numpyro"]["posterior_end_sample_filtered_probs"]
    )
    posterior_transition = np.asarray(payload["hmm"]["numpyro"]["posterior_mean_transition_matrix"])
    posterior_forecast = np.asarray(
        payload["hmm"]["numpyro"]["posterior_first_interval_forecast_probs"]
    )
    forecast = np.asarray(payload["pricing"]["diffrax"]["first_interval_regime_probs"])
    np.testing.assert_allclose(
        posterior_forecast,
        posterior_filtered @ posterior_transition,
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(forecast, posterior_forecast, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(np.sum(forecast), 1.0, rtol=0.0, atol=1.0e-14)
    assert payload["claims"] == {
        "diffrax_scope": (
            "Diffrax Euler is a tested SDE abstraction, not an accuracy upgrade; the aligned "
            "piecewise-constant log diffusion is conditionally exact under the JAX scan oracle."
        ),
        "inference_scope": (
            "NumPyro inference is an empirical-Bayes analysis conditional on volatility-ordered "
            "DYNAMAX EM reference parameters; intervals do not include uncertainty from selecting "
            "or estimating that reference."
        ),
        "market_calibrated": False,
        "measure_policy": (
            "DYNAMAX/NumPyro estimate historical P dynamics; pricing reuses the P transition "
            "matrix under a constrained domestic Qd scenario with no option-implied regime-risk premium."
        ),
        "production_ready": False,
        "regime_timing_scope": (
            "Pricing uses JAX-native daily discrete-HMM regime paths. SciPy CTMC projection is "
            "an embeddability/law diagnostic only; no continuous-time regime simulation is claimed."
        ),
        "research_only": True,
    }
    assert payload["data"]["archive_sha256"] == (
        "ca2debc4fcbf9bd6fb958a5cfcb986a3e8080c7923418c9d2367fcd2d9a99721"
    )
    assert payload["data"]["levels_sha256"] == (
        "aa7ab317266bf37463e27aba9a4e990fa349bb0a6e0aefb5741e93480e0f79f4"
    )
    assert payload["data"]["row_count"] == 4_184
    assert payload["data"]["preprocessing"] == {
        "excluded_level_rows": 161,
        "invalid_equity_rows": 0,
        "invalid_fx_rows": 2,
        "missing_or_unparseable_equity_rows": 153,
        "missing_or_unparseable_fx_rows": 6,
        "quarantined_rows": [
            {"date": "2014-04-10", "reason": "fx_not_finite_or_below_100"},
            {"date": "2016-12-22", "reason": "fx_not_finite_or_below_100"},
        ],
        "return_construction_rule": (
            "adjacent_valid_joint_levels_after_exclusions_no_calendar_gap_rescaling"
        ),
        "source_level_rows": 4_346,
        "valid_level_rows": 4_185,
    }

    hmm = payload["hmm"]
    assert hmm["selection"]["states"] == 4
    assert hmm["selection"]["selected_by_heldout_score"] == 4
    assert hmm["selection"]["selected_model_beats_statsmodels_var"] is True
    assert hmm["statsmodels_var_baseline"]["finite"] is True
    assert all(row["multistart_converged"] for row in hmm["candidate_comparison"])
    assert all(
        row["converged_starts"] >= row["minimum_converged_starts_required"]
        for row in hmm["candidate_comparison"]
    )
    assert hmm["dynamax_full_fit"]["multistart_converged"] is True
    assert hmm["ctmc_generator"]["passed"] is True
    posterior = hmm["numpyro"]
    assert posterior["chains"] >= 2
    assert posterior["draws_per_chain"] >= 200
    assert posterior["divergences"] == 0
    assert "diverging" in posterior["diagnostic_fields"]
    assert posterior["maximum_rhat"] <= 1.05
    assert posterior["minimum_ess"] >= 100.0
    assert posterior["weakly_identified_parameters"] == []
    sensitivity = hmm["prior_sensitivity"]
    assert sensitivity["chains"] >= 2
    assert sensitivity["sensitivity_draws_per_chain"] >= 200
    assert sensitivity["passed"] is True
    for profile in sensitivity["profiles"].values():
        assert profile["diagnostics"]["divergences"] == 0
        assert profile["diagnostics"]["maximum_rhat"] <= 1.05
        assert profile["diagnostics"]["minimum_ess"] >= 100.0
        assert profile["diagnostics"]["finite"] is True

    pricing = payload["pricing"]
    intervals = pricing["posterior_parameter_price_intervals"]
    assert all(
        row["posterior_draws"] >= 128
        and row["posterior_draws_per_chain"] >= 64
        and row["available_draws_per_chain"] >= 300
        and row["paths_per_draw"] >= 131_072
        and row["common_random_numbers"] is True
        and row["mc_se_to_posterior_half_width"] <= 0.5
        and row["split_subsample_max_quantile_delta_to_full_half_width"] <= 0.5
        for row in intervals.values()
    )
    assert pricing["martingale_checks"]["passed"] is True
    assert pricing["strike_monotonicity"]["passed"] is True
    assert pricing["brownian_bridge_refinement"]["passed"] is True
    assert pricing["matched_historical_three_state_jax_numpy_oracle"]["passed"] is True
    assert all(row["passed_5se"] for row in pricing["one_state_analytical_oracles"].values())
    assert payload["verification"]["diagnostics_passed"] is True
    assert all(payload["verification"]["gates"].values())
    assert all("/home/" not in value for value in _strings(payload))


def test_public_report_tracks_hash_bound_numerical_evidence() -> None:
    payload = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    report = (ROOT / "docs/JAX_REGIME_STUDY.md").read_text(encoding="utf-8")
    assert EXPECTED_EVIDENCE_SHA256 in report
    assert EXPECTED_LOCK_SHA256 in report
    assert EXPECTED_VISUAL_LOCK_SHA256 in report
    assert all(digest in report for digest in EXPECTED_ARTIFACT_HASHES.values())

    for candidate in payload["hmm"]["candidate_comparison"]:
        assert f"{abs(candidate['heldout_mean_log_score']):.6f}" in report
    fit = payload["hmm"]["dynamax_full_fit"]
    for value in fit["annualized_composite_volatility_percent"]:
        assert f"{value:.2f}%" in report
    for values in (fit["occupancy"], fit["current_filtered_probs"]):
        for value in values:
            assert f"{100.0 * value:.2f}%" in report

    posterior = payload["hmm"]["numpyro"]
    assert f"{posterior['maximum_rhat']:.4f}" in report
    assert f"{posterior['minimum_ess']:.1f}" in report
    point_prices = payload["pricing"]["diffrax"]["point_prices"]
    intervals = payload["pricing"]["posterior_parameter_price_intervals"]
    for name, point in point_prices.items():
        interval = intervals[name]
        for value in (
            point["price_clp"],
            2.0 * point["standard_error_clp"],
            interval["posterior_parameter_q05_clp"],
            interval["posterior_parameter_q95_clp"],
            interval["total_computational_q05_clp"],
            interval["total_computational_q95_clp"],
        ):
            assert f"{value:,.2f}" in report


def test_jax_regime_public_visuals_are_hash_bound() -> None:
    assert _digest(VISUAL_LOCK) == EXPECTED_VISUAL_LOCK_SHA256
    rows = {
        filename: digest
        for digest, filename in (
            line.split(maxsplit=1)
            for line in ARTIFACT_MANIFEST.read_text(encoding="utf-8").splitlines()
        )
    }
    assert rows == EXPECTED_ARTIFACT_HASHES
    assert {
        filename: _digest(IMAGE_DIR / filename) for filename in EXPECTED_ARTIFACT_HASHES
    } == EXPECTED_ARTIFACT_HASHES
