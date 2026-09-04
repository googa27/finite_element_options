"""Volatility challenger benchmark tests for regime-switching quanto adoption."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
ADOPTION = "finite_element_options.examples.regime_switching_quanto.adoption"


def _synthetic_levels(n: int = 190, break_at: int = 110) -> pd.DataFrame:
    rng = np.random.default_rng(131)
    break_at = min(max(2, break_at), n - 2)
    low = rng.normal(0.0002, 0.004, size=(break_at, 2))
    high = rng.normal([-0.0003, 0.0001], [0.018, 0.012], size=(n - break_at, 2))
    returns = np.vstack([low, high])
    levels = 1000.0 * np.exp(np.cumsum(returns, axis=0))
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=n, freq="B"),
            "sp500": levels[:, 0],
            "usdclp": levels[:, 1],
        }
    )


@pytest.mark.calibration
@pytest.mark.volatility
@pytest.mark.changepoints
@pytest.mark.validation
def test_seeded_synthetic_benchmark_exercises_arch_and_ruptures() -> None:
    """The benchmark fits real arch challengers and ruptures changepoints."""

    pytest.importorskip("arch")
    pytest.importorskip("ruptures")
    from finite_element_options.examples.regime_switching_quanto.adoption.volatility_benchmark import (
        VolatilityBenchmarkConfig,
        run_volatility_benchmark,
    )

    result = run_volatility_benchmark(
        _synthetic_levels(),
        input_sha256="f" * 64,
        config=VolatilityBenchmarkConfig(
            seed=7,
            holdout_size=16,
            rolling_window=80,
            refit_block=8,
            markov_search_reps=0,
            arch_maxiter=60,
            markov_maxiter=80,
            changepoint_window=12,
            changepoint_penalty=2.0,
        ),
    )
    assert result.schema_version == "regime_volatility_benchmark.v1"
    assert result.observed_response == "100 * (sp500_log_return + usdclp_log_return)"
    assert result.train_start < result.train_end < result.holdout_start <= result.holdout_end
    assert result.immutable_input_sha256 == "f" * 64
    assert len(result.candidates) == 4
    assert {candidate.family for candidate in result.candidates} == {"GJR-GARCH", "EGARCH"}
    assert {candidate.distribution for candidate in result.candidates} == {"student-t", "skewed-t"}
    assert all(candidate.fit_count == 2 for candidate in result.candidates)
    assert result.markov_baseline.fit_count == 2
    assert result.changepoints.breakpoints
    assert result.decision.decision in {"promote", "reject"}
    json.dumps(result.to_dict(), sort_keys=True)


@pytest.mark.validation
def test_hash_mismatch_refuses_input(tmp_path: Path) -> None:
    """The CLI refuses immutable input hash mismatches before producing output."""

    csv_path = tmp_path / "levels.csv"
    _synthetic_levels(40).to_csv(csv_path, index=False)
    output_path = tmp_path / "out.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_regime_volatility_benchmark.py",
            "--input",
            str(csv_path),
            "--expected-sha256",
            "0" * 64,
            "--output",
            str(output_path),
            "--holdout-size",
            "8",
            "--rolling-window",
            "20",
        ],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src")},
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "hash mismatch" in completed.stderr.lower()
    assert not output_path.exists()


@pytest.mark.validation
def test_no_leakage_boundaries_and_metric_calculations() -> None:
    """Boundary derivation and metric helpers are deterministic and leak-free."""

    from finite_element_options.examples.regime_switching_quanto.adoption.volatility_benchmark import (
        VolatilityBenchmarkConfig,
        calculate_var_diagnostics,
        qlike_loss,
        rolling_boundaries,
    )

    cfg = VolatilityBenchmarkConfig(holdout_size=6, rolling_window=10, refit_block=4)
    boundaries = rolling_boundaries(20, cfg)
    assert [(b.train_start, b.train_end, b.holdout_start, b.holdout_end) for b in boundaries] == [
        (4, 14, 14, 18),
        (8, 18, 18, 20),
    ]
    assert all(boundary.train_end == boundary.holdout_start for boundary in boundaries)
    qlike = qlike_loss(np.array([1.0, 4.0]), np.array([1.0, 2.0]))
    np.testing.assert_allclose(qlike, np.array([1.0, np.log(2.0) + 2.0]))
    var = calculate_var_diagnostics(
        observations=np.array([-2.0, -0.5, 0.1, -3.0]),
        var_forecasts=np.array([-1.0, -1.0, -1.0, -1.0]),
        alpha=0.25,
    )
    assert var.exceedance_count == 2
    assert var.exceedance_rate == 0.5
    assert var.coverage_error == 0.25
    assert var.kupiec_pvalue is not None


@pytest.mark.changepoints
@pytest.mark.validation
def test_changepoint_recovery_near_known_variance_break() -> None:
    """Ruptures-based detection recovers a deterministic variance shift."""

    pytest.importorskip("ruptures")
    from finite_element_options.examples.regime_switching_quanto.adoption.volatility_benchmark import (
        detect_volatility_changepoints,
    )
    from finite_element_options.examples.regime_switching_quanto.quality import (
        prepare_joint_log_returns,
    )

    returns, _ = prepare_joint_log_returns(_synthetic_levels(n=180, break_at=95))
    comparison = detect_volatility_changepoints(
        returns,
        high_volatility_probability=np.r_[np.zeros(94), np.ones(len(returns) - 94)],
        window=10,
        penalty=1.5,
    )
    indices = [item.index for item in comparison.breakpoints]
    assert any(abs(index - 94) <= 15 for index in indices), indices
    assert comparison.nearest_regime_gap_days is not None


@pytest.mark.validation
def test_failure_records_and_canonical_artifact_are_serializable() -> None:
    """Failure/result records are typed, JSON-safe and canonical-json stable."""

    from finite_element_options.examples.regime_switching_quanto.adoption.volatility_benchmark import (
        CandidateFailure,
        VolatilityBenchmarkConfig,
        canonical_json,
        canonical_json_sha256,
    )

    failure = CandidateFailure(kind="optimizer", message="nonfinite objective", fit_count=3)
    payload = failure.to_dict()
    assert payload == {"kind": "optimizer", "message": "nonfinite objective", "fit_count": 3}
    encoded = canonical_json({"b": 1, "a": payload})
    assert encoded == canonical_json(json.loads(encoded))
    assert (
        len(canonical_json_sha256({"config": VolatilityBenchmarkConfig(seed=11).to_dict()})) == 64
    )


def test_atomic_artifact_write_replaces_target_without_shared_temp_file(tmp_path: Path) -> None:
    """Canonical artifact writes must be atomic and leave no temp-file debris."""

    from finite_element_options.examples.regime_switching_quanto.adoption.volatility_benchmark import (
        write_atomic_json,
    )

    target = tmp_path / "artifact.json"
    target.write_text("stale\n", encoding="utf-8")

    digest = write_atomic_json(target, {"schema": "test", "value": 7})

    assert target.read_text(encoding="utf-8") == '{"schema":"test","value":7}\n'
    assert len(digest) == 64
    assert list(tmp_path.glob(".artifact.json.*.tmp")) == []


def test_adoption_facade_and_base_import_do_not_eagerly_load_optional_stacks() -> None:
    """The volatility benchmark remains lazy behind the adoption facade."""

    code = f"""
    import builtins
    import importlib
    import sys
    blocked = {{'arch', 'ruptures', 'statsmodels', 'pandas'}}
    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name.split('.')[0] in blocked:
            raise ModuleNotFoundError(f'blocked optional dependency: {{name}}', name=name)
        return original_import(name, *args, **kwargs)
    builtins.__import__ = guarded_import
    import finite_element_options
    facade = importlib.import_module({ADOPTION!r})
    assert {ADOPTION + ".volatility_benchmark"!r} not in sys.modules
    assert 'run_volatility_benchmark' in dir(facade)
    """
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src")},
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.validation
def test_nonconverged_markov_baseline_blocks_promotion() -> None:
    """A failed/nonconverged Markov baseline makes challenger promotion invalid."""

    from finite_element_options.examples.regime_switching_quanto.adoption.volatility_benchmark import (
        _promotion_decision,
    )
    from finite_element_options.examples.regime_switching_quanto.adoption.volatility_contracts import (
        CandidateBenchmarkResult,
        CandidateFailure,
        MarkovBaselineResult,
        VarDiagnostics,
    )

    candidate_var = VarDiagnostics(0.05, 6, 0.05, 0.0, 0.0, 1.0)
    candidate = CandidateBenchmarkResult(
        "GJR-GARCH",
        "student-t",
        True,
        None,
        6,
        0.5,
        -1.0,
        candidate_var,
        {"available": True, "l1_relative_first_last": 0.1, "parameter_count": 3},
    )
    baseline = MarkovBaselineResult(
        "statsmodels MarkovAutoregression",
        "gaussian-mixture",
        3,
        2,
        False,
        CandidateFailure("optimizer", "one or more fits did not converge", 6),
        6,
        0.1,
        10.0,
        candidate_var,
        {"available": True},
        "diagnostic forecasts only",
    )

    decision = _promotion_decision([candidate], baseline)

    assert decision.decision == "reject"
    assert decision.selected_candidate == "GJR-GARCH/student-t"
    assert decision.reasons == [
        "invalid Markov AR(2) baseline: optimizer one or more fits did not converge; "
        "challenger promotion is disabled until the baseline converges"
    ]


@pytest.mark.calibration
@pytest.mark.validation
def test_any_nonconverged_markov_refit_fails_baseline_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One nonconverged Markov rolling refit marks the whole baseline invalid."""

    module = pytest.importorskip("statsmodels.tsa.regime_switching.markov_autoregression")
    from finite_element_options.examples.regime_switching_quanto.adoption.volatility_contracts import (
        RollingBoundary,
        VolatilityBenchmarkConfig,
    )
    from finite_element_options.examples.regime_switching_quanto.adoption.volatility_markov import (
        fit_markov_baseline,
    )

    class FakeModel:
        param_names = [
            "const[0]",
            "const[1]",
            "sigma2[0]",
            "sigma2[1]",
            "ar.L1[0]",
            "ar.L1[1]",
            "ar.L2[0]",
            "ar.L2[1]",
        ]

    class FakeFitted:
        k_regimes = 2
        model = FakeModel()
        params = np.array([0.0, 1.0, 0.5, 0.5, 0.1, 0.1, 0.0, 0.0])
        smoothed_marginal_probabilities = np.array([[0.6, 0.4]])
        regime_transition = np.array([[[0.8], [0.1]], [[0.2], [0.9]]])

        def __init__(self, converged: bool) -> None:
            self.mle_retvals = {"converged": converged}

    class FakeMarkovAutoregression:
        fit_count = 0

        def __init__(self, data, **kwargs) -> None:  # type: ignore[no-untyped-def]
            self.data = data
            self.kwargs = kwargs

        def fit(self, **kwargs):  # type: ignore[no-untyped-def]
            del kwargs
            type(self).fit_count += 1
            return FakeFitted(converged=type(self).fit_count == 1)

    monkeypatch.setattr(module, "MarkovAutoregression", FakeMarkovAutoregression)

    result = fit_markov_baseline(
        np.linspace(-0.2, 0.2, 14),
        [RollingBoundary(0, 10, 10, 12), RollingBoundary(2, 12, 12, 14)],
        VolatilityBenchmarkConfig(rolling_window=10, holdout_size=4, refit_block=2),
    )

    assert result.fit_count == 2
    assert result.converged is False
    assert result.failure is not None
    assert result.failure.kind == "optimizer"
    assert result.failure.message == "one or more fits did not converge"
    assert result.failure.fit_count == 2


@pytest.mark.validation
def test_arch_candidate_uses_analytic_one_step_distribution_forecasts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ARCH challengers must not replace Student/skewed-t innovations with Gaussian simulation."""

    from finite_element_options.examples.regime_switching_quanto.adoption import (
        volatility_benchmark,
    )
    from finite_element_options.examples.regime_switching_quanto.adoption.volatility_contracts import (
        RollingBoundary,
        VolatilityBenchmarkConfig,
    )

    calls: dict[str, object] = {}

    class FakeDistribution:
        def loglikelihood(self, params, residuals, variance, *, individual):  # type: ignore[no-untyped-def]
            calls["distribution_params"] = tuple(float(item) for item in params)
            assert individual is True
            return np.full_like(np.asarray(residuals, dtype=float), -0.25)

        def ppf(self, alpha, params):  # type: ignore[no-untyped-def]
            calls["ppf_params"] = tuple(float(item) for item in params)
            return -2.0

    class FakeModel:
        param_names = ["mu", "omega", "alpha[1]", "gamma[1]", "beta[1]", "nu"]
        distribution = FakeDistribution()

        def __init__(self, data) -> None:  # type: ignore[no-untyped-def]
            self.data = np.asarray(data, dtype=float)

        def fit(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            assert args == ()
            assert kwargs["last_obs"] == 10
            calls["fit_last_obs"] = kwargs["last_obs"]
            return FakeFitted(self)

    class FakeFitted:
        convergence_flag = 0
        params = {
            "mu": 0.0,
            "omega": 1.0,
            "alpha[1]": 0.1,
            "gamma[1]": 0.1,
            "beta[1]": 0.8,
            "nu": 8.0,
        }

        def __init__(self, model) -> None:  # type: ignore[no-untyped-def]
            self.model = model

        def forecast(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            assert args == ()
            assert kwargs == {
                "start": 9,
                "horizon": 1,
                "align": "target",
                "method": "analytic",
                "reindex": False,
            }
            calls["forecast_kwargs"] = kwargs
            return FakeForecast()

    class FakeForecast:
        mean = pd.DataFrame({"h.1": [np.nan, 0.1, 0.2]}, index=[9, 10, 11])
        residual_variance = pd.DataFrame({"h.1": [np.nan, 1.0, 1.1]}, index=[9, 10, 11])

    class FakeArch:
        def arch_model(self, data, **kwargs):  # type: ignore[no-untyped-def]
            calls["model_data_len"] = len(data)
            calls["dist"] = kwargs["dist"]
            return FakeModel(data)

    monkeypatch.setattr(volatility_benchmark, "require_optional", lambda name: FakeArch())
    response = np.linspace(-0.2, 0.3, 12)
    result = volatility_benchmark._fit_arch_candidate(
        returns=pd.DataFrame(),
        response=response,
        boundaries=[RollingBoundary(0, 10, 10, 12)],
        cfg=VolatilityBenchmarkConfig(rolling_window=10, holdout_size=2, refit_block=2),
        family="GJR-GARCH",
        distribution="student-t",
    )

    assert calls["model_data_len"] == 12
    assert calls["dist"] == "StudentsT"
    assert calls["distribution_params"] == (8.0,)
    assert calls["ppf_params"] == (8.0,)
    assert result.converged is True
    assert result.fit_count == 1
    assert result.mean_predictive_log_score == -0.25


@pytest.mark.validation
def test_markov_forecast_filters_holdout_without_future_leakage() -> None:
    """Changing y_t updates t+1 Markov forecasts but cannot alter the t forecast."""

    from finite_element_options.examples.regime_switching_quanto.adoption.volatility_markov import (
        markov_forecast,
        statsmodels_row_stochastic_transition,
        update_markov_probabilities,
    )

    class FakeModel:
        param_names = [
            "const[0]",
            "const[1]",
            "sigma2[0]",
            "sigma2[1]",
            "ar.L1[0]",
            "ar.L1[1]",
            "ar.L2[0]",
            "ar.L2[1]",
        ]

    class FakeFitted:
        k_regimes = 2
        model = FakeModel()
        params = np.array([0.0, 4.0, 0.25, 0.25, 0.5, 0.5, 0.0, 0.0])
        smoothed_marginal_probabilities = np.array([[0.5, 0.5]])
        # statsmodels orientation: [next_regime, previous_regime, time]
        regime_transition = np.array([[[0.7], [0.2]], [[0.3], [0.8]]])

    transition = statsmodels_row_stochastic_transition(FakeFitted())
    np.testing.assert_allclose(transition, np.array([[0.7, 0.3], [0.2, 0.8]]))
    np.testing.assert_allclose(transition.sum(axis=1), np.ones(2))
    np.testing.assert_allclose(np.array([1.0, 0.0]) @ transition, np.array([0.7, 0.3]))

    low_path = np.array([1.0, 1.0])
    high_path = np.array([5.0, 1.0])
    low_mean, low_variance, low_log, low_var = markov_forecast(
        FakeFitted(), np.array([1.0, 2.0]), low_path, 0.05
    )
    high_mean, high_variance, high_log, high_var = markov_forecast(
        FakeFitted(), np.array([1.0, 2.0]), high_path, 0.05
    )

    np.testing.assert_allclose(low_mean[0], high_mean[0])
    np.testing.assert_allclose(low_variance[0], high_variance[0])
    np.testing.assert_allclose(low_var[0], high_var[0])
    assert low_log[0] != high_log[0]
    assert low_mean[1] != high_mean[1]

    first_prior = np.array([0.5, 0.5]) @ transition
    first_component_mean = np.array([1.0, 5.0])
    low_filter = update_markov_probabilities(
        first_prior, low_path[0], first_component_mean, np.array([0.25, 0.25])
    )
    high_filter = update_markov_probabilities(
        first_prior, high_path[0], first_component_mean, np.array([0.25, 0.25])
    )
    assert low_filter[0] > 0.99
    assert high_filter[1] > 0.99


@pytest.mark.validation
def test_markov_high_volatility_probability_returns_typed_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed full-sample Markov regime probabilities are serialized instead of zero-filled."""

    import builtins

    from finite_element_options.examples.regime_switching_quanto.adoption.volatility_markov import (
        full_markov_high_volatility_probability,
    )
    from finite_element_options.examples.regime_switching_quanto.adoption.volatility_contracts import (
        VolatilityBenchmarkConfig,
    )

    original_import = builtins.__import__

    def failing_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "statsmodels.tsa.regime_switching.markov_autoregression":
            raise ModuleNotFoundError("blocked statsmodels", name=name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    result = full_markov_high_volatility_probability(
        np.arange(30, dtype=float), VolatilityBenchmarkConfig(rolling_window=10, holdout_size=2)
    )

    assert result.probability is None
    assert result.failure is not None
    assert result.failure.kind == "dependency"
    assert result.failure.fit_count == 0


@pytest.mark.validation
def test_real_artifact_is_bounded_canonical_and_schema_stable() -> None:
    """The checked-in real benchmark artifact is deterministic and summary-bounded."""

    from finite_element_options.examples.regime_switching_quanto.adoption.volatility_benchmark import (
        canonical_json,
    )
    from finite_element_options.examples.regime_switching_quanto.adoption.volatility_metrics import (
        file_sha256,
    )

    artifact = ROOT / "docs/evidence/regime_switching_quanto_volatility_benchmark_2026-09-03.json"
    assert artifact.exists()
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert artifact.read_text(encoding="utf-8") == canonical_json(payload) + "\n"
    assert (
        file_sha256(artifact) == "3ef33542865cc7370bc639b15b60aba207a2be3981ad77b9d1132f5f0e15f9ad"
    )
    encoded = json.dumps(payload, sort_keys=True)
    assert "/tmp/" not in encoded
    assert "pdp_joint_levels.csv" not in encoded
    assert "quarantined_rows" not in payload["data_quality"]
    assert payload["data_quality"]["quarantined_row_count"] >= 0
    assert "breakpoint_count" in payload["changepoints"]
    assert len(payload["changepoints"]["breakpoints"]) <= 20
    assert payload["decision"]["decision"] == "reject"
