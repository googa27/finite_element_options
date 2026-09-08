"""Dependency-light contract and data-loader tests for the JAX regime study."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest

from finite_element_options.examples.regime_switching_quanto.jax_regime.contracts import (
    JaxRegimeStudyConfig,
    MAX_POSTERIOR_PRICING_PATH_STEPS,
    PDPObservationBatch,
    PDPPreprocessingAudit,
)
from finite_element_options.examples.regime_switching_quanto.jax_regime.data import (
    _MEMBER_ROOT,
    load_pdp_observations,
)
from finite_element_options.examples.regime_switching_quanto.jax_regime.hmm.experiment import (
    _select_converged_multistart,
)
from finite_element_options.examples.regime_switching_quanto.jax_regime.hmm.numpyro_model import (
    _diagnostic_extrema,
    _required_divergence_count,
)
from finite_element_options.examples.regime_switching_quanto.jax_regime.pricing.analytic import (
    one_state_price,
)
from finite_element_options.examples.regime_switching_quanto.jax_regime.pricing.study import (
    _bounded_pricing_paths,
    _historical_prices,
    _matched_oracle_config,
    _oracle_z_score,
    _posterior_pricing_paths,
    _publication_or_requested_paths,
)
from finite_element_options.examples.regime_switching_quanto.jax_regime.study import _selection_rule


def _fixture(path: Path) -> tuple[str, str]:
    levels = (
        "date,sp500,usdclp\n"
        "2026-01-01,100,800\n"
        "2026-01-02,101,808\n"
        "2026-01-03,102,5\n"
        "2026-01-04,103,824\n"
    ).encode()
    provenance = json.dumps(
        {"requested_window": {"start": "2026-01-01", "end_exclusive": "2026-01-05"}}
    ).encode()
    pricing = json.dumps(
        {
            "pricing": [
                {
                    "contract": "fixture",
                    "fem_fine_clp": 1.0,
                    "richardson_extrapolated_clp": 1.05,
                    "mc_clp": 1.1,
                    "mc_standard_error_clp": 0.1,
                }
            ]
        }
    ).encode()
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as bundle:
        bundle.writestr(f"{_MEMBER_ROOT}pdp_joint_levels.csv", levels)
        bundle.writestr(f"{_MEMBER_ROOT}input_provenance.json", provenance)
        bundle.writestr(f"{_MEMBER_ROOT}pricing_results.json", pricing)
    return sha256(path.read_bytes()).hexdigest(), sha256(levels).hexdigest()


def test_content_addressed_loader_quarantines_malformed_fx(tmp_path: Path) -> None:
    archive = tmp_path / "fixture.zip"
    archive_hash, levels_hash = _fixture(archive)
    batch = load_pdp_observations(
        archive,
        expected_archive_sha256=archive_hash,
        expected_levels_sha256=levels_hash,
    )
    assert batch.row_count == 2
    assert batch.dates == ("2026-01-02", "2026-01-04")
    assert batch.returns[0] == pytest.approx((math.log(1.01), math.log(1.01)))
    assert batch.equity_spot == 103.0
    assert batch.fx_spot == 824.0
    assert batch.preprocessing.to_dict() == {
        "source_level_rows": 4,
        "valid_level_rows": 3,
        "missing_or_unparseable_equity_rows": 0,
        "missing_or_unparseable_fx_rows": 0,
        "invalid_equity_rows": 0,
        "invalid_fx_rows": 1,
        "quarantined_rows": [{"date": "2026-01-03", "reason": "fx_not_finite_or_below_100"}],
        "return_construction_rule": (
            "adjacent_valid_joint_levels_after_exclusions_no_calendar_gap_rescaling"
        ),
        "excluded_level_rows": 1,
    }
    assert "/home/" not in json.dumps(batch.to_dict())


def test_loader_parses_the_exact_snapshot_that_passed_hash_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "fixture.zip"
    archive_hash, levels_hash = _fixture(archive)
    verified_snapshot = archive.read_bytes()
    archive.write_bytes(b"atomically replaced after snapshot")

    def _snapshot_read(path: Path) -> bytes:
        assert path == archive.resolve()
        return verified_snapshot

    monkeypatch.setattr(Path, "read_bytes", _snapshot_read)
    batch = load_pdp_observations(
        archive,
        expected_archive_sha256=archive_hash,
        expected_levels_sha256=levels_hash,
    )
    assert batch.archive_sha256 == archive_hash
    assert batch.row_count == 2
    assert _historical_prices(verified_snapshot)[0]["contract"] == "fixture"


def test_loader_fails_closed_on_archive_hash_mismatch(tmp_path: Path) -> None:
    archive = tmp_path / "fixture.zip"
    _archive_hash, levels_hash = _fixture(archive)
    with pytest.raises(ValueError, match="archive SHA-256 mismatch"):
        load_pdp_observations(
            archive,
            expected_archive_sha256="0" * 64,
            expected_levels_sha256=levels_hash,
        )


def test_research_claim_flags_cannot_be_weakened() -> None:
    with pytest.raises(ValueError, match="research-only"):
        JaxRegimeStudyConfig(research_only=False)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="research-only"):
        JaxRegimeStudyConfig(market_calibrated=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="research-only"):
        JaxRegimeStudyConfig(research_only=1)  # type: ignore[arg-type]


def test_numpy_scalar_config_values_normalize_to_strict_json_primitives() -> None:
    config = JaxRegimeStudyConfig(
        seed=np.int64(20_260_907),  # type: ignore[arg-type]
        posterior_samples=np.int32(300),  # type: ignore[arg-type]
        maturity_years=np.float32(0.5),  # type: ignore[arg-type]
    )
    payload = config.to_dict()
    json.dumps(payload, allow_nan=False)
    assert type(payload["seed"]) is int
    assert type(payload["posterior_samples"]) is int
    assert type(payload["maturity_years"]) is float


def test_missing_numpyro_divergence_telemetry_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="omitted required 'diverging'"):
        _required_divergence_count({})


def test_nonfinite_numpyro_diagnostics_are_json_safe_and_fail_closed() -> None:
    diagnostics = _diagnostic_extrema(
        {"theta": {"r_hat": np.array([float("nan")]), "n_eff": np.array([2.0])}}
    )
    assert diagnostics["maximum_rhat"] is None
    assert diagnostics["minimum_ess"] is None
    assert diagnostics["finite"] is False
    assert diagnostics["weakly_identified_parameters"][0]["rhat"] is None
    json.dumps(diagnostics, allow_nan=False)


def test_zero_variance_oracle_score_fails_closed_on_nonzero_error() -> None:
    assert _oracle_z_score(0.0, 0.0) == 0.0
    assert _oracle_z_score(2.0, 0.0) is None
    assert _oracle_z_score(-2.0, 0.0) is None
    assert _oracle_z_score(2.0, 0.5) == 4.0
    assert json.dumps({"z_score": _oracle_z_score(2.0, 0.0)}, allow_nan=False) == (
        '{"z_score": null}'
    )


def test_resource_limits_and_finite_observations_fail_closed() -> None:
    with pytest.raises(ValueError, match="em_iters must be an integer"):
        JaxRegimeStudyConfig(em_iters=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="em_iters must be at least 250"):
        JaxRegimeStudyConfig(em_iters=249)
    high_iteration_config = JaxRegimeStudyConfig(em_iters=500)
    assert high_iteration_config.em_iters == 500
    assert "500-iteration starts" in _selection_rule(high_iteration_config)
    with pytest.raises(ValueError, match="seed must be an integer"):
        JaxRegimeStudyConfig(seed=True)
    with pytest.raises(ValueError, match="reserve deterministic offsets"):
        JaxRegimeStudyConfig(seed=-1)
    with pytest.raises(ValueError, match="reserve deterministic offsets"):
        JaxRegimeStudyConfig(seed=0xFFFF_FFFF)
    with pytest.raises(ValueError, match="domestic_rate must be a finite real number"):
        JaxRegimeStudyConfig(domestic_rate=float("nan"))
    with pytest.raises(ValueError, match="domestic_rate must lie"):
        JaxRegimeStudyConfig(domestic_rate=1.1)
    with pytest.raises(ValueError, match="finite split-chain diagnostics"):
        JaxRegimeStudyConfig(posterior_samples=1)
    with pytest.raises(ValueError, match="must not exceed"):
        JaxRegimeStudyConfig(pricing_paths=1_000_001)
    with pytest.raises(ValueError, match="pricing_paths must be at least 2"):
        JaxRegimeStudyConfig(pricing_paths=1)
    with pytest.raises(ValueError, match="at least 2"):
        JaxRegimeStudyConfig(chains=1)
    with pytest.raises(ValueError, match="steps_per_year must be 252"):
        JaxRegimeStudyConfig(steps_per_year=504)
    with pytest.raises(ValueError, match="at least one daily pricing step"):
        JaxRegimeStudyConfig(maturity_years=0.001)
    with pytest.raises(ValueError, match="maturity_years must be a finite real number"):
        JaxRegimeStudyConfig(maturity_years=float("nan"))
    with pytest.raises(ValueError, match="whole number of daily pricing steps"):
        JaxRegimeStudyConfig(maturity_years=0.01)
    assert JaxRegimeStudyConfig(maturity_years=1.0 / 252.0).pricing_steps == 1
    with pytest.raises(ValueError, match="more than 3,660 pricing steps"):
        JaxRegimeStudyConfig(maturity_years=15.0)
    with pytest.raises(ValueError, match="path-steps"):
        JaxRegimeStudyConfig(maturity_years=14.0, pricing_paths=5_000)
    long_horizon_paths = _bounded_pricing_paths(131_072, 2, 3_660)
    assert long_horizon_paths * 3_660 <= 131_072 * 126
    short_config = JaxRegimeStudyConfig(maturity_years=1.0 / 252.0, pricing_paths=200_000)
    matched_config = _matched_oracle_config(
        short_config,
        maturity_years=0.5,
        domestic_rate=0.045,
        foreign_rate=0.0439,
        dividend_yield=0.0,
    )
    assert matched_config.pricing_steps == 126
    assert matched_config.pricing_paths == 131_072

    low_path_config = JaxRegimeStudyConfig(pricing_paths=2)
    assert _posterior_pricing_paths(low_path_config, 128) == 2
    assert _publication_or_requested_paths(low_path_config, 4_096, 126) == 2
    low_path_oracle = _matched_oracle_config(
        low_path_config,
        maturity_years=0.5,
        domestic_rate=0.045,
        foreign_rate=0.0439,
        dividend_yield=0.0,
    )
    assert low_path_oracle.pricing_paths == 2
    canonical_config = JaxRegimeStudyConfig()
    canonical_paths = _posterior_pricing_paths(canonical_config, 128)
    assert canonical_paths == 131_072
    assert (
        canonical_paths * canonical_config.pricing_steps * 128 == MAX_POSTERIOR_PRICING_PATH_STEPS
    )
    heavy_config = JaxRegimeStudyConfig(
        maturity_years=14.0,
        pricing_paths=4_500,
        posterior_samples=64,
        chains=8,
    )
    heavy_draws = 512
    bounded_paths = _posterior_pricing_paths(heavy_config, heavy_draws)
    assert bounded_paths < heavy_config.pricing_paths
    assert (
        bounded_paths * heavy_config.pricing_steps * heavy_draws <= MAX_POSTERIOR_PRICING_PATH_STEPS
    )
    kwargs = {
        "dates": ("2026-01-01",),
        "levels_sha256": "a" * 64,
        "archive_sha256": "b" * 64,
        "requested_start": "2026-01-01",
        "requested_end_exclusive": "2026-01-02",
        "equity_spot": 100.0,
        "fx_spot": 900.0,
        "preprocessing": PDPPreprocessingAudit(
            source_level_rows=1,
            valid_level_rows=1,
            missing_or_unparseable_equity_rows=0,
            missing_or_unparseable_fx_rows=0,
            invalid_equity_rows=0,
            invalid_fx_rows=0,
            quarantined_rows=(),
        ),
    }
    with pytest.raises(ValueError, match="return values must be finite"):
        PDPObservationBatch(returns=((float("nan"), 0.0),), **kwargs)
    with pytest.raises(ValueError, match="equity_spot must be finite and positive"):
        PDPObservationBatch(returns=((0.0, 0.0),), **{**kwargs, "equity_spot": 0.0})


def test_multistart_selection_records_and_rejects_unconverged_fits() -> None:
    def fit(likelihood: float, increment: float, *, finite: bool = True) -> dict[str, object]:
        return {
            "marginal_log_likelihood": likelihood,
            "final_em_increment": increment,
            "minimum_em_increment": min(increment, 0.0),
            "finite": finite,
        }

    selected, diagnostics = _select_converged_multistart(
        [fit(100.0, 0.1), fit(90.0, 1.0e-5), fit(95.0, 2.0e-5)],
        [11, 12, 13],
    )
    assert selected["marginal_log_likelihood"] == 95.0
    assert [row["em_converged"] for row in diagnostics] == [False, True, True]
    assert [row["selected"] for row in diagnostics] == [False, False, True]
    assert [row["seed"] for row in diagnostics] == [11, 12, 13]
    with pytest.raises(RuntimeError, match="no finite converged"):
        _select_converged_multistart(
            [fit(100.0, 0.1), fit(90.0, float("nan")), fit(95.0, 0.0, finite=False)],
            [21, 22, 23],
        )


def test_one_state_composite_put_call_parity() -> None:
    common = {
        "equity_spot": 100.0,
        "fx_spot": 800.0,
        "equity_vol": 0.2,
        "fx_vol": 0.12,
        "correlation": -0.25,
        "domestic_rate": 0.045,
        "foreign_rate": 0.04,
        "dividend_yield": 0.01,
        "maturity": 0.5,
        "strike": 80_000.0,
    }
    call = one_state_price("composite_call", **common)
    put = one_state_price("composite_put", **common)
    expected = 80_000.0 * math.exp(-0.01 * 0.5) - 80_000.0 * math.exp(-0.045 * 0.5)
    assert call - put == pytest.approx(expected, rel=1.0e-12, abs=1.0e-10)
