"""Dependency-light contract and data-loader tests for the JAX regime study."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from finite_element_options.examples.regime_switching_quanto.jax_regime.contracts import (
    JaxRegimeStudyConfig,
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
    _required_divergence_count,
)
from finite_element_options.examples.regime_switching_quanto.jax_regime.pricing.analytic import (
    one_state_price,
)


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
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as bundle:
        bundle.writestr(f"{_MEMBER_ROOT}pdp_joint_levels.csv", levels)
        bundle.writestr(f"{_MEMBER_ROOT}input_provenance.json", provenance)
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


def test_missing_numpyro_divergence_telemetry_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="omitted required 'diverging'"):
        _required_divergence_count({})


def test_resource_limits_and_finite_observations_fail_closed() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        JaxRegimeStudyConfig(pricing_paths=1_000_001)
    with pytest.raises(ValueError, match="at least 2"):
        JaxRegimeStudyConfig(chains=1)
    with pytest.raises(ValueError, match="steps_per_year must be 252"):
        JaxRegimeStudyConfig(steps_per_year=504)
    with pytest.raises(ValueError, match="at least one daily pricing step"):
        JaxRegimeStudyConfig(maturity_years=0.001)
    with pytest.raises(ValueError, match="finite and positive"):
        JaxRegimeStudyConfig(maturity_years=float("nan"))
    with pytest.raises(ValueError, match="more than 3,660 pricing steps"):
        JaxRegimeStudyConfig(maturity_years=15.0)
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
