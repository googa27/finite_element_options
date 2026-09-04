"""Canonical public-synthetic cases and artifact builder."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

from finite_element_options.examples.regime_switching_quanto._types import json_safe
from finite_element_options.examples.regime_switching_quanto.adoption.evidence_io import (
    canonical_json_sha256,
)

from .adapter import run_iminuit_identifiability
from .contracts import (
    CalibrationCase,
    ParameterBounds,
    ProfileGrid,
    QuantoOptionTarget,
    SCHEMA_VERSION,
)

STUDY_SCHEMA_VERSION = "regime_iminuit_identifiability_study.v1"
SCOPE_STATEMENT = (
    "Public-synthetic fixed-FX quanto instrument-target calibration evidence only; "
    "these are not observed/live market prices, not production calibration, and not "
    "a maturity promotion for regime-switching PDE pricing."
)
FORMULA_STATEMENT = (
    "chi2=sum_i((BS_quanto_price(theta; instrument_i)-target_price_i)/price_std_i)^2; "
    "q_eff=q+rd-rf+rho*equity_vol*fx_vol; "
    "year_fraction=(maturity_date-evaluation_date).days/365; Minuit.errordef=1."
)


@dataclass(frozen=True)
class IdentifiabilityStudyResult:
    """JSON-safe two-case iminuit profile-likelihood evidence artifact."""

    schema_version: str
    run_schema_version: str
    study_input_hash: str
    scope: str
    formula: str
    cases: list[dict[str, Any]]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe study artifact."""

        return json_safe(asdict(self))


def default_identifiability_cases() -> tuple[CalibrationCase, ...]:
    """Return canonical identified and weak-rho public-synthetic cases."""

    return (
        _identified_case(),
        _weak_rho_case(),
    )


def canonical_identifiability_input_hash() -> str:
    """Return the canonical two-case input hash without running iminuit."""

    return canonical_json_sha256(_study_inputs(default_identifiability_cases()))


def run_iminuit_identifiability_study(
    cases: tuple[CalibrationCase, ...] | None = None,
) -> IdentifiabilityStudyResult:
    """Run the canonical identifiability study and summarize gate decisions."""

    selected = default_identifiability_cases() if cases is None else cases
    if not selected:
        raise ValueError("identifiability study requires at least one case")
    inputs = _study_inputs(selected)
    study_input_hash = canonical_json_sha256(inputs)
    runs = [run_iminuit_identifiability(case) for case in selected]
    rows = [run.to_dict() for run in runs]
    decisions = {row["case"]["case_id"]: row["identification"]["identified"] for row in rows}
    summary = {
        "case_count": len(rows),
        "identified_case_count": sum(1 for row in rows if row["identification"]["identified"]),
        "all_expected_decisions_passed": decisions.get("identified_quanto_surface") is True
        and decisions.get("weak_rho_fxvol_zero") is False,
        "decisions": decisions,
        "case_input_hashes": {row["case"]["case_id"]: row["case_input_hash"] for row in rows},
        "scope_limitation": SCOPE_STATEMENT,
    }
    return IdentifiabilityStudyResult(
        schema_version=STUDY_SCHEMA_VERSION,
        run_schema_version=SCHEMA_VERSION,
        study_input_hash=study_input_hash,
        scope=SCOPE_STATEMENT,
        formula=FORMULA_STATEMENT,
        cases=rows,
        summary=summary,
    )


def _study_inputs(cases: tuple[CalibrationCase, ...]) -> dict[str, Any]:
    return {
        "schema_version": STUDY_SCHEMA_VERSION,
        "scope": SCOPE_STATEMENT,
        "formula": FORMULA_STATEMENT,
        "cases": [case.to_input_dict() for case in cases],
    }


def _bounds() -> ParameterBounds:
    return ParameterBounds(equity_vol=(0.05, 0.60), correlation=(-0.95, 0.95))


def _identified_case() -> CalibrationCase:
    evaluation = date(2026, 9, 4)
    targets = (
        QuantoOptionTarget(
            "id_atm_6m",
            evaluation,
            date(2027, 3, 5),
            100.0,
            100.0,
            800.0,
            0.032,
            0.014,
            0.011,
            0.16,
            5475.320235574975,
            12.0,
        ),
        QuantoOptionTarget(
            "id_otm_9m",
            evaluation,
            date(2027, 6, 4),
            100.0,
            108.0,
            800.0,
            0.034,
            0.016,
            0.010,
            0.12,
            4178.882708616800,
            12.0,
        ),
        QuantoOptionTarget(
            "id_itm_1y",
            evaluation,
            date(2027, 9, 4),
            105.0,
            95.0,
            820.0,
            0.031,
            0.012,
            0.013,
            0.18,
            12870.863745302224,
            14.0,
        ),
        QuantoOptionTarget(
            "id_otm_15m",
            evaluation,
            date(2027, 12, 6),
            98.0,
            112.0,
            790.0,
            0.036,
            0.018,
            0.009,
            0.20,
            4868.237421490577,
            14.0,
        ),
        QuantoOptionTarget(
            "id_atm_18m",
            evaluation,
            date(2028, 3, 6),
            102.0,
            102.0,
            810.0,
            0.033,
            0.015,
            0.012,
            0.14,
            9906.529677317149,
            16.0,
        ),
        QuantoOptionTarget(
            "id_itm_2y",
            evaluation,
            date(2028, 9, 5),
            110.0,
            100.0,
            805.0,
            0.035,
            0.017,
            0.010,
            0.19,
            17292.039751389089,
            16.0,
        ),
        QuantoOptionTarget(
            "id_low_strike_1y",
            evaluation,
            date(2027, 9, 4),
            94.0,
            90.0,
            795.0,
            0.030,
            0.012,
            0.008,
            0.15,
            8982.599247734919,
            14.0,
        ),
        QuantoOptionTarget(
            "id_high_strike_2y",
            evaluation,
            date(2028, 9, 5),
            100.0,
            120.0,
            815.0,
            0.037,
            0.019,
            0.011,
            0.17,
            6196.896680271982,
            16.0,
        ),
    )
    return CalibrationCase(
        case_id="identified_quanto_surface",
        description="Several fixed-FX quanto calls spanning strikes/maturities with nonzero FX vol.",
        targets=targets,
        initial_equity_vol=0.20,
        initial_correlation=0.0,
        bounds=_bounds(),
        edm_threshold=1.0e-5,
        bound_contact_tolerance=1.0e-4,
        minos_cl=0.682689492137,
        minos_ncall=2000,
        profile_grids=(
            ProfileGrid("equity_vol", 0.18, 0.28, 31),
            ProfileGrid("correlation", -0.80, 0.05, 35),
        ),
        finite_difference_step=1.0e-6,
        synthetic_truth={"equity_vol": 0.23, "correlation": -0.40},
    )


def _weak_rho_case() -> CalibrationCase:
    evaluation = date(2026, 9, 4)
    targets = (
        QuantoOptionTarget(
            "weak_atm_6m",
            evaluation,
            date(2027, 3, 5),
            100.0,
            100.0,
            800.0,
            0.032,
            0.014,
            0.011,
            0.0,
            5158.739024711769,
            12.0,
        ),
        QuantoOptionTarget(
            "weak_otm_9m",
            evaluation,
            date(2027, 6, 4),
            100.0,
            108.0,
            800.0,
            0.034,
            0.016,
            0.010,
            0.0,
            3916.727401012559,
            12.0,
        ),
        QuantoOptionTarget(
            "weak_itm_1y",
            evaluation,
            date(2027, 9, 4),
            105.0,
            95.0,
            820.0,
            0.031,
            0.012,
            0.013,
            0.0,
            11868.830655343836,
            14.0,
        ),
        QuantoOptionTarget(
            "weak_otm_15m",
            evaluation,
            date(2027, 12, 6),
            98.0,
            112.0,
            790.0,
            0.036,
            0.018,
            0.009,
            0.0,
            4200.696850359959,
            14.0,
        ),
        QuantoOptionTarget(
            "weak_atm_18m",
            evaluation,
            date(2028, 3, 6),
            102.0,
            102.0,
            810.0,
            0.033,
            0.015,
            0.012,
            0.0,
            9016.808239335764,
            16.0,
        ),
        QuantoOptionTarget(
            "weak_itm_2y",
            evaluation,
            date(2028, 9, 5),
            110.0,
            100.0,
            805.0,
            0.035,
            0.017,
            0.010,
            0.0,
            15172.431641138088,
            16.0,
        ),
    )
    return CalibrationCase(
        case_id="weak_rho_fxvol_zero",
        description="All FX volatilities are zero, so rho is structurally absent from q_eff.",
        targets=targets,
        initial_equity_vol=0.20,
        initial_correlation=0.25,
        bounds=_bounds(),
        edm_threshold=1.0e-5,
        bound_contact_tolerance=1.0e-4,
        minos_cl=0.682689492137,
        minos_ncall=2000,
        profile_grids=(
            ProfileGrid("equity_vol", 0.18, 0.28, 31),
            ProfileGrid("correlation", -0.90, 0.90, 31),
        ),
        finite_difference_step=1.0e-6,
        synthetic_truth={"equity_vol": 0.23, "correlation": 0.25},
    )


__all__ = [
    "FORMULA_STATEMENT",
    "SCOPE_STATEMENT",
    "STUDY_SCHEMA_VERSION",
    "IdentifiabilityStudyResult",
    "canonical_identifiability_input_hash",
    "default_identifiability_cases",
    "run_iminuit_identifiability_study",
]
