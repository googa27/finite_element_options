"""Deterministic QuantLib/FEM/MC matrix evidence for one-regime reductions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import math
from typing import Any

from finite_element_options.examples.regime_switching_quanto import (
    ContractSpec,
    FEMGridSpec,
    TwoFactorRegimeModel,
    price_contract_fem,
    price_contract_monte_carlo,
)
from finite_element_options.examples.regime_switching_quanto._types import json_safe
from finite_element_options.examples.regime_switching_quanto.adoption.evidence_io import (
    canonical_json_sha256,
)

from .adapter import price_quantlib_oracle
from .contracts import QuantLibOracleSpec

MATRIX_SCHEMA_VERSION = "regime_quantlib_oracle_matrix.v1"
SCOPE_STATEMENT = (
    "This matrix validates only one-regime vanilla and fixed-FX quanto reductions; "
    "it is not evidence for the full multi-regime regime-switching PDE."
)


@dataclass(frozen=True)
class MatrixCase:
    """One public-synthetic oracle comparison case."""

    case_id: str
    spec: QuantLibOracleSpec
    grid: FEMGridSpec
    mc_paths: int
    mc_seed: int
    mc_steps_per_year: int = 252
    fem_abs_tolerance: float = 0.0
    mc_abs_floor: float = 0.0
    mc_standard_error_multiplier: float = 4.0

    def __post_init__(self) -> None:
        """Validate matrix execution controls before evidence generation."""

        if not str(self.case_id).strip():
            raise ValueError("case_id must be nonempty")
        if int(self.mc_paths) < 2:
            raise ValueError("mc_paths must be at least 2")
        if int(self.mc_steps_per_year) < 1:
            raise ValueError("mc_steps_per_year must be at least 1")
        _require_nonnegative_finite("fem_abs_tolerance", self.fem_abs_tolerance)
        _require_nonnegative_finite("mc_abs_floor", self.mc_abs_floor)
        _require_positive_finite(
            "mc_standard_error_multiplier",
            self.mc_standard_error_multiplier,
        )

    def to_input_dict(self) -> dict[str, Any]:
        """Return JSON-safe matrix input fields only."""

        payload = asdict(self)
        payload["spec"] = self.spec.to_dict()
        payload["grid"] = self.grid.to_dict()
        return json_safe(payload)


@dataclass(frozen=True)
class MatrixRunResult:
    """JSON-safe deterministic QuantLib oracle matrix artifact."""

    schema_version: str
    matrix_spec_hash: str
    scope: str
    cases: list[dict[str, Any]]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""

        return json_safe(asdict(self))


def default_matrix_cases() -> tuple[MatrixCase, ...]:
    """Return the canonical public-synthetic vanilla/quanto matrix."""

    return (
        MatrixCase(
            case_id="vanilla_atm_one_year",
            spec=QuantLibOracleSpec(
                evaluation_date=date(2026, 9, 4),
                maturity_date=date(2027, 9, 4),
                spot=100.0,
                strike=100.0,
                equity_vol=0.20,
                domestic_rate=0.030,
                foreign_rate=0.030,
                dividend_yield=0.010,
                fx_vol=0.0,
                correlation=0.0,
                fixed_fx=1.0,
                kind="vanilla",
            ),
            grid=FEMGridSpec((-1.45, 1.45), (-0.55, 0.55), nx=53, ny=9, time_steps=64),
            mc_paths=90_000,
            mc_seed=132_001,
            fem_abs_tolerance=0.050,
            mc_abs_floor=0.040,
        ),
        MatrixCase(
            case_id="vanilla_otm_six_month",
            spec=QuantLibOracleSpec(
                evaluation_date=date(2026, 10, 15),
                maturity_date=date(2027, 4, 15),
                spot=95.0,
                strike=105.0,
                equity_vol=0.25,
                domestic_rate=0.041,
                foreign_rate=0.041,
                dividend_yield=0.004,
                fx_vol=0.0,
                correlation=0.0,
                fixed_fx=1.0,
                kind="vanilla",
            ),
            grid=FEMGridSpec((-1.45, 1.45), (-0.55, 0.55), nx=53, ny=9, time_steps=56),
            mc_paths=90_000,
            mc_seed=132_002,
            fem_abs_tolerance=0.030,
            mc_abs_floor=0.040,
        ),
        MatrixCase(
            case_id="quanto_positive_correlation",
            spec=QuantLibOracleSpec(
                evaluation_date=date(2026, 9, 4),
                maturity_date=date(2027, 12, 6),
                spot=100.0,
                strike=105.0,
                equity_vol=0.20,
                domestic_rate=0.035,
                foreign_rate=0.015,
                dividend_yield=0.010,
                fx_vol=0.12,
                correlation=0.35,
                fixed_fx=850.0,
                kind="fixed_fx_quanto",
            ),
            grid=FEMGridSpec((-1.60, 1.60), (-0.70, 0.70), nx=61, ny=9, time_steps=80),
            mc_paths=110_000,
            mc_seed=132_003,
            fem_abs_tolerance=7.0,
            mc_abs_floor=10.0,
        ),
        MatrixCase(
            case_id="quanto_negative_correlation",
            spec=QuantLibOracleSpec(
                evaluation_date=date(2026, 11, 2),
                maturity_date=date(2027, 8, 2),
                spot=120.0,
                strike=110.0,
                equity_vol=0.28,
                domestic_rate=0.025,
                foreign_rate=0.045,
                dividend_yield=0.020,
                fx_vol=0.18,
                correlation=-0.45,
                fixed_fx=780.0,
                kind="fixed_fx_quanto",
            ),
            grid=FEMGridSpec((-1.55, 1.55), (-0.70, 0.70), nx=61, ny=9, time_steps=72),
            mc_paths=110_000,
            mc_seed=132_004,
            fem_abs_tolerance=15.0,
            mc_abs_floor=14.0,
        ),
    )


def run_quantlib_oracle_matrix(cases: tuple[MatrixCase, ...] | None = None) -> MatrixRunResult:
    """Run the deterministic one-regime QuantLib/FEM/MC matrix."""

    selected = default_matrix_cases() if cases is None else cases
    if not selected:
        raise ValueError("QuantLib oracle matrix requires at least one case")
    matrix_inputs = _matrix_inputs(selected)
    matrix_spec_hash = canonical_json_sha256(matrix_inputs)
    rows = [_run_case(case) for case in selected]
    all_passed = all(row["gates"]["all_passed"] for row in rows)
    max_quantlib_error = max(row["errors"]["quantlib_vs_analytical_abs"] for row in rows)
    max_fem_error = max(row["errors"]["fem_vs_analytical_abs"] for row in rows)
    max_mc_z = max(row["errors"]["mc_vs_analytical_standard_errors"] for row in rows)
    return MatrixRunResult(
        schema_version=MATRIX_SCHEMA_VERSION,
        matrix_spec_hash=matrix_spec_hash,
        scope=SCOPE_STATEMENT,
        cases=rows,
        summary={
            "case_count": len(rows),
            "vanilla_case_count": sum(1 for row in rows if row["spec"]["kind"] == "vanilla"),
            "quanto_case_count": sum(1 for row in rows if row["spec"]["kind"] == "fixed_fx_quanto"),
            "all_passed": all_passed,
            "max_quantlib_vs_analytical_abs": max_quantlib_error,
            "max_fem_vs_analytical_abs": max_fem_error,
            "max_mc_vs_analytical_standard_errors": max_mc_z,
            "scope_limitation": SCOPE_STATEMENT,
        },
    )


def canonical_matrix_input_hash() -> str:
    """Return the hash of the canonical matrix inputs without executing pricers."""

    return canonical_json_sha256(_matrix_inputs(default_matrix_cases()))


def _matrix_inputs(cases: tuple[MatrixCase, ...]) -> dict[str, Any]:
    return {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "scope": SCOPE_STATEMENT,
        "cases": [case.to_input_dict() for case in cases],
    }


def _require_nonnegative_finite(field: str, value: float) -> None:
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")


def _require_positive_finite(field: str, value: float) -> None:
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{field} must be finite and positive")


def _run_case(case: MatrixCase) -> dict[str, Any]:
    quantlib = price_quantlib_oracle(case.spec)
    maturity = quantlib.year_fraction
    model = _one_regime_model(case.spec)
    contract = ContractSpec(
        kind="quanto_call",
        strike=float(case.spec.strike),
        fixed_fx=float(case.spec.fixed_fx),
    )
    fem = price_contract_fem(
        model,
        contract,
        maturity=maturity,
        equity_spot=float(case.spec.spot),
        fx_spot=1.0,
        grid=case.grid,
    )
    mc = price_contract_monte_carlo(
        model,
        contract,
        maturity=maturity,
        equity_spot=float(case.spec.spot),
        fx_spot=1.0,
        paths=case.mc_paths,
        seed=case.mc_seed,
        steps_per_year=case.mc_steps_per_year,
    )
    analytical = quantlib.analytical_price
    fem_error = abs(fem.mixture_price - analytical)
    mc_error = abs(mc.price - analytical)
    mc_tolerance = max(case.mc_abs_floor, case.mc_standard_error_multiplier * mc.standard_error)
    quantlib_error = quantlib.analytical_absolute_error
    gates = {
        "quantlib_analytical_passed": quantlib.analytical_passed,
        "fem_passed": fem_error <= case.fem_abs_tolerance,
        "mc_passed": mc_error <= mc_tolerance,
    }
    gates["all_passed"] = all(gates.values())
    return {
        "case_id": case.case_id,
        "spec": case.spec.to_dict(),
        "grid": case.grid.to_dict(),
        "conventions": quantlib.conventions,
        "model_reduction": _model_reduction(case.spec),
        "prices": {
            "analytical": analytical,
            "quantlib": quantlib.price,
            "quantlib_unscaled_npv": quantlib.quantlib_npv,
            "fem": fem.mixture_price,
            "mc": mc.price,
        },
        "errors": {
            "quantlib_vs_analytical_abs": quantlib_error,
            "fem_vs_analytical_abs": fem_error,
            "mc_vs_analytical_abs": mc_error,
            "mc_vs_analytical_standard_errors": mc_error / mc.standard_error,
        },
        "tolerances": {
            "quantlib_vs_analytical_abs": quantlib.analytical_tolerance,
            "fem_vs_analytical_abs": case.fem_abs_tolerance,
            "mc_vs_analytical_abs": mc_tolerance,
            "mc_standard_error_multiplier": case.mc_standard_error_multiplier,
        },
        "quantlib_result": quantlib.to_dict(),
        "fem_result": fem.to_dict(),
        "mc_result": {
            **mc.to_dict(),
            "seed": case.mc_seed,
            "steps_per_year": case.mc_steps_per_year,
        },
        "gates": gates,
    }


def _one_regime_model(spec: QuantLibOracleSpec) -> TwoFactorRegimeModel:
    foreign_rate = spec.domestic_rate if spec.kind == "vanilla" else spec.foreign_rate
    fx_vol = 0.0 if spec.kind == "vanilla" else spec.fx_vol
    correlation = 0.0 if spec.kind == "vanilla" else spec.correlation
    return TwoFactorRegimeModel(
        equity_vol=[float(spec.equity_vol)],
        fx_vol=[float(fx_vol)],
        correlation=[float(correlation)],
        generator=[[0.0]],
        current_probabilities=[1.0],
        domestic_rate=float(spec.domestic_rate),
        foreign_rate=float(foreign_rate),
        dividend_yield=float(spec.dividend_yield),
        measure_note=(
            "One-regime deterministic QuantLib oracle reduction: equity drift is "
            "rf - q - rho*sigmaS*sigmaFX; vanilla sets rf=rd, fx_vol=0, rho=0."
        ),
    )


def _model_reduction(spec: QuantLibOracleSpec) -> dict[str, Any]:
    return {
        "equity_drift_formula": "rf - q - rho*sigmaS*sigmaFX",
        "effective_dividend_yield_formula": "q + rd - rf + rho*sigmaS*sigmaFX",
        "quanto_adjustment": spec.quanto_adjustment,
        "effective_dividend_yield": spec.effective_dividend_yield,
        "fixed_fx_scaling": spec.fixed_fx if spec.kind == "fixed_fx_quanto" else 1.0,
        "scope": SCOPE_STATEMENT,
    }


__all__ = [
    "MATRIX_SCHEMA_VERSION",
    "SCOPE_STATEMENT",
    "MatrixCase",
    "MatrixRunResult",
    "canonical_matrix_input_hash",
    "default_matrix_cases",
    "run_quantlib_oracle_matrix",
]
