"""Maintained QuantLib pricing adapter for vanilla and fixed-FX quanto calls."""

from __future__ import annotations

from datetime import date
from types import ModuleType
from typing import Any

from finite_element_options.core.market import Market
from finite_element_options.core.vanilla_bs import EuropeanOptionBs
from finite_element_options.examples.regime_switching_quanto.adoption.quantlib_state import (
    quantlib_evaluation_date,
)

from .contracts import QuantLibOracleResult, QuantLibOracleSpec, QuantLibReductionError

SCHEMA_VERSION = "regime_quantlib_oracle.v1"
ANALYTICAL_TOLERANCE = 1.0e-9


def price_quantlib_oracle(spec: QuantLibOracleSpec) -> QuantLibOracleResult:
    """Price ``spec`` with QuantLib and compare to the repository BS oracle."""

    # Re-run pure-Python validation before this call can mutate QuantLib's
    # process-global evaluation date, including manually altered dataclass values.
    spec.__post_init__()
    with quantlib_evaluation_date(spec.evaluation_date) as quantlib:
        ql_eval = _ql_date(quantlib, spec.evaluation_date)
        explicit_maturity = _ql_date(quantlib, spec.maturity_date)
        calendar = quantlib.TARGET()
        ql_maturity = calendar.adjust(explicit_maturity, quantlib.Unadjusted)
        _require_matching_quantlib_date(
            field="business_day_convention_result",
            received=ql_maturity,
            expected=explicit_maturity,
            kind=spec.kind,
        )
        day_count = quantlib.Actual365Fixed()
        year_fraction = float(day_count.yearFraction(ql_eval, ql_maturity))
        process = _bsm_process(quantlib, spec, ql_eval, calendar, day_count)
        payoff = quantlib.PlainVanillaPayoff(quantlib.Option.Call, float(spec.strike))
        exercise = quantlib.EuropeanExercise(ql_maturity)
        exercise_dates = list(exercise.dates())
        received_exercise_date = exercise_dates[0] if len(exercise_dates) == 1 else exercise_dates
        _require_matching_quantlib_date(
            field="exercise_date",
            received=received_exercise_date,
            expected=explicit_maturity,
            kind=spec.kind,
        )
        if spec.kind == "vanilla":
            option = quantlib.EuropeanOption(payoff, exercise)
            option.setPricingEngine(quantlib.AnalyticEuropeanEngine(process))
            quantlib_npv = float(option.NPV())
            price = quantlib_npv
        else:
            option = quantlib.QuantoVanillaOption(payoff, exercise)
            option.setPricingEngine(
                quantlib.QuantoEuropeanEngine(
                    process,
                    _flat_rate_handle(quantlib, ql_eval, day_count, spec.foreign_rate),
                    _flat_vol_handle(quantlib, ql_eval, calendar, day_count, spec.fx_vol),
                    quantlib.QuoteHandle(quantlib.SimpleQuote(float(spec.correlation))),
                )
            )
            quantlib_npv = float(option.NPV())
            price = quantlib_npv * float(spec.fixed_fx)

    analytical = analytical_oracle_price(spec, year_fraction=year_fraction)
    error = abs(price - analytical)
    return QuantLibOracleResult(
        schema_version=SCHEMA_VERSION,
        spec=spec.to_dict(),
        price=price,
        quantlib_npv=quantlib_npv,
        analytical_price=analytical,
        analytical_absolute_error=error,
        analytical_tolerance=ANALYTICAL_TOLERANCE,
        analytical_passed=error <= ANALYTICAL_TOLERANCE,
        year_fraction=year_fraction,
        quanto_adjustment=spec.quanto_adjustment,
        effective_dividend_yield=spec.effective_dividend_yield,
        quantlib_version=str(getattr(quantlib, "__version__", "unknown")),
        conventions={
            "calendar": spec.calendar,
            "day_count": spec.day_count,
            "business_day_convention": spec.business_day_convention,
            "rate_compounding": spec.rate_compounding,
            "exercise": spec.exercise,
            "option_type": spec.option_type,
        },
    )


def analytical_oracle_price(spec: QuantLibOracleSpec, *, year_fraction: float) -> float:
    """Return the repository analytical BS reduction for ``spec``."""

    option = EuropeanOptionBs(
        k=float(spec.strike),
        q=spec.effective_dividend_yield,
        mkt=Market(r=float(spec.domestic_rate)),
    )
    raw = float(
        option.call_from_volatility(year_fraction, float(spec.spot), float(spec.equity_vol))
    )
    return raw if spec.kind == "vanilla" else raw * float(spec.fixed_fx)


def _require_matching_quantlib_date(*, field: str, received: Any, expected: Any, kind: str) -> None:
    """Raise a JSON-safe typed error when QuantLib changes an explicit date."""

    if received != expected:
        raise QuantLibReductionError(
            field=field,
            received=str(received),
            expected=str(expected),
            kind=kind,
        )


def _bsm_process(
    quantlib: ModuleType,
    spec: QuantLibOracleSpec,
    evaluation_date: Any,
    calendar: Any,
    day_count: Any,
) -> Any:
    return quantlib.BlackScholesMertonProcess(
        quantlib.QuoteHandle(quantlib.SimpleQuote(float(spec.spot))),
        _flat_rate_handle(quantlib, evaluation_date, day_count, spec.dividend_yield),
        _flat_rate_handle(quantlib, evaluation_date, day_count, spec.domestic_rate),
        _flat_vol_handle(quantlib, evaluation_date, calendar, day_count, spec.equity_vol),
    )


def _flat_rate_handle(
    quantlib: ModuleType, evaluation_date: Any, day_count: Any, rate: float
) -> Any:
    return quantlib.YieldTermStructureHandle(
        quantlib.FlatForward(
            evaluation_date,
            float(rate),
            day_count,
            quantlib.Continuous,
            quantlib.Annual,
        )
    )


def _flat_vol_handle(
    quantlib: ModuleType,
    evaluation_date: Any,
    calendar: Any,
    day_count: Any,
    volatility: float,
) -> Any:
    return quantlib.BlackVolTermStructureHandle(
        quantlib.BlackConstantVol(evaluation_date, calendar, float(volatility), day_count)
    )


def _ql_date(quantlib: ModuleType, value: date) -> Any:
    return quantlib.Date(value.day, value.month, value.year)


__all__ = [
    "ANALYTICAL_TOLERANCE",
    "SCHEMA_VERSION",
    "analytical_oracle_price",
    "price_quantlib_oracle",
]
