"""Validation fixtures and executable parity evidence."""

from .black_scholes_parity import (
    DEFAULT_TOLERANCE_ABSOLUTE,
    DEFAULT_TOLERANCE_RELATIVE,
    EXPECTED_BLACK_SCHOLES_CALL_PRICE,
    PUBLIC_SYNTHETIC_BLACK_SCHOLES_BENCHMARK_ID,
    PUBLIC_SYNTHETIC_PROBLEM_ID,
    FEMParityConvergenceRow,
    FEMParityReport,
    run_public_black_scholes_parity_fixture,
)

__all__ = [
    "DEFAULT_TOLERANCE_ABSOLUTE",
    "DEFAULT_TOLERANCE_RELATIVE",
    "EXPECTED_BLACK_SCHOLES_CALL_PRICE",
    "PUBLIC_SYNTHETIC_BLACK_SCHOLES_BENCHMARK_ID",
    "PUBLIC_SYNTHETIC_PROBLEM_ID",
    "FEMParityConvergenceRow",
    "FEMParityReport",
    "run_public_black_scholes_parity_fixture",
]
