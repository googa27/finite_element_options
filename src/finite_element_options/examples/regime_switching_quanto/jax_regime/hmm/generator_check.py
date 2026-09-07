"""Executable CTMC-generator law check for fitted daily transition matrices."""

from __future__ import annotations

from typing import Any

import numpy as np

from finite_element_options.examples.regime_switching_quanto.generator import (
    TRADING_DAYS,
    discrete_to_continuous_generator,
)


def check_ctmc_generator(transition: Any) -> dict[str, Any]:
    """Project a daily transition matrix and verify the generator cone laws."""

    generator, reconstruction_residual = discrete_to_continuous_generator(
        np.asarray(transition, dtype=float), periods_per_year=TRADING_DAYS
    )
    off_diagonal = generator.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    minimum_off_diagonal = float(np.min(off_diagonal))
    maximum_diagonal = float(np.max(np.diag(generator)))
    maximum_row_sum_error = float(np.max(np.abs(generator.sum(axis=1))))
    finite = bool(np.isfinite(generator).all() and np.isfinite(reconstruction_residual))
    passed = (
        finite
        and minimum_off_diagonal >= -1.0e-12
        and maximum_diagonal <= 1.0e-12
        and maximum_row_sum_error <= 1.0e-10
        and reconstruction_residual <= 5.0e-2
    )
    return {
        "periods_per_year": TRADING_DAYS,
        "generator": generator.tolist(),
        "minimum_off_diagonal": minimum_off_diagonal,
        "maximum_diagonal": maximum_diagonal,
        "maximum_row_sum_error": maximum_row_sum_error,
        "daily_transition_reconstruction_residual": reconstruction_residual,
        "finite": finite,
        "passed": passed,
    }
