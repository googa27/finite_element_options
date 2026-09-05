"""Internal QuantLib process-global state adapter for adoption research."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from threading import RLock
from types import ModuleType
from typing import Any

from finite_element_options.examples.regime_switching_quanto.adoption.optional import (
    require_optional,
)

_EVALUATION_DATE_LOCK = RLock()


@contextmanager
def quantlib_evaluation_date(evaluation_date: Any) -> Iterator[ModuleType]:
    """Temporarily set QuantLib's process-global evaluation date.

    This is an adapter-only internal research surface for the regime-switching
    quanto adoption package.  QuantLib stores ``Settings.instance().evaluationDate``
    as process-global mutable state, so this context manager serializes callers
    with a re-entrant lock, restores the previous value in ``finally`` on success
    and failure, and yields the lazily imported module only inside the boundary.
    Public result and domain contracts must not expose QuantLib objects.
    """

    with _EVALUATION_DATE_LOCK:
        quantlib = require_optional("QuantLib")
        settings = quantlib.Settings.instance()
        previous_date = settings.evaluationDate
        converted_date = _to_quantlib_date(quantlib, evaluation_date)
        settings.evaluationDate = converted_date
        try:
            yield quantlib
        finally:
            settings.evaluationDate = previous_date


def _to_quantlib_date(quantlib: ModuleType, evaluation_date: Any) -> Any:
    """Convert stdlib dates after lazy QuantLib import; preserve sentinels."""

    if isinstance(evaluation_date, date):
        return quantlib.Date(evaluation_date.day, evaluation_date.month, evaluation_date.year)
    return evaluation_date
