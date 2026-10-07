"""American obstacle diagnostic acceptance without public report ownership."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from typing import Any

from finite_element_options.time_integration import LCPDiagnostics
from .benchmark_registry import ValidationGateError


def _has_exercise_front(exercise_set: Sequence[bool]) -> bool:
    """Return true only when exercise and continuation regions meet."""

    if not exercise_set:
        return False
    has_exercise = any(exercise_set)
    has_continuation = not all(exercise_set)
    has_transition = any(
        left != right for left, right in zip(exercise_set, exercise_set[1:])
    )
    return has_exercise and has_continuation and has_transition


def american_lcp_gate_values(
    benchmark_id: str, diagnostics: Sequence[LCPDiagnostics]
) -> dict[str, Any]:
    """Evaluate the unchanged diagnostic policy into public report values."""

    if not diagnostics:
        raise ValidationGateError(
            "American LCP gate requires at least one diagnostic row"
        )
    failures: list[str] = []
    rows: list[dict[str, Any]] = []
    exercise_front_observed = False
    max_complementarity = 0.0
    max_projected_residual = 0.0
    for index, item in enumerate(diagnostics):
        tolerance = item.tolerance
        diagnostic_values = {
            "tolerance": tolerance,
            "relaxation": item.relaxation,
            "primal_violation_max": item.primal_violation_max,
            "dual_violation_max": item.dual_violation_max,
            "complementarity_max": item.complementarity_max,
            "projected_residual_max": item.projected_residual_max,
            "max_update": item.max_update,
            "solve_time_sec": item.solve_time_sec,
        }
        nonfinite_fields = tuple(
            field
            for field, value in diagnostic_values.items()
            if not isfinite(float(value))
        )
        if nonfinite_fields:
            failures.append(
                f"row {index} has non-finite diagnostics {nonfinite_fields}"
            )
        if isfinite(float(tolerance)) and tolerance <= 0.0:
            failures.append(f"row {index} tolerance must be positive")
        if item.iterations < 0 or item.exercise_count < 0:
            failures.append(f"row {index} has invalid iteration/exercise counts")
        max_complementarity = max(max_complementarity, item.complementarity_max)
        max_projected_residual = max(
            max_projected_residual, item.projected_residual_max
        )
        row_has_front = _has_exercise_front(item.exercise_set)
        exercise_front_observed = exercise_front_observed or row_has_front
        if not item.success:
            failures.append(f"row {index} did not converge: {item.message}")
        if item.primal_violation_max > tolerance:
            failures.append(f"row {index} primal violation exceeds tolerance")
        if item.dual_violation_max > tolerance:
            failures.append(f"row {index} dual violation exceeds tolerance")
        if item.complementarity_max > tolerance:
            failures.append(f"row {index} complementarity exceeds tolerance")
        if item.projected_residual_max > tolerance:
            failures.append(f"row {index} projected residual exceeds tolerance")
        rows.append(
            {
                "row": index,
                "iterations": item.iterations,
                "tolerance": tolerance,
                "primal_violation_max": item.primal_violation_max,
                "dual_violation_max": item.dual_violation_max,
                "complementarity_max": item.complementarity_max,
                "projected_residual_max": item.projected_residual_max,
                "exercise_count": item.exercise_count,
                "exercise_front_observed": row_has_front,
            }
        )
    if not exercise_front_observed:
        failures.append("exercise front was not observed")
    report = dict(
        accepted=not failures,
        benchmark_id=benchmark_id,
        exercise_front_observed=exercise_front_observed,
        max_complementarity=max_complementarity,
        max_projected_residual=max_projected_residual,
        failures=tuple(failures),
        rows=tuple(rows),
    )
    return report
