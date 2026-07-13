"""FPF-facing solver-result evidence boundary declarations.

These records are contract metadata only: finite_element_options advertises the
public fields it emits for downstream Financial Problem Formulations consumers
without importing FPF, PDP, UI, or product-private runtime modules.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

FPF_SOLVER_RESULT_EVIDENCE_REQUIRED_FIELDS: Final[tuple[str, ...]] = (
    "problem_id",
    "problem_hash",
    "status",
    "measure",
    "numeraire",
    "units",
    "backend_capability_status",
    "diagnostics",
)


def fpf_solver_result_evidence_contract() -> dict[str, object]:
    """Return JSON-safe FPF result-evidence metadata for public FEM contracts."""

    return {
        "contract_family": "FPF.solver_result_evidence.v1",
        "required_fields": list(FPF_SOLVER_RESULT_EVIDENCE_REQUIRED_FIELDS),
    }


def validate_fpf_solver_result_evidence_payload(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Return schema-contract violations for an FPF solver result evidence payload."""

    violations: list[str] = []
    for field_name in FPF_SOLVER_RESULT_EVIDENCE_REQUIRED_FIELDS:
        if field_name not in payload:
            violations.append(f"missing required field: {field_name}")
            continue
        value = payload[field_name]
        if value is None or value == "" or value == {} or value == []:
            violations.append(f"empty required field: {field_name}")

    if "units" in payload and not isinstance(payload["units"], Mapping):
        violations.append("units must be a mapping")
    if "diagnostics" in payload and not isinstance(payload["diagnostics"], Mapping):
        violations.append("diagnostics must be a mapping")
    if "backend_capability_status" in payload and not isinstance(
        payload["backend_capability_status"], Mapping
    ):
        violations.append("backend_capability_status must be a mapping")
    if "status" in payload and not isinstance(payload["status"], str):
        violations.append("status must be a string")
    return tuple(violations)


__all__ = [
    "FPF_SOLVER_RESULT_EVIDENCE_REQUIRED_FIELDS",
    "fpf_solver_result_evidence_contract",
    "validate_fpf_solver_result_evidence_payload",
]
