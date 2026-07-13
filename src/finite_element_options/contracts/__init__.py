"""Finite-element backend contracts and diagnostics."""

from .backend_capabilities import (
    CapabilityStatus,
    DEFAULT_FEM_CAPABILITY_MANIFEST,
    DEFAULT_LINEAR_SOLVER_CAPABILITIES,
    DEFAULT_RELEASED_FEM_SOLVER_CONTRACT,
    FEMCapabilityManifest,
    FEMRouteRequest,
    FEMSolverContract,
    SolverBackendCapability,
    UnsupportedReason,
    UnsupportedRouteDiagnostic,
    UnsupportedRouteError,
    diagnose_unsupported_route,
    ensure_route_supported,
)
from .capability_matrix import (
    DEFAULT_CAPABILITY_RECORDS,
    CapabilityRecord,
    CapabilityStatus as CapabilityMaturityStatus,
    public_capability_records,
)
from .formula_bundle import (
    finite_element_formula_bundle,
    formula_bundle_json,
    validate_formula_bundle,
)
from .fpf_evidence import (
    fpf_solver_result_evidence_contract,
    validate_fpf_solver_result_evidence_payload,
)

__all__ = [
    "CapabilityStatus",
    "CapabilityMaturityStatus",
    "DEFAULT_CAPABILITY_RECORDS",
    "CapabilityRecord",
    "DEFAULT_FEM_CAPABILITY_MANIFEST",
    "DEFAULT_LINEAR_SOLVER_CAPABILITIES",
    "DEFAULT_RELEASED_FEM_SOLVER_CONTRACT",
    "FEMCapabilityManifest",
    "FEMRouteRequest",
    "FEMSolverContract",
    "SolverBackendCapability",
    "UnsupportedReason",
    "UnsupportedRouteDiagnostic",
    "UnsupportedRouteError",
    "diagnose_unsupported_route",
    "ensure_route_supported",
    "finite_element_formula_bundle",
    "formula_bundle_json",
    "fpf_solver_result_evidence_contract",
    "public_capability_records",
    "validate_formula_bundle",
    "validate_fpf_solver_result_evidence_payload",
]
