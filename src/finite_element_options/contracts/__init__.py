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
from .formula_bundle import (
    finite_element_formula_bundle,
    formula_bundle_json,
    validate_formula_bundle,
)

__all__ = [
    "CapabilityStatus",
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
    "validate_formula_bundle",
]
