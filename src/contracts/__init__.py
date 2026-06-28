"""Finite-element backend contracts and diagnostics."""

from .backend_capabilities import (
    CapabilityStatus,
    DEFAULT_FEM_CAPABILITY_MANIFEST,
    FEMCapabilityManifest,
    FEMRouteRequest,
    UnsupportedReason,
    UnsupportedRouteDiagnostic,
    UnsupportedRouteError,
    diagnose_unsupported_route,
    ensure_route_supported,
)

__all__ = [
    "CapabilityStatus",
    "DEFAULT_FEM_CAPABILITY_MANIFEST",
    "FEMCapabilityManifest",
    "FEMRouteRequest",
    "UnsupportedReason",
    "UnsupportedRouteDiagnostic",
    "UnsupportedRouteError",
    "diagnose_unsupported_route",
    "ensure_route_supported",
]
