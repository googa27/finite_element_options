"""Adapter from FEM capabilities to Haircut's public solver protocol.

Haircut owns the generic identity, capability, and discovery contracts.  This
module imports only that public seam at factory time and fails closed when the
released Haircut package is unavailable or contract majors drift.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from importlib import metadata
from typing import Any

from finite_element_options.contracts import FEMCapabilityManifest

HAIRCUT_BACKEND_ENTRY_POINT_GROUP = "haircut.solver_backends"
HAIRCUT_BACKEND_ENTRY_POINT = (
    "finite_element_options.integrations.haircut_backend:create_backend"
)
HAIRCUT_PUBLIC_CONTRACT_VERSION = "0.1.0"
_DISTRIBUTION_NAME = "finite-element-options"
_ISSUE_REFS = (
    "googa27/finite_element_options#115",
    "googa27/haircut-engine#217",
)


class HaircutProtocolUnavailableError(RuntimeError):
    """Raised when Haircut's public solver protocol seam is not installed."""


class ContractMajorMismatchError(RuntimeError):
    """Raised when FEM and Haircut backend contract majors are incompatible."""


@dataclass(frozen=True)
class HaircutContracts:
    """Haircut-owned identity and capability objects for one FEM plugin."""

    identity: Any
    capability_manifest: Any


@dataclass(frozen=True)
class _HaircutPublicSolverSeam:
    ContractVersion: Any
    BackendMaturity: Any
    BackendIdentity: Any
    MethodMaturity: Any
    MethodCapability: Any
    BackendCapabilityManifest: Any


def build_haircut_contracts(
    manifest: FEMCapabilityManifest,
    *,
    expected_contract_version: str = HAIRCUT_PUBLIC_CONTRACT_VERSION,
) -> HaircutContracts:
    """Construct Haircut-owned protocol values from an evidenced FEM manifest."""

    seam = _load_public_solver_seam()
    _ensure_contract_major_compatible(
        seam, manifest.contract_version, expected_contract_version
    )
    distribution_version = _distribution_version()
    identity = seam.BackendIdentity(
        distribution_name=_DISTRIBUTION_NAME,
        distribution_version=distribution_version,
        implementation_id=manifest.backend_id,
        implementation_version=distribution_version,
        contract_version=seam.ContractVersion.parse(manifest.contract_version),
        maturity=seam.BackendMaturity.VALIDATED,
        license_identifier="MIT",
        build_metadata={
            "entry_point_group": HAIRCUT_BACKEND_ENTRY_POINT_GROUP,
            "entry_point": HAIRCUT_BACKEND_ENTRY_POINT,
            "issue_refs": ",".join(_ISSUE_REFS),
            "privacy_class": "public-synthetic",
        },
    )
    method = seam.MethodCapability(
        method_id="finite_element_options.pinares_proxy_p2_theta.v0",
        backend_id=manifest.backend_id,
        family="finite_element",
        exactness="numerical_weak_form",
        maturity=seam.MethodMaturity.PRODUCTION_GATED,
        equation_families=("black_scholes_1d", "quant_problem_spec_parabolic_1d"),
        dimensions=manifest.supported_dimensions,
        state_variables=("tradable_spot",),
        boundary_conditions=manifest.boundary_conditions,
        smoothness_assumptions=(
            "piecewise_smooth_payoff",
            "public_synthetic_coefficients",
        ),
        output_types=manifest.outputs,
        runtime_controls=("theta", "time_steps", "refinement_levels", "scipy_direct"),
        validation_gates=(
            "benchmark:PINARES-FEM-FIXED-PRICE-PROXY-V0",
            "benchmark:PINARES-QPS-FIXED-PRICE-PROXY-V0",
            "pytest:tests/test_fem_backend_capabilities.py",
            "pytest:tests/test_validation_gates.py",
        ),
        failure_modes=manifest.diagnostics,
        fallback_route_id=None,
        fallback_triggers=(),
        fallback_policy="fail_closed_no_fallback",
        references=_ISSUE_REFS,
    )
    capability_manifest = seam.BackendCapabilityManifest(
        backend_id=manifest.backend_id,
        contract_version=manifest.contract_version,
        methods=(method,),
    )
    errors = capability_manifest.validate()
    if errors:
        raise RuntimeError(
            "invalid Haircut FEM capability manifest: " + "; ".join(errors)
        )
    return HaircutContracts(identity=identity, capability_manifest=capability_manifest)


def _load_public_solver_seam() -> _HaircutPublicSolverSeam:
    try:
        backend_protocol = importlib.import_module("haircut.solvers.backend_protocol")
        contracts = importlib.import_module("haircut.solvers.contracts")
    except ImportError as exc:
        raise HaircutProtocolUnavailableError(
            "Haircut public solver protocol seam is required to instantiate the FEM backend. "
            "Install a released haircut-engine wheel; local source-tree imports are not used."
        ) from exc
    return _HaircutPublicSolverSeam(
        ContractVersion=backend_protocol.ContractVersion,
        BackendMaturity=backend_protocol.BackendMaturity,
        BackendIdentity=backend_protocol.BackendIdentity,
        MethodMaturity=contracts.MethodMaturity,
        MethodCapability=contracts.MethodCapability,
        BackendCapabilityManifest=contracts.BackendCapabilityManifest,
    )


def _ensure_contract_major_compatible(
    seam: _HaircutPublicSolverSeam, provider: str, expected: str
) -> None:
    provider_version = seam.ContractVersion.parse(provider)
    expected_version = seam.ContractVersion.parse(expected)
    if not provider_version.is_compatible_with(expected_version):
        raise ContractMajorMismatchError(
            "contract major mismatch between finite_element_options and Haircut solver protocol: "
            f"provider={provider_version}, expected={expected_version}"
        )


def _distribution_version() -> str:
    try:
        return metadata.version(_DISTRIBUTION_NAME)
    except metadata.PackageNotFoundError:
        return "0.1.0"
