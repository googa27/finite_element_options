"""Public constants and screening helpers for compiled weak-form fixtures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any

from finite_element_options.contracts import DEFAULT_FEM_CAPABILITY_MANIFEST

FIXTURE_SCHEMA_VERSION = "finite-element-options.compiled-weak-form-fixture/v0"
COMPILED_OPERATOR_SCHEMA_VERSION = "compiled_symbolic_operator.v0"
PDE_IR_SCHEMA_VERSION = "pde_ir.v0"
PUBLIC_BS_FIXTURE_ID = "VQPW-FEM-COMPILED-BS-CALL-V0"
PUBLIC_BS_SOURCE_PROBLEM_ID = "black_scholes_call_public_synthetic"
PUBLIC_BS_FORMULATION_ID = "black_scholes_call_pde_v0"
PUBLIC_BS_SOURCE_HASH = "sha256:5ab53779a5e322284a6cb18b22302c119f22bc740659aedf1c07823529d68a47"
PUBLIC_BS_COMPILED_HASH = "sha256:970088e5dcb16535edfd230bfe992ea7eb68aede901c7b543682b39f1a5ac32e"

ALLOWED_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "fixture_id",
        "privacy_class",
        "source_repo",
        "source_fixture_path",
        "compiler_issue",
        "fem_issue",
        "pde_ir",
        "compiled_operator",
        "fem_route",
    }
)
SUPPORTED_OUTPUTS = ("value", "delta", "gamma")
SUPPORTED_ROUTE = {
    "backend_id": DEFAULT_FEM_CAPABILITY_MANIFEST.backend_id,
    "mesh_family": "line_uniform",
    "element_family": "lagrange_p2",
    "time_integrator": "theta_crank_nicolson",
    "linear_solver": "scipy_direct",
}


@dataclass(frozen=True)
class CompiledWeakFormDiagnostic:
    """One fail-closed screening diagnostic for a compiled weak-form fixture."""

    code: str
    field: str
    value: str
    message: str

    def to_public_dict(self) -> dict[str, str]:
        """Return deterministic JSON-compatible diagnostic data."""

        return {
            "code": self.code,
            "field": self.field,
            "value": self.value,
            "message": self.message,
        }


@dataclass(frozen=True)
class CompiledWeakFormScreen:
    """Screening result produced before mesh, basis or assembly allocation."""

    accepted: bool
    fixture_id: str | None
    problem_id: str | None
    source_ir_hash: str | None
    compiled_operator_hash: str | None
    route_hash: str | None
    diagnostics: tuple[CompiledWeakFormDiagnostic, ...]
    request: dict[str, Any]

    def to_public_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible screen response."""

        return {
            "accepted": self.accepted,
            "fixture_id": self.fixture_id,
            "problem_id": self.problem_id,
            "source_ir_hash": self.source_ir_hash,
            "compiled_operator_hash": self.compiled_operator_hash,
            "route_hash": self.route_hash,
            "capability_status": capability_status(),
            "request": self.request,
            "diagnostics": [item.to_public_dict() for item in self.diagnostics],
        }


class CompiledWeakFormUnsupportedError(ValueError):
    """Raised when a compiled weak-form fixture is rejected before assembly."""

    def __init__(self, screen: CompiledWeakFormScreen) -> None:
        """Create an exception carrying the rejected screening report."""

        self.screen = screen
        messages = "; ".join(item.message for item in screen.diagnostics)
        super().__init__(messages or "compiled weak-form fixture is unsupported")


def capability_status() -> dict[str, Any]:
    """Return deterministic public capability status for compiled routes."""

    return {
        "backend_id": DEFAULT_FEM_CAPABILITY_MANIFEST.backend_id,
        "manifest_status": DEFAULT_FEM_CAPABILITY_MANIFEST.status.value,
        "released_route": "line_uniform/lagrange_p2/theta_crank_nicolson/scipy_direct",
        "unsupported": [
            "fenicsx",
            "petsc",
            "adaptive",
            "multidimensional",
            "american",
            "private_fixtures",
        ],
    }


def reject_private_markers(
    value: Any, diagnostics: list[CompiledWeakFormDiagnostic], path: str = ""
) -> None:
    """Append diagnostics for private or credential-shaped fixture fields."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            lowered = str(key).lower()
            allowed_public_flag = lowered == "contains_private_data" and str(child).lower() == "false"
            if not allowed_public_flag and (
                "private" in lowered or lowered in {"credential", "credentials", "secret", "token"}
            ):
                append_diagnostic(
                    diagnostics,
                    "compiled_weak_form.private_field",
                    child_path,
                    "<redacted>",
                    "Private/credential-bearing fields are not allowed in public fixtures.",
                )
            reject_private_markers(child, diagnostics, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_private_markers(child, diagnostics, f"{path}[{index}]")


def expect_field(
    actual: Any,
    expected: Any,
    field: str,
    code: str,
    diagnostics: list[CompiledWeakFormDiagnostic],
) -> None:
    """Append a deterministic diagnostic when ``actual`` differs from ``expected``."""

    if actual != expected:
        append_diagnostic(
            diagnostics,
            code,
            field,
            stringify(actual),
            f"Expected {field}={expected!r}; got {actual!r}.",
        )


def append_diagnostic(
    diagnostics: list[CompiledWeakFormDiagnostic],
    code: str,
    field: str,
    value: str,
    message: str,
) -> None:
    """Append one compiled weak-form diagnostic."""

    diagnostics.append(CompiledWeakFormDiagnostic(code=code, field=field, value=value, message=message))


def rejected(fixture_id: str | None, field: str, value: str, message: str) -> CompiledWeakFormScreen:
    """Build a rejected screen for payload loading failures."""

    diagnostic = CompiledWeakFormDiagnostic("compiled_weak_form.load", field, value, message)
    return CompiledWeakFormScreen(False, fixture_id, None, None, None, None, (diagnostic,), {})


def as_mapping(value: Any) -> Mapping[str, Any]:
    """Return ``value`` as a mapping or an empty mapping."""

    return value if isinstance(value, Mapping) else {}


def as_sequence(value: Any) -> Sequence[Any]:
    """Return ``value`` as a non-string sequence or an empty tuple."""

    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) else ()


def optional_string(value: Any) -> str | None:
    """Return ``value`` as a string unless it is missing."""

    return None if value is None else str(value)


def stringify(value: Any) -> str:
    """Return a deterministic public string for diagnostics."""

    return "<missing>" if value is None else str(value)


def hash_json(payload: Mapping[str, Any]) -> str:
    """Return a deterministic SHA-256 hash for a JSON mapping."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + sha256(encoded).hexdigest()


def json_roundtrip(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-normalized copy of ``payload``."""

    return json.loads(json.dumps(payload, sort_keys=True))
