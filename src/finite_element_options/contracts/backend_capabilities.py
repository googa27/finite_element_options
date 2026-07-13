"""Finite-element backend capability manifest and route diagnostics.

This module is deliberately domain-neutral.  It maps the shared
QuantProblemSpec vocabulary into a FEM adapter-screening envelope before any
mesh, basis, weak-form, or linear-solver work is allocated.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CapabilityStatus(str, Enum):
    """Maturity of one advertised backend capability."""

    PRODUCTION = "production"
    VALIDATED = "validated"
    EXPERIMENTAL = "experimental"
    UNSUPPORTED = "unsupported"


class UnsupportedReason(str, Enum):
    """Stable diagnostic codes for unsupported FEM route requests."""

    UNSUPPORTED_DIMENSION = "unsupported_dimension"
    UNSUPPORTED_MESH = "unsupported_mesh_family"
    UNSUPPORTED_ELEMENT = "unsupported_element_family"
    UNSUPPORTED_TERM = "unsupported_pde_term"
    UNSUPPORTED_BOUNDARY = "unsupported_boundary_condition"
    UNSUPPORTED_EXERCISE = "unsupported_exercise_style"
    UNSUPPORTED_OUTPUT = "unsupported_output"
    UNSUPPORTED_STABILITY_CONTROL = "unsupported_stability_control"
    UNSUPPORTED_LINEAR_SOLVER = "unsupported_linear_solver"
    UNSUPPORTED_BACKEND = "unsupported_backend"
    UNSUPPORTED_BENCHMARK = "unsupported_benchmark"
    MISSING_CONVENTION = "missing_convention"


@dataclass(frozen=True)
class UnsupportedRouteDiagnostic:
    """Actionable reason why a FEM route request is unsupported."""

    reason: UnsupportedReason
    field: str
    value: str
    supported: tuple[str, ...]
    message: str


@dataclass(frozen=True)
class SolverBackendCapability:
    """One linear-solver route advertised by the FEM backend contract."""

    name: str
    status: CapabilityStatus
    method: str
    factorization_reuse: bool
    cache_scope: str
    extra: str | None = None
    unsupported_reason: str | None = None

    def to_public_dict(self) -> dict[str, str | bool | None]:
        """Return a JSON-safe solver-route declaration."""

        return {
            "name": self.name,
            "status": self.status.value,
            "method": self.method,
            "factorization_reuse": self.factorization_reuse,
            "cache_scope": self.cache_scope,
            "extra": self.extra,
            "unsupported_reason": self.unsupported_reason,
        }


@dataclass(frozen=True)
class FEMCapabilityManifest:
    """Declarative finite-element backend support matrix.

    The manifest intentionally advertises only the validated transitional route
    used by the Project #5 QuantProblemSpec fixture: one-dimensional
    Black-Scholes/GBM, uniform line mesh, Lagrange P2 elements, theta stepping,
    and SciPy direct solves.  Higher-dimensional/adaptive/research capabilities
    must land with their own evidence before being added here.
    """

    backend_id: str
    contract_version: str
    status: CapabilityStatus
    supported_dimensions: tuple[int, ...]
    mesh_families: tuple[str, ...]
    element_families: tuple[str, ...]
    pde_terms: tuple[str, ...]
    boundary_conditions: tuple[str, ...]
    exercise_styles: tuple[str, ...]
    outputs: tuple[str, ...]
    stability_controls: tuple[str, ...]
    linear_solvers: tuple[str, ...]
    required_conventions: tuple[str, ...]
    diagnostics: tuple[str, ...]
    solver_backends: tuple[SolverBackendCapability, ...] = ()
    notes: tuple[str, ...] = ()

    def supports(self, request: FEMRouteRequest) -> bool:
        """Return ``True`` only when no fail-closed diagnostics are produced."""

        return not diagnose_unsupported_route(request, self)

    def to_public_dict(self) -> dict[str, Any]:
        """Return the public adapter/capability manifest as primitive values."""

        return {
            "backend_id": self.backend_id,
            "contract_version": self.contract_version,
            "status": self.status.value,
            "supported_dimensions": list(self.supported_dimensions),
            "mesh_families": list(self.mesh_families),
            "element_families": list(self.element_families),
            "pde_terms": list(self.pde_terms),
            "boundary_conditions": list(self.boundary_conditions),
            "exercise_styles": list(self.exercise_styles),
            "outputs": list(self.outputs),
            "stability_controls": list(self.stability_controls),
            "linear_solvers": list(self.linear_solvers),
            "required_conventions": list(self.required_conventions),
            "diagnostics": list(self.diagnostics),
            "solver_backends": [item.to_public_dict() for item in self.solver_backends],
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class FEMSolverContract:
    """Released public FEM solver contract for downstream parity consumers."""

    contract_id: str
    contract_version: str
    backend_id: str
    privacy_class: str
    manifest: FEMCapabilityManifest
    public_fixture_ids: tuple[str, ...]
    public_fixture_paths: tuple[str, ...]
    compatibility_notes: tuple[str, ...]
    forbidden_dependencies: tuple[str, ...]

    def to_public_dict(self) -> dict[str, Any]:
        """Return a JSON-safe released solver contract."""

        return {
            "contract_id": self.contract_id,
            "contract_version": self.contract_version,
            "backend_id": self.backend_id,
            "privacy_class": self.privacy_class,
            "capability_manifest": self.manifest.to_public_dict(),
            "public_fixture_ids": list(self.public_fixture_ids),
            "public_fixture_paths": list(self.public_fixture_paths),
            "compatibility_notes": list(self.compatibility_notes),
            "forbidden_dependencies": list(self.forbidden_dependencies),
        }


@dataclass(frozen=True)
class FEMRouteRequest:
    """FEM adapter request distilled from a QuantProblemSpec-like payload.

    This is the capability-screening envelope, not the assembled weak form.  It
    preserves conventions agents must not drop: measure, numeraire, units,
    valuation/maturity timing, boundary classes, requested outputs, mesh/element
    policy and solver controls.
    """

    dimension: int
    mesh_family: str
    element_family: str
    pde_terms: tuple[str, ...]
    boundary_conditions: tuple[str, ...]
    exercise_style: str
    requested_outputs: tuple[str, ...]
    stability_controls: tuple[str, ...]
    linear_solver: str
    measure: str | None
    numeraire: str | None
    units: Mapping[str, str] = field(default_factory=dict)
    boundary_details: Mapping[str, str] = field(default_factory=dict)
    valuation_date: str | None = None
    maturity_date: str | None = None
    time_domain: str | None = None
    source_schema_version: str | None = None
    backend_id: str | None = None

    @classmethod
    def from_quant_problem_spec(cls, payload: Mapping[str, Any]) -> FEMRouteRequest:
        """Map a shared QuantProblemSpec-like mapping to a FEM route request."""

        math = _mapping(payload.get("mathematical_problem"))
        solver = _mapping(payload.get("solver_plan"))
        context = _mapping(payload.get("valuation_context"))
        conventions = _mapping(payload.get("conventions"))
        vintage = _mapping(payload.get("vintage"))
        domain = _mapping(math.get("domain"))
        boundary_details = _mapping(math.get("boundary_conditions"))

        dimension = int(
            _first_present(
                math,
                ("dimension", "dimensions", "state_dimension"),
                default=_state_dimension(math.get("state_variables")),
            )
        )

        return cls(
            dimension=dimension,
            mesh_family=str(
                _first_present(
                    solver,
                    ("mesh_family", "mesh", "mesh_type", "grid_type", "grid_family"),
                    default="line_uniform",
                )
            ),
            element_family=str(
                _first_present(
                    solver, ("element_family", "element", "space_family"), default="lagrange_p2"
                )
            ),
            pde_terms=_tuple_of_strings(
                _first_present(
                    math, ("pde_terms", "terms"), default=("drift", "diffusion", "reaction")
                )
            ),
            boundary_conditions=_boundary_condition_classes(
                _first_present(
                    math,
                    ("boundary_types", "boundaries", "boundary_conditions"),
                    default=("dirichlet",),
                )
            ),
            exercise_style=str(
                _first_present(math, ("exercise_style", "exercise"), default="european")
            ),
            requested_outputs=_tuple_of_strings(
                _first_present(
                    solver, ("requested_outputs", "required_outputs", "outputs"), default=("value",)
                )
            ),
            stability_controls=_tuple_of_strings(
                _first_present(solver, ("stability_controls", "stability"), default=("theta",))
            ),
            linear_solver=str(
                _first_present(
                    solver,
                    ("linear_solver", "solver", "linear_solver_policy"),
                    default="scipy_direct",
                )
            ),
            measure=_optional_string(
                _first_present(
                    context,
                    ("measure",),
                    default=_first_present(
                        math, ("measure_id", "measure"), default=conventions.get("measure")
                    ),
                )
            ),
            numeraire=_optional_string(
                _first_present(
                    context,
                    ("numeraire",),
                    default=_first_present(
                        math, ("numeraire_id", "numeraire"), default=conventions.get("numeraire")
                    ),
                )
            ),
            units=_mapping(
                _first_present(
                    context,
                    ("units",),
                    default=_first_present(math, ("units",), default=conventions.get("units", {})),
                )
            ),
            boundary_details={str(key): str(value) for key, value in boundary_details.items()},
            valuation_date=_optional_string(
                _first_present(
                    context, ("valuation_date", "as_of_date"), default=vintage.get("valuation_date")
                )
            ),
            maturity_date=_optional_string(
                _first_present(context, ("maturity_date",), default=vintage.get("maturity_date"))
            ),
            time_domain=_optional_string(
                _first_present(context, ("time_domain",), default=domain.get("t"))
            ),
            source_schema_version=_optional_string(payload.get("schema_version")),
            backend_id=_optional_string(
                _first_present(solver, ("backend_id", "backend"), default=None)
            ),
        )


DEFAULT_LINEAR_SOLVER_CAPABILITIES = (
    SolverBackendCapability(
        name="scipy_direct",
        status=CapabilityStatus.VALIDATED,
        method="scipy.sparse.linalg.splu",
        factorization_reuse=True,
        cache_scope="per invariant theta-system solve; invalidated by mesh, dt, theta, operator or boundary-dof changes",
    ),
    SolverBackendCapability(
        name="scipy_banded",
        status=CapabilityStatus.UNSUPPORTED,
        method="scipy.linalg.solve_banded",
        factorization_reuse=False,
        cache_scope="not advertised for current FEM sparse boundary-eliminated matrices",
        unsupported_reason="banded extraction/equal-error residual evidence is not part of the released FEM route",
    ),
    SolverBackendCapability(
        name="amg",
        status=CapabilityStatus.UNSUPPORTED,
        method="pyamg or scipy iterative with AMG preconditioner",
        factorization_reuse=False,
        cache_scope="optional dependency route fails closed",
        extra="amg",
        unsupported_reason="AMG convergence, tolerance and equal-error benchmark evidence is absent",
    ),
    SolverBackendCapability(
        name="petsc",
        status=CapabilityStatus.UNSUPPORTED,
        method="petsc4py/KSP",
        factorization_reuse=False,
        cache_scope="optional dependency route fails closed",
        extra="petsc",
        unsupported_reason="PETSc platform/profile and parity evidence is absent",
    ),
)


DEFAULT_FEM_CAPABILITY_MANIFEST = FEMCapabilityManifest(
    backend_id="finite_element_options.fem_backend.v0",
    contract_version="0.1.0",
    status=CapabilityStatus.VALIDATED,
    supported_dimensions=(1,),
    mesh_families=("line_uniform",),
    element_families=("lagrange_p2",),
    pde_terms=("drift", "diffusion", "reaction"),
    boundary_conditions=("dirichlet",),
    exercise_styles=("european",),
    outputs=("value", "delta", "gamma"),
    stability_controls=("theta", "crank_nicolson"),
    linear_solvers=("scipy_direct",),
    required_conventions=(
        "measure",
        "numeraire",
        "units",
        "valuation_date",
        "maturity_or_time_domain",
    ),
    diagnostics=(
        "unsupported dimension",
        "unsupported mesh family",
        "unsupported element family",
        "unsupported PDE term",
        "unsupported boundary condition",
        "unsupported exercise style",
        "unsupported output",
        "missing measure/numeraire/units/date convention",
    ),
    solver_backends=DEFAULT_LINEAR_SOLVER_CAPABILITIES,
    notes=(
        "The executable parity fixture validates the 1D public-synthetic Black-Scholes call value.",
        "The Pinares fixed-price proxy validates the same 1D weak-form envelope with public-synthetic UF units, survival-scaled terminal payoff, Lagrange P2 line mesh, theta stepping and SciPy direct solves.",
        "Adaptive meshes, higher-order elements, American exercise, obstacles/free boundaries, HJB/control and jump terms fail closed until evidenced.",
        "Greek output names are backed by deterministic central-stencil Delta/Gamma errors in the public parity fixtures; broader kink-aware production evidence remains separate.",
    ),
)


DEFAULT_RELEASED_FEM_SOLVER_CONTRACT = FEMSolverContract(
    contract_id="finite-element-options-fem-solver-contract-v0.1",
    contract_version="0.1.0",
    backend_id=DEFAULT_FEM_CAPABILITY_MANIFEST.backend_id,
    privacy_class="public_synthetic",
    manifest=DEFAULT_FEM_CAPABILITY_MANIFEST,
    public_fixture_ids=(
        "fem-bs-001",
        "PINARES-FEM-FIXED-PRICE-PROXY-V0",
        "PINARES-QPS-FIXED-PRICE-PROXY-V0",
        "PINARES-FEM-FAIL-CLOSED-V0",
    ),
    public_fixture_paths=(
        "tests/fixtures/fem_bs_001/problem_spec.json",
        "tests/fixtures/fem_bs_001/result_export.json",
        "tests/fixtures/quant_problem_specs/pinares_fixed_price_proxy.json",
        "tests/fixtures/fem_pinares_fixed_price_proxy_v1/problem_spec.json",
        "tests/fixtures/fem_pinares_fixed_price_proxy_v1/result_export.json",
        "tests/fixtures/fem_pinares_fixed_price_proxy_v1/unsupported_full_deal_problem_spec.json",
    ),
    compatibility_notes=(
        "Downstream consumers call released finite_element_options public contracts and fixtures, not Pinares private modules.",
        "Only the public-synthetic fixed-price proxy route is supported for Pinares parity.",
        "Full family-contract, ROFR, obstacle/free-boundary, jump/liquidity and HJB/control requests fail closed before mesh allocation.",
    ),
    forbidden_dependencies=(
        "Pinares private modules",
        "haircut-engine domain/application modules",
        "PDP ingestion modules",
        "UI/calibration/orchestration internals",
    ),
)


def diagnose_unsupported_route(
    request: FEMRouteRequest,
    manifest: FEMCapabilityManifest = DEFAULT_FEM_CAPABILITY_MANIFEST,
) -> tuple[UnsupportedRouteDiagnostic, ...]:
    """Return fail-closed diagnostics for unsupported request fields."""

    diagnostics: list[UnsupportedRouteDiagnostic] = []
    if request.backend_id and request.backend_id != manifest.backend_id:
        diagnostics.append(
            _diagnostic(
                UnsupportedReason.UNSUPPORTED_BACKEND,
                "backend_id",
                request.backend_id,
                (manifest.backend_id,),
                f"Unsupported backend_id {request.backend_id!r}; expected {manifest.backend_id!r}.",
            )
        )

    if request.dimension not in manifest.supported_dimensions:
        diagnostics.append(
            _diagnostic(
                UnsupportedReason.UNSUPPORTED_DIMENSION,
                "dimension",
                str(request.dimension),
                tuple(str(item) for item in manifest.supported_dimensions),
                f"FEM backend supports dimensions {manifest.supported_dimensions}, got {request.dimension}D.",
            )
        )

    _extend_set_diagnostics(
        diagnostics,
        UnsupportedReason.UNSUPPORTED_MESH,
        "mesh_family",
        (request.mesh_family,),
        manifest.mesh_families,
    )
    _extend_set_diagnostics(
        diagnostics,
        UnsupportedReason.UNSUPPORTED_ELEMENT,
        "element_family",
        (request.element_family,),
        manifest.element_families,
    )
    _extend_set_diagnostics(
        diagnostics,
        UnsupportedReason.UNSUPPORTED_TERM,
        "pde_terms",
        request.pde_terms,
        manifest.pde_terms,
    )
    if request.boundary_conditions:
        _extend_set_diagnostics(
            diagnostics,
            UnsupportedReason.UNSUPPORTED_BOUNDARY,
            "boundary_conditions",
            request.boundary_conditions,
            manifest.boundary_conditions,
        )
    else:
        diagnostics.append(
            _diagnostic(
                UnsupportedReason.UNSUPPORTED_BOUNDARY,
                "boundary_conditions",
                "<missing>",
                manifest.boundary_conditions,
                "QuantProblemSpec mapping must declare at least one boundary condition class before FEM routing.",
            )
        )
    _extend_set_diagnostics(
        diagnostics,
        UnsupportedReason.UNSUPPORTED_EXERCISE,
        "exercise_style",
        (request.exercise_style,),
        manifest.exercise_styles,
    )
    _extend_set_diagnostics(
        diagnostics,
        UnsupportedReason.UNSUPPORTED_OUTPUT,
        "requested_outputs",
        request.requested_outputs,
        manifest.outputs,
    )
    _extend_set_diagnostics(
        diagnostics,
        UnsupportedReason.UNSUPPORTED_STABILITY_CONTROL,
        "stability_controls",
        request.stability_controls,
        manifest.stability_controls,
    )
    _extend_set_diagnostics(
        diagnostics,
        UnsupportedReason.UNSUPPORTED_LINEAR_SOLVER,
        "linear_solver",
        (request.linear_solver,),
        manifest.linear_solvers,
    )

    conventions = {
        "measure": request.measure,
        "numeraire": request.numeraire,
        "units": request.units,
        "valuation_date": request.valuation_date,
        "maturity_or_time_domain": request.maturity_date or request.time_domain,
    }
    for field_name in manifest.required_conventions:
        if not conventions.get(field_name):
            diagnostics.append(
                _diagnostic(
                    UnsupportedReason.MISSING_CONVENTION,
                    field_name,
                    "<missing>",
                    ("required",),
                    f"QuantProblemSpec mapping must preserve {field_name}; it was missing or empty.",
                )
            )

    return tuple(diagnostics)


def ensure_route_supported(
    request: FEMRouteRequest,
    manifest: FEMCapabilityManifest = DEFAULT_FEM_CAPABILITY_MANIFEST,
) -> None:
    """Raise :class:`UnsupportedRouteError` if the manifest rejects the request."""

    diagnostics = diagnose_unsupported_route(request, manifest)
    if diagnostics:
        raise UnsupportedRouteError(diagnostics)


class UnsupportedRouteError(ValueError):
    """Raised when a route request is unsupported before numerical work."""

    def __init__(self, diagnostics: Iterable[UnsupportedRouteDiagnostic]) -> None:
        """Store diagnostics and render a concise human-readable reason list."""

        self.diagnostics = tuple(diagnostics)
        reasons = "; ".join(d.message for d in self.diagnostics)
        super().__init__(reasons)


def _diagnostic(
    reason: UnsupportedReason,
    field_name: str,
    value: str,
    supported: tuple[str, ...],
    message: str,
) -> UnsupportedRouteDiagnostic:
    return UnsupportedRouteDiagnostic(
        reason=reason,
        field=field_name,
        value=value,
        supported=supported,
        message=message,
    )


def _extend_set_diagnostics(
    diagnostics: list[UnsupportedRouteDiagnostic],
    reason: UnsupportedReason,
    field_name: str,
    requested: tuple[str, ...],
    supported: tuple[str, ...],
) -> None:
    supported_set = set(supported)
    for value in requested:
        if value not in supported_set:
            diagnostics.append(
                _diagnostic(
                    reason,
                    field_name,
                    value,
                    supported,
                    f"Unsupported {field_name} value {value!r}; supported values are {supported}.",
                )
            )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_present(mapping: Mapping[str, Any], keys: tuple[str, ...], *, default: Any) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _tuple_of_strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable) and not isinstance(value, Mapping):
        return tuple(str(item) for item in value)
    return (str(value),)


def _state_dimension(value: Any) -> int:
    if isinstance(value, str):
        return 1
    if isinstance(value, Iterable) and not isinstance(value, Mapping):
        values = tuple(value)
        return len(values) or 1
    return 1


def _boundary_condition_classes(value: Any) -> tuple[str, ...]:
    """Normalize public schema boundary formulas to FEM capability classes."""

    if isinstance(value, Mapping):
        raw_items: Iterable[tuple[str, Any]] = value.items()
    else:
        raw_items = (("", item) for item in _tuple_of_strings(value))

    classes: list[str] = []
    for location, item in raw_items:
        text = str(item).lower().replace("-", "_")
        location_text = str(location).lower().replace("-", "_")
        if "free" in text and "boundary" in text:
            boundary_class = "free_boundary"
        elif "robin" in text:
            boundary_class = "robin"
        elif "neumann" in text or "slope" in text:
            boundary_class = "neumann"
        elif "dirichlet" in text or "absorbing" in text or text.strip() in {"0", "zero"}:
            boundary_class = "dirichlet"
        elif ("linear" in text or "growth" in text) and any(
            marker in location_text
            for marker in ("s=0", "s_min", "lower", "left", "s_max", "upper", "right", "far_field")
        ):
            boundary_class = "dirichlet"
        else:
            boundary_class = text
        if boundary_class not in classes:
            classes.append(boundary_class)
    return tuple(classes)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


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
