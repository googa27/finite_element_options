"""Validated UI-bound configuration for finite-element demos.

The Streamlit application is an optional delivery surface.  This module is the
pure-Python gate that turns widget values into immutable, shareable, and
capability-screened problem specs before any mesh, sparse matrix, or backend
resource is allocated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from math import exp, isfinite, sqrt
import json
from statistics import NormalDist
from typing import Any, Literal

from finite_element_options.contracts.backend_capabilities import (
    DEFAULT_FEM_CAPABILITY_MANIFEST,
    DEFAULT_RELEASED_FEM_SOLVER_CONTRACT,
    FEMCapabilityManifest,
    FEMRouteRequest,
    diagnose_unsupported_route,
)
from finite_element_options.space.domain import DomainAxis

UiModelName = Literal["black_scholes", "heston"]
ExerciseStyle = Literal["european", "american"]

_ALLOWED_BOUNDARIES = {"s_min", "s_max", "v_min", "v_max"}
_SCHEMA_VERSION = "finite-element-options-ui-config-v1"


@dataclass(frozen=True)
class UiValidationDiagnostic:
    """One actionable validation diagnostic for UI consumers."""

    code: str
    field: str
    message: str
    severity: Literal["error", "warning", "info"] = "error"
    value: str | None = None
    supported: tuple[str, ...] = ()

    def to_public_dict(self) -> dict[str, Any]:
        """Return a JSON-safe diagnostic record."""

        payload = asdict(self)
        payload["supported"] = list(self.supported)
        return payload


class UiValidationError(ValueError):
    """Raised when strict UI validation sees error diagnostics."""

    def __init__(self, diagnostics: Sequence[UiValidationDiagnostic]) -> None:
        """Build an exception message from blocking UI diagnostics."""

        self.diagnostics = tuple(diagnostics)
        fields = ", ".join(diag.field for diag in self.diagnostics)
        super().__init__(f"invalid UI problem configuration: {fields}")


@dataclass(frozen=True)
class UiModelSpec:
    """Immutable model/widget values before backend allocation."""

    model: UiModelName
    strike: float
    maturity: float
    rate: float
    carry: float
    volatility: float
    kappa: float | None = None
    long_run_variance: float | None = None
    vol_of_variance: float | None = None
    correlation: float | None = None
    variance_upper: float | None = None

    def to_public_dict(self) -> dict[str, Any]:
        """Return a stable shareable model payload."""

        return asdict(self)


@dataclass(frozen=True)
class UiGridSpec:
    """Immutable grid/time controls from the UI."""

    mesh_refine: int
    time_steps: int
    alpha_tail: float = 0.1
    dimension: int | None = None

    def to_public_dict(self) -> dict[str, Any]:
        """Return a stable shareable grid payload."""

        return asdict(self)


@dataclass(frozen=True)
class UiSolverOptions:
    """Immutable solver controls exposed by the UI."""

    theta: float
    exercise_style: ExerciseStyle = "european"
    dirichlet_boundaries: tuple[str, ...] = ("s_min", "s_max")
    requested_outputs: tuple[str, ...] = ("value", "delta", "gamma")
    linear_solver: str = "scipy_direct"

    def __init__(
        self,
        theta: float,
        exercise_style: ExerciseStyle = "european",
        dirichlet_boundaries: Sequence[str] = ("s_min", "s_max"),
        requested_outputs: Sequence[str] = ("value", "delta", "gamma"),
        linear_solver: str = "scipy_direct",
    ) -> None:
        """Materialize sequence widget values into immutable tuples."""

        object.__setattr__(self, "theta", float(theta))
        object.__setattr__(self, "exercise_style", exercise_style)
        object.__setattr__(
            self,
            "dirichlet_boundaries",
            tuple(str(item) for item in dirichlet_boundaries),
        )
        object.__setattr__(
            self,
            "requested_outputs",
            tuple(str(item) for item in requested_outputs),
        )
        object.__setattr__(self, "linear_solver", str(linear_solver))

    def to_public_dict(self) -> dict[str, Any]:
        """Return a stable shareable solver payload."""

        return {
            "theta": self.theta,
            "exercise_style": self.exercise_style,
            "dirichlet_boundaries": list(self.dirichlet_boundaries),
            "requested_outputs": list(self.requested_outputs),
            "linear_solver": self.linear_solver,
        }


@dataclass(frozen=True)
class UiResourceLimits:
    """Hard pre-allocation work limits for UI routes."""

    max_dofs: int = 250_000
    max_time_steps: int = 1_000
    max_matrix_bytes: int = 256 * 1024 * 1024
    max_solves: int = 1_000

    def to_public_dict(self) -> dict[str, int]:
        """Return a stable public limits payload."""

        return asdict(self)


@dataclass(frozen=True)
class UiWorkEstimate:
    """Preflight estimate of sparse FEM work before allocation."""

    dimension: int
    mesh_refine: int
    time_steps: int
    estimated_nodes: int
    estimated_dofs: int
    estimated_elements: int
    estimated_matrix_bytes: int
    solve_count: int

    @property
    def summary(self) -> str:
        """Human-readable work estimate summary."""

        mib = self.estimated_matrix_bytes / (1024 * 1024)
        return (
            f"~{self.estimated_dofs:,} dofs, {self.time_steps:,} time steps, "
            f"~{mib:.1f} MiB sparse matrix footprint"
        )

    def to_public_dict(self) -> dict[str, Any]:
        """Return a JSON-safe work estimate."""

        payload = asdict(self)
        payload["summary"] = self.summary
        return payload


@dataclass(frozen=True)
class ValidatedUiProblem:
    """Capability-screened UI problem contract."""

    model: UiModelSpec
    grid: UiGridSpec
    solver: UiSolverOptions
    limits: UiResourceLimits
    route_status: Literal["supported", "unsupported", "analytical_limit"]
    requires_numerical_solve: bool
    domain_axes: tuple[DomainAxis, ...]
    work_estimate: UiWorkEstimate
    route_request: FEMRouteRequest | None
    diagnostics: tuple[UiValidationDiagnostic, ...] = field(default_factory=tuple)

    @property
    def error_diagnostics(self) -> tuple[UiValidationDiagnostic, ...]:
        """Return diagnostics that block numerical execution."""

        return tuple(diag for diag in self.diagnostics if diag.severity == "error")

    @property
    def can_run(self) -> bool:
        """Whether the UI may run or show an explicit analytical result."""

        return not self.error_diagnostics

    @property
    def cache_key(self) -> str:
        """Return a deterministic cache key for immutable UI inputs."""

        payload = {
            "schema_version": _SCHEMA_VERSION,
            "model": self.model.to_public_dict(),
            "grid": self.grid.to_public_dict(),
            "solver": self.solver.to_public_dict(),
            "limits": self.limits.to_public_dict(),
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return sha256(raw.encode("utf-8")).hexdigest()

    def raise_for_errors(self) -> None:
        """Raise :class:`UiValidationError` if the problem is not executable."""

        errors = self.error_diagnostics
        if errors:
            raise UiValidationError(errors)

    def to_shareable_dict(self) -> dict[str, Any]:
        """Return a deterministic, secret-free, shareable UI configuration."""

        return {
            "schema_version": _SCHEMA_VERSION,
            "cache_key": self.cache_key,
            "route_status": self.route_status,
            "requires_numerical_solve": self.requires_numerical_solve,
            "model": self.model.to_public_dict(),
            "grid": self.grid.to_public_dict(),
            "solver": self.solver.to_public_dict(),
            "limits": self.limits.to_public_dict(),
            "domain": {
                "dimension": len(self.domain_axes),
                "axes": [axis.to_public_dict() for axis in self.domain_axes],
                "boundary_facets": [
                    label for axis in self.domain_axes for label in (axis.min_label, axis.max_label)
                ],
            },
            "work_estimate": self.work_estimate.to_public_dict(),
            "diagnostics": [diag.to_public_dict() for diag in self.diagnostics],
        }

    def to_status_dict(
        self,
        *,
        solver_diagnostics: Mapping[str, Any] | None = None,
        domain_diagnostics: Mapping[str, Any] | None = None,
        time_grid_diagnostics: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return route/result status for UI display and tests."""

        request = self.route_request
        if self.route_status == "supported":
            convergence_status = "benchmark_evidenced_not_reestimated_in_ui_run"
        elif self.route_status == "analytical_limit":
            convergence_status = "analytical_limit_no_fem_convergence_required"
        else:
            convergence_status = "blocked_before_allocation"
        return {
            "schema_version": f"{_SCHEMA_VERSION}-status",
            "cache_key": self.cache_key,
            "route_status": self.route_status,
            "can_run": self.can_run,
            "requires_numerical_solve": self.requires_numerical_solve,
            "backend": {
                "backend_id": DEFAULT_FEM_CAPABILITY_MANIFEST.backend_id,
                "contract_version": DEFAULT_FEM_CAPABILITY_MANIFEST.contract_version,
                "capability_maturity": DEFAULT_FEM_CAPABILITY_MANIFEST.status.value,
            },
            "benchmark_ids": list(DEFAULT_RELEASED_FEM_SOLVER_CONTRACT.public_fixture_ids),
            "approximation_status": {
                "model": self.model.model,
                "mesh_family": request.mesh_family if request else "analytical_limit",
                "element_family": request.element_family if request else "analytical_limit",
                "exercise_style": self.solver.exercise_style,
                "theta": self.solver.theta,
                "mesh_refine": self.grid.mesh_refine,
                "time_steps": self.grid.time_steps,
                "estimated_dofs": self.work_estimate.estimated_dofs,
                "requested_outputs": list(self.solver.requested_outputs),
            },
            "convergence_status": convergence_status,
            "solver_diagnostics": dict(solver_diagnostics or {}),
            "domain_diagnostics": dict(domain_diagnostics or {}),
            "time_grid_diagnostics": dict(time_grid_diagnostics or {}),
            "diagnostics": [diag.to_public_dict() for diag in self.diagnostics],
        }


def validate_ui_problem(
    *,
    model: UiModelSpec,
    grid: UiGridSpec,
    solver: UiSolverOptions,
    limits: UiResourceLimits | None = None,
    manifest: FEMCapabilityManifest = DEFAULT_FEM_CAPABILITY_MANIFEST,
    strict: bool = False,
) -> ValidatedUiProblem:
    """Validate UI inputs and capability-screen the route before allocation."""

    resolved_limits = limits or UiResourceLimits()
    diagnostics: list[UiValidationDiagnostic] = []
    analytical_limit = _analytical_limit_reason(model)
    diagnostics.extend(
        _validate_scalar_inputs(
            model,
            grid,
            solver,
            requires_numerical_solve=analytical_limit is None,
        )
    )

    dimension = _dimension(model, grid)
    domain_axes = _safe_domain_axes(
        model,
        dimension,
        grid.alpha_tail,
        analytical_limit,
        diagnostics,
    )
    diagnostics.extend(_validate_boundaries_for_domain(solver, domain_axes))
    work_estimate = estimate_ui_work(dimension, grid.mesh_refine, grid.time_steps)
    if analytical_limit is None:
        diagnostics.extend(_validate_work_estimate(work_estimate, resolved_limits))

    route_request: FEMRouteRequest | None = None
    route_status: Literal["supported", "unsupported", "analytical_limit"]
    requires_numerical_solve = analytical_limit is None
    if analytical_limit is not None:
        route_status = "analytical_limit"
        diagnostics.append(
            UiValidationDiagnostic(
                code="analytical_limit",
                field="model",
                message=analytical_limit,
                severity="info",
            )
        )
    else:
        route_request = _route_request(model, dimension, solver)
        route_diags = diagnose_unsupported_route(route_request, manifest)
        if route_diags:
            route_status = "unsupported"
            diagnostics.extend(
                UiValidationDiagnostic(
                    code=diag.reason.value,
                    field=diag.field,
                    value=diag.value,
                    supported=diag.supported,
                    message=diag.message,
                )
                for diag in route_diags
            )
        else:
            route_status = "supported"

    result = ValidatedUiProblem(
        model=model,
        grid=grid,
        solver=solver,
        limits=resolved_limits,
        route_status=route_status,
        requires_numerical_solve=requires_numerical_solve,
        domain_axes=domain_axes,
        work_estimate=work_estimate,
        route_request=route_request,
        diagnostics=tuple(diagnostics),
    )
    if strict:
        result.raise_for_errors()
    return result


def estimate_ui_work(dimension: int, mesh_refine: int, time_steps: int) -> UiWorkEstimate:
    """Estimate node, DOF, and sparse matrix cost without creating a mesh."""

    refine = max(0, int(mesh_refine))
    intervals = 2**refine
    if dimension == 1:
        elements = intervals
        nodes = 2 * intervals + 1  # P2 line nodes
        stencil = 5
    elif dimension == 2:
        elements = 2 * intervals**2
        nodes = (2 * intervals + 1) ** 2  # P2 tensor upper bound
        stencil = 25
    else:
        elements = 6 * intervals**3
        nodes = (2 * intervals + 1) ** 3
        stencil = 81
    dofs = int(nodes)
    matrix_bytes = int(dofs * stencil * 16)  # value + index rough sparse footprint
    return UiWorkEstimate(
        dimension=dimension,
        mesh_refine=refine,
        time_steps=int(time_steps),
        estimated_nodes=int(nodes),
        estimated_dofs=dofs,
        estimated_elements=int(elements),
        estimated_matrix_bytes=matrix_bytes,
        solve_count=max(0, int(time_steps) - 1),
    )


def ui_problem_from_shareable(payload: Mapping[str, Any]) -> ValidatedUiProblem:
    """Re-validate a shareable UI configuration payload."""

    if payload.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("unsupported UI configuration schema_version")
    limits_payload = payload.get("limits")
    limits = UiResourceLimits(**_mapping(limits_payload)) if limits_payload else None
    result = validate_ui_problem(
        model=UiModelSpec(**_mapping(payload.get("model"))),
        grid=UiGridSpec(**_mapping(payload.get("grid"))),
        solver=UiSolverOptions(**_mapping(payload.get("solver"))),
        limits=limits,
        strict=True,
    )
    expected_cache_key = payload.get("cache_key")
    if expected_cache_key is not None and expected_cache_key != result.cache_key:
        raise ValueError("shareable UI configuration cache_key does not match payload")
    return result


def _validate_scalar_inputs(
    model: UiModelSpec,
    grid: UiGridSpec,
    solver: UiSolverOptions,
    *,
    requires_numerical_solve: bool,
) -> list[UiValidationDiagnostic]:
    diagnostics: list[UiValidationDiagnostic] = []
    _require_positive("strike", model.strike, diagnostics)
    _require_nonnegative("maturity", model.maturity, diagnostics)
    _require_nonnegative("volatility", model.volatility, diagnostics)
    if requires_numerical_solve and int(grid.time_steps) < 2:
        diagnostics.append(
            UiValidationDiagnostic(
                code="invalid_time_grid",
                field="time_steps",
                value=str(grid.time_steps),
                message="time_steps must be at least 2 so the time grid has a start and end",
            )
        )
    if requires_numerical_solve and not 0.0 < float(grid.alpha_tail) < 0.5:
        diagnostics.append(
            UiValidationDiagnostic(
                code="invalid_tail_probability",
                field="alpha_tail",
                value=str(grid.alpha_tail),
                supported=("0 < alpha_tail < 0.5",),
                message="tail probability must be interior for domain truncation diagnostics",
            )
        )
    if requires_numerical_solve and not 0.0 <= float(solver.theta) <= 1.0:
        diagnostics.append(
            UiValidationDiagnostic(
                code="invalid_theta_scheme",
                field="theta",
                value=str(solver.theta),
                supported=("0 <= theta <= 1",),
                message="theta must lie in [0, 1] for the advertised theta scheme",
            )
        )
    unknown_boundaries = sorted(set(solver.dirichlet_boundaries) - _ALLOWED_BOUNDARIES)
    if unknown_boundaries:
        diagnostics.append(
            UiValidationDiagnostic(
                code="unknown_boundary",
                field="dirichlet_boundaries",
                value=", ".join(unknown_boundaries),
                supported=tuple(sorted(_ALLOWED_BOUNDARIES)),
                message="UI boundary names must match the solver domain facets",
            )
        )
    if model.model == "heston":
        if requires_numerical_solve:
            _require_positive("kappa", model.kappa, diagnostics)
            _require_positive("long_run_variance", model.long_run_variance, diagnostics)
            _require_nonnegative("vol_of_variance", model.vol_of_variance, diagnostics)
        else:
            _require_nonnegative("kappa", model.kappa, diagnostics)
            _require_nonnegative("long_run_variance", model.long_run_variance, diagnostics)
            _require_nonnegative("vol_of_variance", model.vol_of_variance, diagnostics)
        if model.correlation is None or not isfinite(float(model.correlation)):
            diagnostics.append(_finite_diag("correlation", model.correlation))
        elif not -1.0 < float(model.correlation) < 1.0:
            diagnostics.append(
                UiValidationDiagnostic(
                    code="singular_correlation",
                    field="correlation",
                    value=str(model.correlation),
                    supported=("-1 < correlation < 1",),
                    message="Heston correlation endpoints produce a singular covariance matrix",
                )
            )
    return diagnostics


def _validate_work_estimate(
    estimate: UiWorkEstimate, limits: UiResourceLimits
) -> list[UiValidationDiagnostic]:
    diagnostics: list[UiValidationDiagnostic] = []
    breaches = []
    if estimate.estimated_dofs > limits.max_dofs:
        breaches.append(f"dofs {estimate.estimated_dofs:,} > {limits.max_dofs:,}")
    if estimate.time_steps > limits.max_time_steps:
        breaches.append(f"time_steps {estimate.time_steps:,} > {limits.max_time_steps:,}")
    if estimate.estimated_matrix_bytes > limits.max_matrix_bytes:
        breaches.append(
            f"matrix_bytes {estimate.estimated_matrix_bytes:,} > {limits.max_matrix_bytes:,}"
        )
    if estimate.solve_count > limits.max_solves:
        breaches.append(f"solves {estimate.solve_count:,} > {limits.max_solves:,}")
    if breaches:
        diagnostics.append(
            UiValidationDiagnostic(
                code="work_limit_exceeded",
                field="work_estimate",
                value="; ".join(breaches),
                message=(
                    "estimated UI solve exceeds configured resource limits; lower "
                    "mesh refinement, time steps, or dimension before allocation"
                ),
            )
        )
    return diagnostics


def _route_request(model: UiModelSpec, dimension: int, solver: UiSolverOptions) -> FEMRouteRequest:
    terms: tuple[str, ...]
    if model.model == "black_scholes":
        mesh_family = "line_uniform"
        terms = ("drift", "diffusion", "reaction")
    else:
        mesh_family = "tri_uniform"
        terms = ("drift", "diffusion", "reaction", "mixed_derivative")
    return FEMRouteRequest(
        dimension=dimension,
        mesh_family=mesh_family,
        element_family="lagrange_p2",
        pde_terms=terms,
        boundary_conditions=("dirichlet",) if solver.dirichlet_boundaries else (),
        exercise_style=solver.exercise_style,
        requested_outputs=solver.requested_outputs,
        stability_controls=("theta",),
        linear_solver=solver.linear_solver,
        measure="risk_neutral",
        numeraire="money_market_account",
        units={"spot": "model_unit", "price": "model_unit"},
        valuation_date="ui-relative-t0",
        maturity_date=None,
        time_domain="tau_from_0_to_maturity",
        source_schema_version=_SCHEMA_VERSION,
        backend_id=DEFAULT_FEM_CAPABILITY_MANIFEST.backend_id,
    )


def _safe_domain_axes(
    model: UiModelSpec,
    dimension: int,
    alpha_tail: float,
    analytical_limit: str | None,
    diagnostics: list[UiValidationDiagnostic],
) -> tuple[DomainAxis, ...]:
    """Build domain axes or return a safe diagnostic placeholder."""

    try:
        return _domain_axes(model, dimension, alpha_tail, analytical_limit, diagnostics)
    except (OverflowError, ValueError) as exc:
        diagnostics.append(
            UiValidationDiagnostic(
                code="invalid_domain",
                field="domain",
                value=str(exc),
                message="UI inputs produce an invalid finite-element domain",
            )
        )
        return (DomainAxis("s", 0.0, 1.0, truncation_policy="invalid_input"),)


def _validate_boundaries_for_domain(
    solver: UiSolverOptions, domain_axes: Sequence[DomainAxis]
) -> list[UiValidationDiagnostic]:
    """Reject boundary facets that do not exist on the validated domain."""

    available = {
        label for axis in domain_axes for label in (axis.min_label, axis.max_label)
    }
    invalid = sorted(set(solver.dirichlet_boundaries) - available)
    if not invalid:
        return []
    return [
        UiValidationDiagnostic(
            code="unsupported_boundary_for_domain",
            field="dirichlet_boundaries",
            value=", ".join(invalid),
            supported=tuple(sorted(available)),
            message="selected boundary facets are not present on the validated domain",
        )
    ]


def _domain_axes(
    model: UiModelSpec,
    dimension: int,
    alpha_tail: float,
    analytical_limit: str | None,
    diagnostics: list[UiValidationDiagnostic],
) -> tuple[DomainAxis, ...]:
    if analytical_limit is not None:
        spot_upper = max(model.strike, 1.0)
        return (DomainAxis("s", 0.0, spot_upper, truncation_policy="analytical_limit"),)
    spot_upper = _spot_upper(model, alpha_tail)
    axes = [
        DomainAxis(
            "s",
            0.0,
            spot_upper,
            truncation_policy="ui_tail_alpha",
            tail_mass=max(min(float(alpha_tail), 0.5 - 1.0e-12), 1.0e-6),
        )
    ]
    if dimension == 2:
        variance_upper = _variance_upper(model)
        if variance_upper <= 0 or not isfinite(variance_upper):
            diagnostics.append(
                UiValidationDiagnostic(
                    code="invalid_domain",
                    field="variance_upper",
                    value=str(variance_upper),
                    message="variance upper bound must be finite and positive",
                )
            )
            variance_upper = 1.0
        axes.append(
            DomainAxis(
                "v",
                0.0,
                variance_upper,
                truncation_policy="ui_tail_alpha",
                tail_mass=max(min(float(alpha_tail), 0.5 - 1.0e-12), 1.0e-6),
            )
        )
    return tuple(axes)


def _spot_upper(model: UiModelSpec, alpha_tail: float) -> float:
    volatility = max(float(model.volatility), sqrt(max(float(model.long_run_variance or 0.0), 0.0)))
    maturity = max(float(model.maturity), 1.0e-12)
    drift = abs(float(model.rate - model.carry)) * maturity
    z_score = max(1.0, NormalDist().inv_cdf(1.0 - float(alpha_tail)))
    diffusion = z_score * max(volatility, 1.0e-12) * sqrt(maturity)
    return max(2.0 * model.strike, model.strike * exp(drift + diffusion))


def _variance_upper(model: UiModelSpec) -> float:
    if model.variance_upper is not None:
        return float(model.variance_upper)
    base = max(float(model.long_run_variance or 0.0), float(model.volatility) ** 2)
    vol_of_variance = max(float(model.vol_of_variance or 0.0), 0.0)
    return max(5.0 * base + 2.0 * vol_of_variance * sqrt(max(model.maturity, 0.0)), 1.0e-10)


def _analytical_limit_reason(model: UiModelSpec) -> str | None:
    if float(model.maturity) == 0.0:
        return "zero maturity uses the intrinsic payoff path; no mesh or time solve is allocated"
    if model.model == "black_scholes" and float(model.volatility) == 0.0:
        return "zero Black-Scholes volatility uses a deterministic discounted payoff path"
    if (
        model.model == "heston"
        and float(model.long_run_variance or 0.0) == 0.0
        and float(model.vol_of_variance or 0.0) == 0.0
    ):
        return "zero Heston variance uses a deterministic zero-variance limit"
    return None


def _dimension(model: UiModelSpec, grid: UiGridSpec) -> int:
    if grid.dimension is not None:
        return int(grid.dimension)
    return 1 if model.model == "black_scholes" else 2


def _require_positive(
    field: str, value: float | None, diagnostics: list[UiValidationDiagnostic]
) -> None:
    if value is None or not isfinite(float(value)):
        diagnostics.append(_finite_diag(field, value))
    elif float(value) <= 0.0:
        diagnostics.append(
            UiValidationDiagnostic(
                code="must_be_positive",
                field=field,
                value=str(value),
                message=f"{field} must be strictly positive for numerical routes",
            )
        )


def _require_nonnegative(
    field: str, value: float | None, diagnostics: list[UiValidationDiagnostic]
) -> None:
    if value is None or not isfinite(float(value)):
        diagnostics.append(_finite_diag(field, value))
    elif float(value) < 0.0:
        diagnostics.append(
            UiValidationDiagnostic(
                code="must_be_nonnegative",
                field=field,
                value=str(value),
                message=f"{field} must be non-negative",
            )
        )


def _finite_diag(field: str, value: object) -> UiValidationDiagnostic:
    return UiValidationDiagnostic(
        code="must_be_finite",
        field=field,
        value=str(value),
        message=f"{field} must be finite",
    )


def _mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    raise TypeError("expected mapping payload")
