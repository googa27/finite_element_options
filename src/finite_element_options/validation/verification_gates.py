"""Executable numerical verification gates for FEM capability claims.

The gates in this module are deliberately lightweight and deterministic.  They
encode the evidence contract for Project #5 issue #42: production/validated FEM
claims must cite benchmark identifiers, convergence studies must separate error
budgets, arbitrage and complementarity failures must fail closed, and backend
parity must compare identical mathematical inputs rather than just similarly
named examples.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite, log
from typing import Any

import numpy as np

from finite_element_options.contracts.capability_matrix import CapabilityRecord, CapabilityStatus
from finite_element_options.time_integration import LCPDiagnostics

REQUIRED_TOLERANCE_COMPONENTS = ("discretization", "oracle", "floating_point")
DEFAULT_VALIDATION_BENCHMARK_ID = "FEM-VALIDATION-GATES-V0"


class ValidationGateError(AssertionError):
    """Raised when a numerical verification gate rejects a claim."""


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    """Machine-readable benchmark contract for one production claim."""

    benchmark_id: str
    model: str
    instrument: str
    state_convention: str
    domain: str
    grid: str
    time_schedule: str
    oracle: str
    norm: str
    expected_order: float
    tolerance_components: Mapping[str, float]

    def validate(self) -> None:
        """Validate that the benchmark can be audited without tribal context."""

        fields = {
            "benchmark_id": self.benchmark_id,
            "model": self.model,
            "instrument": self.instrument,
            "state_convention": self.state_convention,
            "domain": self.domain,
            "grid": self.grid,
            "time_schedule": self.time_schedule,
            "oracle": self.oracle,
            "norm": self.norm,
        }
        missing = tuple(name for name, value in fields.items() if not str(value).strip())
        if missing:
            raise ValidationGateError(f"benchmark {self.benchmark_id!r} missing fields: {missing}")
        if not isfinite(self.expected_order) or self.expected_order < 0.0:
            raise ValidationGateError(f"benchmark {self.benchmark_id!r} has invalid expected order")
        missing_components = tuple(
            component
            for component in REQUIRED_TOLERANCE_COMPONENTS
            if component not in self.tolerance_components
        )
        if missing_components:
            raise ValidationGateError(
                f"benchmark {self.benchmark_id!r} missing tolerance components: "
                f"{missing_components}"
            )
        for component, value in self.tolerance_components.items():
            if not isfinite(float(value)) or float(value) < 0.0:
                raise ValidationGateError(
                    f"benchmark {self.benchmark_id!r} has invalid tolerance component {component!r}"
                )

    def to_public_dict(self) -> dict[str, Any]:
        """Return a JSON-safe benchmark declaration."""

        self.validate()
        return {
            "benchmark_id": self.benchmark_id,
            "model": self.model,
            "instrument": self.instrument,
            "state_convention": self.state_convention,
            "domain": self.domain,
            "grid": self.grid,
            "time_schedule": self.time_schedule,
            "oracle": self.oracle,
            "norm": self.norm,
            "expected_order": self.expected_order,
            "tolerance_components": dict(self.tolerance_components),
        }


@dataclass(frozen=True, slots=True)
class BenchmarkCoverageAudit:
    """Audit result for capability-record benchmark coverage."""

    accepted: bool
    missing_benchmark_ids: tuple[str, ...]
    production_without_benchmark_ids: tuple[str, ...]
    validated_without_benchmark_ids: tuple[str, ...]
    validated_records: tuple[str, ...]

    def to_public_dict(self) -> dict[str, Any]:
        """Return an actionable benchmark-coverage table."""

        return {
            "accepted": self.accepted,
            "missing_benchmark_ids": list(self.missing_benchmark_ids),
            "production_without_benchmark_ids": list(self.production_without_benchmark_ids),
            "validated_without_benchmark_ids": list(self.validated_without_benchmark_ids),
            "validated_records": list(self.validated_records),
        }


def default_benchmark_registry() -> dict[str, BenchmarkSpec]:
    """Return benchmark specs for every committed validated production claim."""

    specs = (
        BenchmarkSpec(
            benchmark_id="pytest-benchmark:black_scholes_benchmark",
            model="Black-Scholes",
            instrument="European call and put",
            state_convention="forward pseudo-time tau on normalized spot domain",
            domain="one-dimensional line mesh [0, 4K]",
            grid="line_uniform lagrange_p2 refinement smoke grid",
            time_schedule="theta/Crank-Nicolson fixed-step smoke schedule",
            oracle="closed-form Black-Scholes analytical value",
            norm="absolute wall-clock benchmark plus price smoke tolerance",
            expected_order=0.0,
            tolerance_components={
                "discretization": 2.0e-3,
                "oracle": 1.0e-12,
                "floating_point": 1.0e-12,
            },
        ),
        BenchmarkSpec(
            benchmark_id="fem-bs-001",
            model="Black-Scholes",
            instrument="European call",
            state_convention="risk-neutral money-market numeraire, forward tau",
            domain="public synthetic normalized spot [0, 4]",
            grid="line_uniform lagrange_p2 refinements 4/5/6",
            time_schedule="theta_crank_nicolson, 80 time steps",
            oracle="closed-form Black-Scholes price, Delta and Gamma",
            norm="price/Delta/Gamma absolute and relative error budget",
            expected_order=2.0,
            tolerance_components={
                "discretization": 2.0e-3,
                "oracle": 1.0e-12,
                "floating_point": 1.0e-10,
            },
        ),
        BenchmarkSpec(
            benchmark_id="FEM-SOLVER-CACHE-001",
            model="Black-Scholes",
            instrument="European call repeated solve",
            state_convention="same matrix policy across repeated theta solves",
            domain="one-dimensional public synthetic line mesh",
            grid="line_uniform lagrange_p2 fixed refinement",
            time_schedule="theta/Crank-Nicolson repeated fixed schedule",
            oracle="factorization reuse count and identical terminal value",
            norm="factorization reuse ratio and max absolute value drift",
            expected_order=0.0,
            tolerance_components={
                "discretization": 1.0e-12,
                "oracle": 1.0e-12,
                "floating_point": 1.0e-12,
            },
        ),
        BenchmarkSpec(
            benchmark_id="FEM-THETA-TIME-GRID",
            model="Black-Scholes theta-family semidiscrete PDE",
            instrument="European call with nonuniform output grids",
            state_convention="forward pseudo-time tau with new-time boundary/source refresh",
            domain="one-dimensional line-uniform finite-element mesh",
            grid="lagrange_p1/p2 deterministic smoke grids",
            time_schedule="increasing nonuniform local dt with optional Rannacher startup",
            oracle="linear solve diagnostics and boundary/source timing invariants",
            norm="grid validation, finite residuals, startup schedule and factorization reuse",
            expected_order=0.0,
            tolerance_components={
                "discretization": 1.0e-10,
                "oracle": 1.0e-12,
                "floating_point": 1.0e-12,
            },
        ),
        BenchmarkSpec(
            benchmark_id="FEM-AMERICAN-LCP-REFERENCE",
            model="Black-Scholes lower-obstacle LCP",
            instrument="American put exercise diagnostic problem",
            state_convention="forward pseudo-time tau with lower payoff obstacle",
            domain="small sparse deterministic one-dimensional systems",
            grid="synthetic obstacle and theta-step sparse matrices",
            time_schedule="theta/Crank-Nicolson steps with Rannacher-compatible diagnostics",
            oracle="KKT complementarity, projected residual and exercise-front invariants",
            norm="primal/dual/complementarity/projected residual tolerance table",
            expected_order=0.0,
            tolerance_components={
                "discretization": 1.0e-8,
                "oracle": 1.0e-12,
                "floating_point": 1.0e-12,
            },
        ),
        BenchmarkSpec(
            benchmark_id="PINARES-FEM-FIXED-PRICE-PROXY-V0",
            model="Black-Scholes-style public Pinares proxy",
            instrument="fixed-price purchase-option proxy",
            state_convention="UF money-market numeraire proxy with survival scaling",
            domain="public synthetic one-dimensional UF price interval",
            grid="line_uniform lagrange_p2 refinements",
            time_schedule="theta_crank_nicolson fixed public schedule",
            oracle="analytical survival-scaled Black-Scholes proxy",
            norm="UF price absolute/relative error and convergence rows",
            expected_order=2.0,
            tolerance_components={
                "discretization": 1.0e-3,
                "oracle": 1.0e-10,
                "floating_point": 1.0e-10,
            },
        ),
        BenchmarkSpec(
            benchmark_id=DEFAULT_VALIDATION_BENCHMARK_ID,
            model="validation meta-suite",
            instrument="capability, convergence, arbitrage, parity and LCP gates",
            state_convention="declared per benchmark before comparison",
            domain="per-benchmark declared spatial and time domains",
            grid="per-benchmark grid and backend hashes",
            time_schedule="per-benchmark time schedule hashes",
            oracle="manufactured, analytical, property and cross-backend oracles",
            norm="gate-specific actionable error tables",
            expected_order=0.0,
            tolerance_components={
                "discretization": 1.0e-3,
                "oracle": 1.0e-10,
                "floating_point": 1.0e-12,
            },
        ),
    )
    registry = {spec.benchmark_id: spec for spec in specs}
    for spec in registry.values():
        spec.validate()
    return registry


def audit_capability_benchmark_coverage(
    records: Sequence[CapabilityRecord], registry: Mapping[str, BenchmarkSpec] | None = None
) -> BenchmarkCoverageAudit:
    """Check that capability records cite committed benchmark specs."""

    benchmark_registry = registry or default_benchmark_registry()
    missing: list[str] = []
    production_without: list[str] = []
    validated_without: list[str] = []
    validated_records: list[str] = []
    for record in records:
        if record.status in {CapabilityStatus.VALIDATED, CapabilityStatus.PRODUCTION}:
            validated_records.append(record.capability_id)
            if not record.benchmark_ids:
                if record.status == CapabilityStatus.PRODUCTION:
                    production_without.append(record.capability_id)
                else:
                    validated_without.append(record.capability_id)
        for benchmark_id in record.benchmark_ids:
            if benchmark_id not in benchmark_registry:
                missing.append(f"{record.capability_id}:{benchmark_id}")
    return BenchmarkCoverageAudit(
        accepted=not missing and not production_without and not validated_without,
        missing_benchmark_ids=tuple(missing),
        production_without_benchmark_ids=tuple(production_without),
        validated_without_benchmark_ids=tuple(validated_without),
        validated_records=tuple(validated_records),
    )


@dataclass(frozen=True, slots=True)
class ManufacturedSolutionCase:
    """Small manufactured PDE residual canary."""

    operator_family: str
    equation: str
    sample_points: tuple[tuple[float, ...], ...]
    residual_tolerance: float

    def residual(self, point: tuple[float, ...]) -> float:
        """Return the manufactured residual at ``point``.

        Each supported operator family has an explicit smooth manufactured
        solution and forcing term.  Returning a residual for an unknown family
        is forbidden because it would let a misspelled or unsupported operator
        masquerade as a passed verification case.
        """

        if self.operator_family == "diffusion":
            if len(point) != 2:
                raise ValidationGateError(f"diffusion point must be (x, tau): {point}")
            x, tau = point
            value = float(np.sin(np.pi * x) * np.exp(-tau))
            u_tau = -value
            u_xx = -(np.pi**2) * value
            diffusion = 0.25 + 0.05 * x
            forcing = u_tau - diffusion * u_xx
            return u_tau - diffusion * u_xx - forcing
        if self.operator_family == "convection_diffusion":
            if len(point) != 2:
                raise ValidationGateError(f"convection-diffusion point must be (x, tau): {point}")
            x, tau = point
            value = float(np.cos(np.pi * x / 2.0) * np.exp(-0.5 * tau))
            u_tau = -0.5 * value
            u_x = float(-np.pi / 2.0 * np.sin(np.pi * x / 2.0) * np.exp(-0.5 * tau))
            u_xx = -((np.pi / 2.0) ** 2) * value
            drift = 0.1 + 0.02 * x
            diffusion = 0.15 + 0.03 * x * x
            forcing = u_tau - drift * u_x - diffusion * u_xx
            return u_tau - drift * u_x - diffusion * u_xx - forcing
        if self.operator_family == "mixed_derivative":
            if len(point) != 3:
                raise ValidationGateError(f"mixed-derivative point must be (x, y, tau): {point}")
            x, y, tau = point
            value = float(np.sin(np.pi * x) * np.cos(np.pi * y / 2.0) * np.exp(-tau))
            u_tau = -value
            u_xx = -(np.pi**2) * value
            u_yy = -((np.pi / 2.0) ** 2) * value
            u_xy = float(
                -np.pi * (np.pi / 2.0) * np.cos(np.pi * x) * np.sin(np.pi * y / 2.0) * np.exp(-tau)
            )
            forcing = u_tau - 0.2 * u_xx - 0.05 * u_xy - 0.15 * u_yy
            return u_tau - 0.2 * u_xx - 0.05 * u_xy - 0.15 * u_yy - forcing
        if self.operator_family == "state_dependent_reaction":
            if len(point) != 2:
                raise ValidationGateError(
                    f"state-dependent-reaction point must be (x, tau): {point}"
                )
            x, tau = point
            value = float((1.0 + x) * np.exp(-tau))
            u_tau = -value
            reaction = 0.03 + 0.02 * x
            forcing = u_tau + reaction * value
            return u_tau + reaction * value - forcing
        raise ValidationGateError(
            f"unsupported manufactured operator family: {self.operator_family}"
        )

    def to_public_dict(self) -> dict[str, Any]:
        """Return a JSON-safe manufactured-solution declaration."""

        return {
            "operator_family": self.operator_family,
            "equation": self.equation,
            "sample_points": [list(point) for point in self.sample_points],
            "residual_tolerance": self.residual_tolerance,
        }


def manufactured_solution_cases() -> dict[str, ManufacturedSolutionCase]:
    """Return deterministic manufactured cases for required operator families."""

    return {
        "diffusion": ManufacturedSolutionCase(
            operator_family="diffusion",
            equation="u_tau - sigma^2 x^2 u_xx / 2 = f",
            sample_points=((0.2, 0.1), (0.5, 0.4), (0.8, 0.9)),
            residual_tolerance=1.0e-12,
        ),
        "convection_diffusion": ManufacturedSolutionCase(
            operator_family="convection_diffusion",
            equation="u_tau - a(x)u_x - b(x)u_xx = f",
            sample_points=((0.25, 0.2), (0.75, 0.6)),
            residual_tolerance=1.0e-12,
        ),
        "mixed_derivative": ManufacturedSolutionCase(
            operator_family="mixed_derivative",
            equation="u_tau - a u_xx - 2 rho u_xy - b u_yy = f",
            sample_points=((0.2, 0.3, 0.1), (0.7, 0.4, 0.5)),
            residual_tolerance=1.0e-12,
        ),
        "state_dependent_reaction": ManufacturedSolutionCase(
            operator_family="state_dependent_reaction",
            equation="u_tau - L[u] + r(x)u = f",
            sample_points=((0.1, 0.2), (0.6, 0.7), (0.9, 0.3)),
            residual_tolerance=1.0e-12,
        ),
    }


@dataclass(frozen=True, slots=True)
class ConvergenceRow:
    """One row in a convergence study."""

    resolution: int
    step: float
    error: float

    def validate(self) -> None:
        """Validate row finiteness and positivity."""

        if self.resolution <= 0:
            raise ValidationGateError("convergence resolution must be positive")
        if not isfinite(self.step) or self.step <= 0.0:
            raise ValidationGateError("convergence step must be positive and finite")
        if not isfinite(self.error) or self.error < 0.0:
            raise ValidationGateError("convergence error must be non-negative and finite")


@dataclass(frozen=True, slots=True)
class ConvergenceReport:
    """Evaluated convergence evidence with actionable row details."""

    benchmark_id: str
    dimension: str
    accepted: bool
    expected_order: float
    observed_order: float | None
    final_error: float
    error_budget: float
    rows: tuple[dict[str, float | int | None], ...]
    failures: tuple[str, ...]

    @property
    def actionable_table(self) -> tuple[dict[str, float | int | None], ...]:
        """Return the row table to upload on failure."""

        return self.rows

    def to_public_dict(self) -> dict[str, Any]:
        """Return a JSON-safe convergence report."""

        return {
            "benchmark_id": self.benchmark_id,
            "dimension": self.dimension,
            "accepted": self.accepted,
            "expected_order": self.expected_order,
            "observed_order": self.observed_order,
            "final_error": self.final_error,
            "error_budget": self.error_budget,
            "rows": list(self.rows),
            "failures": list(self.failures),
        }


@dataclass(frozen=True, slots=True)
class ConvergenceStudy:
    """Convergence gate for one isolated discretization dimension."""

    benchmark_id: str
    dimension: str
    expected_order: float
    order_tolerance: float
    rows: tuple[ConvergenceRow, ...]
    tolerance_components: Mapping[str, float]

    def evaluate(self) -> ConvergenceReport:
        """Evaluate observed order and final error budget."""

        if len(self.rows) < 2:
            raise ValidationGateError("at least two convergence rows are required")
        if (
            not isfinite(self.expected_order)
            or self.expected_order < 0.0
            or not isfinite(self.order_tolerance)
            or self.order_tolerance < 0.0
        ):
            raise ValidationGateError(
                f"{self.benchmark_id}:{self.dimension} invalid convergence order controls"
            )
        for row in self.rows:
            row.validate()
        if any(
            component not in self.tolerance_components
            for component in REQUIRED_TOLERANCE_COMPONENTS
        ):
            raise ValidationGateError(
                f"{self.benchmark_id}:{self.dimension} missing tolerance components"
            )
        for component, value in self.tolerance_components.items():
            tolerance = float(value)
            if not isfinite(tolerance) or tolerance < 0.0:
                raise ValidationGateError(
                    f"{self.benchmark_id}:{self.dimension} invalid tolerance "
                    f"component {component!r}"
                )
        error_budget = float(sum(float(value) for value in self.tolerance_components.values()))
        row_payloads: list[dict[str, float | int | None]] = []
        observed_orders: list[float] = []
        previous: ConvergenceRow | None = None
        for row in self.rows:
            observed_order: float | None = None
            if previous is not None and row.error > 0.0 and previous.error > 0.0:
                step_ratio = previous.step / row.step
                error_ratio = previous.error / row.error
                if step_ratio > 1.0 and error_ratio > 0.0:
                    observed_order = log(error_ratio) / log(step_ratio)
                    observed_orders.append(observed_order)
            row_payloads.append(
                {
                    "resolution": row.resolution,
                    "step": row.step,
                    "error": row.error,
                    "observed_order": observed_order,
                }
            )
            previous = row
        final_error = self.rows[-1].error
        representative_order = min(observed_orders) if observed_orders else None
        failures: list[str] = []
        if final_error > error_budget:
            failures.append(
                f"final {self.dimension} error {final_error:.6g} exceeds budget {error_budget:.6g}"
            )
        if self.expected_order > 0.0:
            threshold = self.expected_order - self.order_tolerance
            if representative_order is None or representative_order < threshold:
                failures.append(
                    f"observed {self.dimension} order {representative_order} below "
                    f"threshold {threshold:.6g}"
                )
        elif any(later.error > earlier.error for earlier, later in zip(self.rows, self.rows[1:])):
            failures.append(f"{self.dimension} errors are not monotone non-increasing")
        return ConvergenceReport(
            benchmark_id=self.benchmark_id,
            dimension=self.dimension,
            accepted=not failures,
            expected_order=self.expected_order,
            observed_order=representative_order,
            final_error=final_error,
            error_budget=error_budget,
            rows=tuple(row_payloads),
            failures=tuple(failures),
        )

    def require_passed(self) -> ConvergenceReport:
        """Return the report or raise with actionable failure details."""

        report = self.evaluate()
        if not report.accepted:
            raise ValidationGateError(
                f"convergence gate failed for {self.benchmark_id}:{self.dimension}: "
                f"{report.failures}; rows={report.actionable_table}"
            )
        return report


@dataclass(frozen=True, slots=True)
class OptionSurfacePoint:
    """One option-surface point for arbitrage property gates."""

    spot: float
    strike: float
    rate: float
    maturity: float
    price: float
    delta: float
    gamma: float

    def validate(self) -> None:
        """Validate surface point finiteness and economics basics."""

        values = (
            self.spot,
            self.strike,
            self.rate,
            self.maturity,
            self.price,
            self.delta,
            self.gamma,
        )
        if not all(isfinite(value) for value in values):
            raise ValidationGateError("option surface point contains non-finite values")
        if self.spot <= 0.0 or self.strike <= 0.0 or self.maturity < 0.0:
            raise ValidationGateError("option surface point has invalid economic coordinates")


@dataclass(frozen=True, slots=True)
class ArbitrageReport:
    """Report for call-surface arbitrage checks."""

    accepted: bool
    failures: tuple[str, ...]
    rows: tuple[dict[str, float], ...]

    def to_public_dict(self) -> dict[str, Any]:
        """Return a JSON-safe arbitrage report."""

        return {"accepted": self.accepted, "failures": list(self.failures), "rows": list(self.rows)}


def evaluate_call_arbitrage(
    points: Sequence[OptionSurfacePoint], *, fail_on_error: bool = False
) -> ArbitrageReport:
    """Check finite call prices for elementary no-arbitrage properties."""

    if len(points) < 2:
        raise ValidationGateError("at least two surface points are required")
    sorted_points = tuple(sorted(points, key=lambda point: point.spot))
    failures: list[str] = []
    rows: list[dict[str, float]] = []
    contract = sorted_points[0]
    previous: OptionSurfacePoint | None = None
    previous_slope: float | None = None
    for point in sorted_points:
        point.validate()
        if (
            not np.isclose(point.strike, contract.strike, rtol=0.0, atol=1.0e-12)
            or not np.isclose(point.rate, contract.rate, rtol=0.0, atol=1.0e-12)
            or not np.isclose(point.maturity, contract.maturity, rtol=0.0, atol=1.0e-12)
        ):
            failures.append("surface points must share same strike/rate/maturity")
        discounted_strike = point.strike * float(np.exp(-point.rate * point.maturity))
        lower_bound = max(point.spot - discounted_strike, 0.0)
        upper_bound = point.spot
        if point.price + 1.0e-12 < lower_bound:
            failures.append(
                f"price {point.price:.6g} below lower no-arbitrage bound {lower_bound:.6g}"
            )
        if point.price - 1.0e-12 > upper_bound:
            failures.append(
                f"price {point.price:.6g} above upper no-arbitrage bound {upper_bound:.6g}"
            )
        if not 0.0 <= point.delta <= 1.0:
            failures.append(f"delta {point.delta:.6g} outside [0, 1]")
        if point.gamma < -1.0e-12:
            failures.append(f"gamma {point.gamma:.6g} is negative")
        if previous is not None:
            if point.price + 1.0e-12 < previous.price:
                failures.append("call price is not monotone increasing in spot")
            spot_gap = point.spot - previous.spot
            if abs(spot_gap) <= 1.0e-12:
                failures.append(f"duplicate spot coordinate {point.spot:.6g}")
            else:
                slope = (point.price - previous.price) / spot_gap
                if previous_slope is not None and slope + 1.0e-12 < previous_slope:
                    failures.append("call price is not convex in spot")
                previous_slope = slope
        rows.append(
            {
                "spot": point.spot,
                "price": point.price,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "delta": point.delta,
                "gamma": point.gamma,
            }
        )
        previous = point
    report = ArbitrageReport(accepted=not failures, failures=tuple(failures), rows=tuple(rows))
    if fail_on_error and not report.accepted:
        raise ValidationGateError(f"arbitrage gate failed: {report.failures}; rows={report.rows}")
    return report


@dataclass(frozen=True, slots=True)
class BackendValidationReport:
    """Comparable evidence from one numerical backend."""

    benchmark_id: str
    backend_id: str
    pde_convention_hash: str
    grid_hash: str
    time_schedule_hash: str
    values: Mapping[str, float]

    def validate(self) -> None:
        """Validate backend evidence before comparison."""

        if not self.benchmark_id or not self.backend_id:
            raise ValidationGateError("backend report missing identifiers")
        for field_name in ("pde_convention_hash", "grid_hash", "time_schedule_hash"):
            if not getattr(self, field_name):
                raise ValidationGateError(f"backend report missing {field_name}")
        if not self.values:
            raise ValidationGateError("backend report has no values")
        for metric, value in self.values.items():
            if not isfinite(float(value)):
                raise ValidationGateError(f"backend metric {metric!r} is non-finite")


@dataclass(frozen=True, slots=True)
class CrossBackendComparisonReport:
    """Report for same-input cross-backend parity."""

    accepted: bool
    benchmark_id: str
    left_backend_id: str
    right_backend_id: str
    max_abs_difference: float
    differences: Mapping[str, float]
    failures: tuple[str, ...]

    def to_public_dict(self) -> dict[str, Any]:
        """Return a JSON-safe cross-backend report."""

        return {
            "accepted": self.accepted,
            "benchmark_id": self.benchmark_id,
            "left_backend_id": self.left_backend_id,
            "right_backend_id": self.right_backend_id,
            "max_abs_difference": self.max_abs_difference,
            "differences": dict(self.differences),
            "failures": list(self.failures),
        }


def compare_backend_reports(
    left: BackendValidationReport,
    right: BackendValidationReport,
    *,
    tolerances: Mapping[str, float],
    fail_on_error: bool = True,
) -> CrossBackendComparisonReport:
    """Compare two backend reports on identical PDE/grid/time inputs."""

    left.validate()
    right.validate()
    failures: list[str] = []
    if left.benchmark_id != right.benchmark_id:
        failures.append("backend reports use different benchmark IDs")
    if left.pde_convention_hash != right.pde_convention_hash:
        failures.append("cross-backend parity requires identical PDE conventions")
    if left.grid_hash != right.grid_hash:
        failures.append("cross-backend parity requires identical grid conventions")
    if left.time_schedule_hash != right.time_schedule_hash:
        failures.append("cross-backend parity requires identical time schedules")
    if set(left.values) != set(right.values):
        failures.append("cross-backend parity requires identical metric sets")
    metric_names = set(left.values) | set(right.values)
    missing_tolerances = tuple(
        metric for metric in sorted(metric_names) if metric not in tolerances
    )
    if missing_tolerances:
        failures.append(f"missing tolerances for metrics {missing_tolerances}")
    for metric, tolerance_value in tolerances.items():
        tolerance = float(tolerance_value)
        if not isfinite(tolerance) or tolerance < 0.0:
            failures.append(f"invalid tolerance for metric {metric!r}")
    differences: dict[str, float] = {}
    for metric, left_value in left.values.items():
        if metric not in right.values:
            failures.append(f"right backend missing metric {metric!r}")
            continue
        difference = abs(float(left_value) - float(right.values[metric]))
        differences[metric] = difference
        tolerance = float(tolerances.get(metric, 0.0))
        if difference > tolerance:
            failures.append(
                f"metric {metric!r} difference {difference:.6g} exceeds tolerance {tolerance:.6g}"
            )
    max_abs_difference = max(differences.values(), default=0.0)
    report = CrossBackendComparisonReport(
        accepted=not failures,
        benchmark_id=left.benchmark_id,
        left_backend_id=left.backend_id,
        right_backend_id=right.backend_id,
        max_abs_difference=max_abs_difference,
        differences=differences,
        failures=tuple(failures),
    )
    if fail_on_error and not report.accepted:
        raise ValidationGateError(f"cross-backend gate failed: {report.failures}")
    return report


@dataclass(frozen=True, slots=True)
class AmericanLCPGateReport:
    """Complementarity and exercise-front report for American LCP solves."""

    accepted: bool
    benchmark_id: str
    exercise_front_observed: bool
    max_complementarity: float
    max_projected_residual: float
    failures: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]

    def to_public_dict(self) -> dict[str, Any]:
        """Return a JSON-safe American LCP report."""

        return {
            "accepted": self.accepted,
            "benchmark_id": self.benchmark_id,
            "exercise_front_observed": self.exercise_front_observed,
            "max_complementarity": self.max_complementarity,
            "max_projected_residual": self.max_projected_residual,
            "failures": list(self.failures),
            "rows": list(self.rows),
        }


def _has_exercise_front(exercise_set: Sequence[bool]) -> bool:
    """Return true only when exercise and continuation regions meet."""

    if not exercise_set:
        return False
    has_exercise = any(exercise_set)
    has_continuation = not all(exercise_set)
    has_transition = any(left != right for left, right in zip(exercise_set, exercise_set[1:]))
    return has_exercise and has_continuation and has_transition


def evaluate_american_lcp_gate(
    benchmark_id: str,
    diagnostics: Sequence[LCPDiagnostics],
    *,
    fail_on_error: bool = False,
) -> AmericanLCPGateReport:
    """Require converged complementarity and exercise-front evidence."""

    if not diagnostics:
        raise ValidationGateError("American LCP gate requires at least one diagnostic row")
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
            field for field, value in diagnostic_values.items() if not isfinite(float(value))
        )
        if nonfinite_fields:
            failures.append(f"row {index} has non-finite diagnostics {nonfinite_fields}")
        if isfinite(float(tolerance)) and tolerance <= 0.0:
            failures.append(f"row {index} tolerance must be positive")
        if item.iterations < 0 or item.exercise_count < 0:
            failures.append(f"row {index} has invalid iteration/exercise counts")
        max_complementarity = max(max_complementarity, item.complementarity_max)
        max_projected_residual = max(max_projected_residual, item.projected_residual_max)
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
    report = AmericanLCPGateReport(
        accepted=not failures,
        benchmark_id=benchmark_id,
        exercise_front_observed=exercise_front_observed,
        max_complementarity=max_complementarity,
        max_projected_residual=max_projected_residual,
        failures=tuple(failures),
        rows=tuple(rows),
    )
    if fail_on_error and not report.accepted:
        raise ValidationGateError(f"American LCP gate failed: {report.failures}")
    return report
