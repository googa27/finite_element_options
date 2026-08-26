"""Benchmark registry declarations for FEM validation evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any

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
            benchmark_id="VQPW-FEM-COMPILED-BS-CALL-V0",
            model="compiled FPF pde_ir.v0 Black-Scholes",
            instrument="European call compiled weak-form fixture",
            state_convention=(
                "Q measure, USD money-market numeraire, backward source PDE mapped to forward tau"
            ),
            domain="public synthetic one-dimensional spot interval [0, 400]",
            grid="line_uniform lagrange_p2 refinements 4/5/6",
            time_schedule="theta_crank_nicolson, 80 time steps",
            oracle=("closed-form Black-Scholes price, Delta and Gamma plus exact compiler hashes"),
            norm="screening acceptance plus price/Delta/Gamma absolute and relative error budget",
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
