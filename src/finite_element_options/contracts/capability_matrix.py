"""Machine-readable capability and maturity registry for FEM docs.

The records in this module are the source of truth for public README capability
claims. They deliberately separate code existence from validation and production
maturity so downstream consumers can distinguish implemented examples from
routes backed by benchmark or release evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CapabilityStatus(str, Enum):
    """Public maturity levels for advertised FEM capabilities."""

    SCAFFOLD = "scaffold"
    EXPERIMENTAL = "experimental"
    IMPLEMENTED = "implemented"
    VALIDATED = "validated"
    PRODUCTION = "production-qualified"
    DEPRECATED = "deprecated"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class CapabilityRecord:
    """One advertised capability with evidence and limitations."""

    capability_id: str
    title: str
    status: CapabilityStatus
    workstream: str
    summary: str
    evidence_ids: tuple[str, ...]
    evidence_scope: str
    limitations: str
    benchmark_ids: tuple[str, ...] = ()
    reference_ids: tuple[str, ...] = ()
    optional_extra: str | None = None
    absence_behavior: str = ""


DEFAULT_CAPABILITY_RECORDS: tuple[CapabilityRecord, ...] = (
    CapabilityRecord(
        capability_id="FEM-BS-1D-EUROPEAN",
        title="1D European Black-Scholes FEM route",
        status=CapabilityStatus.VALIDATED,
        workstream="Numerics",
        summary=(
            "Line-mesh Black-Scholes call/put examples with explicit volatility "
            "versus variance APIs and analytical-oracle parity checks."
        ),
        evidence_ids=(
            "tests/test_black_scholes_1d.py",
            "tests/test_benchmark_black_scholes.py",
        ),
        evidence_scope=(
            "Black-Scholes, one spatial dimension, European exercise, "
            "line-uniform scikit-fem route."
        ),
        limitations=(
            "Does not validate American exercise, Heston calibration, or "
            "multi-dimensional production claims."
        ),
        benchmark_ids=("pytest-benchmark:black_scholes_benchmark",),
        reference_ids=("EuropeanOptionBs analytical oracle",),
    ),
    CapabilityRecord(
        capability_id="FEM-THETA-TIME-GRID",
        title="Theta-family time integration",
        status=CapabilityStatus.VALIDATED,
        workstream="Numerics",
        summary=(
            "Finite increasing grids, nonuniform local steps, endpoint-aware "
            "boundary/source timing, and Rannacher startup diagnostics."
        ),
        evidence_ids=(
            "tests/test_time_stepper.py",
            "tests/test_state_time_coefficients.py",
        ),
        evidence_scope="Theta stepping mechanics and coefficient refresh behavior.",
        limitations=(
            "Adaptive time stepping, large-scale PETSc variational inequalities, "
            "and financial-product validation remain separate capabilities."
        ),
        benchmark_ids=("FEM-THETA-TIME-GRID",),
        reference_ids=("FR-FEM-006",),
    ),
    CapabilityRecord(
        capability_id="FEM-AMERICAN-LCP-REFERENCE",
        title="Reference American lower-obstacle LCP solve",
        status=CapabilityStatus.VALIDATED,
        workstream="Numerics",
        summary=(
            "Projected-SOR lower-obstacle LCP stepping for American exercise, "
            "with primal, dual, complementarity, exercise-set, nonconvergence, "
            "and Rannacher schedule diagnostics."
        ),
        evidence_ids=("tests/test_american_lcp.py",),
        evidence_scope=(
            "Discrete sparse lower-obstacle LCP systems coupled into the theta "
            "stepper; verifies complementarity residuals and explicit failure."
        ),
        limitations=(
            "This is the base SciPy reference solver, not a PETSc/SNES VI route "
            "or a full American-product benchmark suite against QuantLib."
        ),
        benchmark_ids=("FEM-AMERICAN-LCP-REFERENCE",),
        reference_ids=("FEM-AMERICAN-LCP", "issue-41"),
    ),
    CapabilityRecord(
        capability_id="FEM-VALIDATION-GATES-V0",
        title="Convergence, arbitrage, manufactured-solution and backend gates",
        status=CapabilityStatus.VALIDATED,
        workstream="Validation",
        summary=(
            "Executable benchmark registry, separated error-budget convergence "
            "studies, manufactured residual canaries, call arbitrage checks, "
            "cross-backend parity, and American LCP complementarity gates."
        ),
        evidence_ids=("tests/test_validation_gates.py",),
        evidence_scope=(
            "Numerical verification meta-gates for production/validated claims; "
            "failure reports carry actionable row tables."
        ),
        limitations=(
            "The default suite is deterministic and lightweight; large external "
            "QuantLib/FEniCSx work-precision studies remain separate evidence."
        ),
        benchmark_ids=("FEM-VALIDATION-GATES-V0",),
        reference_ids=("issue-42",),
    ),
    CapabilityRecord(
        capability_id="FEM-SOLVER-CACHE-001",
        title="SciPy direct sparse factorization reuse",
        status=CapabilityStatus.VALIDATED,
        workstream="Performance",
        summary=(
            "Invariant one-dimensional theta systems assemble and factor once, "
            "then reuse sparse LU across repeated right-hand sides."
        ),
        evidence_ids=("tests/test_solver_cache_benchmark.py",),
        evidence_scope=("Repeated 1D line-uniform SciPy-direct solves with residual checks."),
        limitations=(
            "Banded, AMG, PETSc and equal-error work-precision routes remain "
            "fail-closed until separately evidenced."
        ),
        benchmark_ids=("FEM-SOLVER-CACHE-001",),
    ),
    CapabilityRecord(
        capability_id="FEM-HC-SOLVER-CONTRACT-V0.1",
        title="Released public FEM solver contract",
        status=CapabilityStatus.PRODUCTION,
        workstream="Solver Contracts",
        summary=(
            "Public-synthetic solver contract and fail-closed route diagnostics "
            "for downstream Haircut Engine compatibility checks."
        ),
        evidence_ids=(
            "tests/test_fem_backend_capabilities.py",
            "tests/fixtures/fem_bs_001/",
        ),
        evidence_scope=(
            "Public synthetic vanilla-call contract, capability manifest, "
            "and unsupported-route diagnostics."
        ),
        limitations=(
            "Contract maturity does not imply every model/backend combination "
            "is production-qualified."
        ),
        benchmark_ids=("fem-bs-001",),
        reference_ids=("finite-element-options-fem-solver-contract-v0.1",),
    ),
    CapabilityRecord(
        capability_id="PINARES-FEM-FIXED-PRICE-PROXY-V0",
        title="Pinares fixed-price weak-form proxy fixture",
        status=CapabilityStatus.VALIDATED,
        workstream="Solver Contracts",
        summary=(
            "Public-synthetic UF-denominated Pinares proxy fixture with "
            "survival-scaled payoff and full-deal fail-closed diagnostics."
        ),
        evidence_ids=(
            "tests/test_pinares_fem_proxy.py",
            "tests/fixtures/fem_pinares_fixed_price_proxy_v1/",
        ),
        evidence_scope="Numerical compatibility proxy, not a family-contract valuation.",
        limitations=(
            "ROFR, legal coordination, taxes, HJB controls, obstacles and "
            "liquidity/default jumps intentionally fail closed."
        ),
        benchmark_ids=("PINARES-FEM-FIXED-PRICE-PROXY-V0",),
    ),
    CapabilityRecord(
        capability_id="FEM-Heston-CIR-MOMENTS",
        title="Heston/CIR moment diagnostics",
        status=CapabilityStatus.IMPLEMENTED,
        workstream="Numerics",
        summary=(
            "Shared finite-time CIR moment helpers for Heston boundaries and "
            "conservative variance-domain tail diagnostics."
        ),
        evidence_ids=("tests/test_heston_moments.py",),
        evidence_scope="Moment formulas and limiting cases for Heston variance domains.",
        limitations=(
            "Moment/domain diagnostics are not a full Heston PDE validation or calibration proof."
        ),
        reference_ids=("FR-FEM-004",),
    ),
    CapabilityRecord(
        capability_id="FEM-ADAPTIVE-REFINE-TRANSFER",
        title="Adaptive mesh refinement with transfer diagnostics",
        status=CapabilityStatus.IMPLEMENTED,
        workstream="Numerics",
        summary=(
            "Refinement preserves domain measure and metadata, transfers nodal "
            "values, and keeps coarsening disabled until coverage is proved."
        ),
        evidence_ids=("tests/test_solver.py",),
        evidence_scope="Refine-with-transfer mechanics and topology safety checks.",
        limitations=(
            "Goal-oriented estimators, reversible coarsening and convergence "
            "effectivity are not yet production claims."
        ),
        reference_ids=("FR-FEM-008",),
    ),
    CapabilityRecord(
        capability_id="FEM-FENICSX-EXPERIMENTAL",
        title="Optional FEniCSx backend spike",
        status=CapabilityStatus.EXPERIMENTAL,
        workstream="Backend",
        summary=(
            "DOLFINx/PETSc boundary-condition contract tests and optional "
            "Black-Scholes parity smoke where a FEniCSx environment exists."
        ),
        evidence_ids=("tests/test_fenics_solver.py", "tests/test_benchmark_fenics.py"),
        evidence_scope="Optional backend API contract and skip-visible smoke tests.",
        limitations=(
            "Not part of the base install and not advertised as a production solver route."
        ),
        optional_extra="external FEniCSx environment",
        absence_behavior="Tests skip explicitly when dolfinx/petsc4py are unavailable.",
    ),
    CapabilityRecord(
        capability_id="FEM-FD-COMPAT-DEPRECATED",
        title="Finite-difference compatibility shim",
        status=CapabilityStatus.DEPRECATED,
        workstream="Compatibility",
        summary=(
            "Legacy FD reference route retained only for transition benchmarks "
            "and parity checks under the fd optional profile."
        ),
        evidence_ids=("tests/test_fd_black_scholes.py",),
        evidence_scope="Compatibility warnings and Black-Scholes parity smoke.",
        limitations=(
            "Production finite-difference ownership belongs to "
            "finite_difference_options; removal target is published."
        ),
        optional_extra="fd",
        absence_behavior="The base wheel does not import findiff/pandas/xarray.",
    ),
    CapabilityRecord(
        capability_id="FEM-JAX-GREEKS-EXPERIMENTAL",
        title="Optional JAX analytical Greek helper",
        status=CapabilityStatus.EXPERIMENTAL,
        workstream="Sensitivities",
        summary=(
            "Analytical Black-Scholes Greek helper with method/object "
            "diagnostics and synchronized JAX timing metadata."
        ),
        evidence_ids=("tests/test_jax_greeks.py",),
        evidence_scope="Analytical helper diagnostics, not AD through FEM assembly.",
        limitations=(
            "Does not differentiate the FEM grid solution; numerical-solution "
            "sensitivities are separate capabilities."
        ),
        optional_extra="jax",
        absence_behavior="The base wheel does not import JAX; optional profile gates import it.",
    ),
    CapabilityRecord(
        capability_id="FEM-CALIBRATION-RESEARCH",
        title="Bounded pricing calibration adapters",
        status=CapabilityStatus.EXPERIMENTAL,
        workstream="Calibration",
        summary=(
            "SciPy pricing calibration, Statsmodels compatibility and PyMC "
            "research helpers are isolated behind the calibration extra, with "
            "bounded weighted residual objectives, holdout diagnostics, Heston "
            "parameter constraints and validated-engine provenance."
        ),
        evidence_ids=("tests/test_calibrator.py",),
        evidence_scope=(
            "Adapter smoke, optional-profile import boundaries, bounded pricing "
            "least-squares, bid-ask and vega weighting, robust loss metadata, "
            "holdout RMSE, deterministic multi-start diagnostics, Heston posterior "
            "constraint checks, Feller policy, MCMC diagnostics, and provenance "
            "retention for supplied PyMC/pricing artifacts."
        ),
        limitations=(
            "The package still does not ship a production Heston PDE/Fourier "
            "pricing engine; Heston calibration requires an injected, separately "
            "validated engine artifact and fails closed for toy/synthetic/polynomial "
            "engines."
        ),
        optional_extra="calibration",
        absence_behavior="The base wheel does not import PyMC, ArviZ or Statsmodels.",
    ),
    CapabilityRecord(
        capability_id="FEM-STREAMLIT-UI-EXPERIMENTAL",
        title="Streamlit exploration UI",
        status=CapabilityStatus.EXPERIMENTAL,
        workstream="UI/API",
        summary=(
            "Installed-package Streamlit entry point for exploratory demos, "
            "kept outside core FEM dependencies."
        ),
        evidence_ids=(
            ".github/workflows/ci.yml#optional_imports-ui",
            "tests/test_ui_config.py",
        ),
        evidence_scope=(
            "Optional-profile import gate plus pure-Python validation for route "
            "gating, shareable config, analytical limits and work estimates."
        ),
        limitations=(
            "The Streamlit surface remains exploratory; Heston routes and "
            "product-level American benchmarks fail closed until their numerical "
            "capabilities land."
        ),
        optional_extra="ui",
        absence_behavior="The base wheel does not import Streamlit or UI-only plotting packages.",
    ),
)


def public_capability_records() -> tuple[CapabilityRecord, ...]:
    """Return the ordered public capability registry."""

    return DEFAULT_CAPABILITY_RECORDS
