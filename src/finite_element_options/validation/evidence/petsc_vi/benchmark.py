"""Reproducible PETSc trigger and American-option VI assessment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
import platform
from time import perf_counter
from typing import Any, cast

import numpy as np
import scipy.sparse as sps

from finite_element_options.contracts.capability_matrix import (
    DEFAULT_CAPABILITY_RECORDS,
    CapabilityStatus,
)
from finite_element_options.core.dynamics_black_scholes import DynamicsParametersBlackScholes
from finite_element_options.core.market import Market
from finite_element_options.core.vanilla_bs import EuropeanOptionBs
from finite_element_options.space.mesh import create_mesh
from finite_element_options.space.solver import SpaceSolver
from finite_element_options.time_integration.lcp import (
    DiscreteLCP,
    LCPConvergenceError,
    ProjectedSORSolverSettings,
)
from finite_element_options.time_integration.stepper import ThetaScheme
from finite_element_options.validation.evidence.serialization import (
    distribution_install_mode,
    file_sha256,
    quantize_json_floats,
)

from .adapter import PetscVISolver, PetscVISolverSettings, petsc_runtime_doctor


SCHEMA_VERSION = "petsc-american-vi-assessment/v1"
PREDECESSOR_PATH = "docs/evidence/black_scholes_pymor_rom_2026-09-05.json"
PREDECESSOR_SHA256 = "f30d712e054937ac7e17ea452fc2bcbc0a874087b1ec180caa5e63dc190ea4b7"


@dataclass(frozen=True, slots=True)
class PetscVIAssessmentConfig:
    """Public-synthetic equal-discretization American-put comparison controls."""

    rate: float = 0.05
    volatility: float = 0.20
    maturity: float = 1.0
    strike: float = 1.0
    spot: float = 1.0
    domain_max: float = 4.0
    refinement_level: int = 8
    time_steps: int = 80
    theta: float = 1.0
    lcp_tolerance: float = 1.0e-9
    max_iterations: int = 10_000
    psor_relaxation: float = 1.0
    greek_bump: float = 0.02
    grid_abs_tolerance: float = 5.0e-7
    price_abs_tolerance: float = 2.0e-7
    delta_abs_tolerance: float = 2.0e-6
    gamma_abs_tolerance: float = 2.0e-5
    repeats: int = 3

    def __post_init__(self) -> None:
        """Reject unsupported assessment controls."""

        positive = (
            self.volatility,
            self.maturity,
            self.strike,
            self.spot,
            self.domain_max,
            self.lcp_tolerance,
            self.greek_bump,
            self.grid_abs_tolerance,
            self.price_abs_tolerance,
            self.delta_abs_tolerance,
            self.gamma_abs_tolerance,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("PETSc assessment controls must be finite and positive")
        if not np.isfinite(self.rate):
            raise ValueError("rate must be finite")
        if self.domain_max <= self.strike or not self.greek_bump < min(
            self.spot, self.domain_max - self.spot
        ):
            raise ValueError("domain and Greek bump must contain the put evaluation stencil")
        if self.refinement_level < 2 or self.time_steps < 2 or self.max_iterations < 1:
            raise ValueError("assessment discretization and iteration controls are too small")
        if not 0.5 <= self.theta <= 1.0 or not 0.0 < self.psor_relaxation < 2.0:
            raise ValueError("theta/PSOR relaxation controls are invalid")
        if self.repeats < 3:
            raise ValueError("assessment requires at least three timing repeats")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe canonical inputs."""

        return {"schema_version": SCHEMA_VERSION, **asdict(self)}


def run_petsc_vi_assessment(
    *,
    config: PetscVIAssessmentConfig | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Run trigger, real PETSc runtime, parity, failure, timing, and memory gates."""

    selected = config or PetscVIAssessmentConfig()
    predecessor = _predecessor(root)
    trigger = _trigger_evidence()
    install_mode = distribution_install_mode("finite-element-options")
    doctor = petsc_runtime_doctor()
    space = _space(selected)
    times = np.linspace(0.0, selected.maturity, selected.time_steps + 1)
    european = ThetaScheme(theta=selected.theta).solve(times, space, is_american=False)
    psor = _solve_psor(selected, space, times)
    petsc = _solve_petsc(selected, space, times)
    errors = _parity_errors(
        selected,
        space,
        european,
        psor["values"],
        petsc["values"],
    )
    timing = _timings(selected)
    failure = _failure_evidence(selected)
    memory = _memory_evidence(space, petsc["backend"])
    residual_passed = (
        psor["diagnostics"]["projected_residual_max"] <= selected.lcp_tolerance
        and petsc["diagnostics"]["projected_residual_max"] <= selected.lcp_tolerance
    )
    parity_passed = (
        errors["grid_max_abs"] <= selected.grid_abs_tolerance
        and errors["price_abs"] <= selected.price_abs_tolerance
        and errors["delta_abs"] <= selected.delta_abs_tolerance
        and errors["gamma_abs"] <= selected.gamma_abs_tolerance
    )
    checks = {
        "predecessor_verified": bool(predecessor["verified"]),
        "installed_wheel": install_mode == "wheel",
        "american_vi_trigger": bool(trigger["triggered"]),
        "runtime_ksp_snes_vi_ts": bool(doctor["passed"]),
        "equal_discretization_parity": parity_passed,
        "american_dominates_european": bool(
            errors["projected_sor_early_exercise_premium"] >= -selected.price_abs_tolerance
            and errors["petsc_early_exercise_premium"] >= -selected.price_abs_tolerance
        ),
        "canonical_residuals": residual_passed,
        "typed_failure": bool(failure["passed"]),
        "single_rank_only": doctor["comm_size"] == 1,
    }
    promoted = all(checks.values())
    payload = {
        "schema_version": SCHEMA_VERSION,
        "privacy_class": "public_synthetic",
        "scope": (
            "Single-rank petsc4py SNES-VI solve adapter over scikit-fem/SciPy assembly; "
            "not distributed FEM assembly or production American-option qualification."
        ),
        "input": selected.to_dict(),
        "predecessor": predecessor,
        "trigger": trigger,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": version("numpy"),
            "scipy": version("scipy"),
            "scikit_fem": version("scikit-fem"),
            "finite_element_options_install_mode": install_mode,
            "petsc4py": doctor["petsc4py_version"],
            "petsc": doctor["petsc_version"],
            "comm_size": doctor["comm_size"],
            "scalar_type": doctor["scalar_type"],
        },
        "runtime_doctor": doctor,
        "equal_discretization": {
            "mesh_dofs": int(space.Vh.N),
            "time_steps": selected.time_steps,
            "theta": selected.theta,
            "matrix_assembly_owner": "scikit-fem/SciPy",
            "solve_owner": "projected-SOR or PETSc SNES-VI",
        },
        "projected_sor": psor["diagnostics"],
        "petsc_snes_vi": {**petsc["diagnostics"], "backend": petsc["backend"]},
        "parity_errors": errors,
        "timing": timing,
        "memory": memory,
        "failure_evidence": failure,
        "decision": {
            "status": "promote_external_single_rank_vi_adapter" if promoted else "defer_petsc",
            "promoted": promoted,
            "checks": checks,
            "scipy_remains_canonical": True,
            "capability_matrix_upgrade": False,
            "distributed_assembly_claim": False,
        },
    }
    normalized = quantize_json_floats(payload, significant_digits=10)
    if not isinstance(normalized, dict):  # pragma: no cover - payload is a dict
        raise TypeError("PETSc assessment serialization must return a mapping")
    return normalized


def _trigger_evidence() -> dict[str, object]:
    capability_id = "FEM-AMERICAN-LCP-REFERENCE"
    record = next(
        (item for item in DEFAULT_CAPABILITY_RECORDS if item.capability_id == capability_id),
        None,
    )
    triggered = bool(
        record is not None
        and record.status is CapabilityStatus.VALIDATED
        and "tests/test_american_lcp.py" in record.evidence_ids
    )
    return {
        "triggered": triggered,
        "capability_id": capability_id,
        "capability_status": None if record is None else record.status.value,
        "evidence_ids": [] if record is None else list(record.evidence_ids),
        "reason": "existing validated lower-obstacle route" if triggered else "trigger absent",
        "canonical_scipy_solver": "projected_sor",
    }


def _space(config: PetscVIAssessmentConfig) -> SpaceSolver:
    mesh, finite_element_config = create_mesh([config.domain_max], config.refinement_level)
    mesh = mesh.with_boundaries(
        {
            "left": lambda x: np.isclose(x[0], 0.0),
            "right": lambda x: np.isclose(x[0], config.domain_max),
        }
    )
    dynamics = DynamicsParametersBlackScholes(r=config.rate, q=0.0, sig=config.volatility)
    option = EuropeanOptionBs(k=config.strike, q=0.0, mkt=Market(r=config.rate))
    return SpaceSolver(mesh, dynamics, option, is_call=False, config=finite_element_config)


def _solve_psor(
    config: PetscVIAssessmentConfig,
    space: SpaceSolver,
    times: np.ndarray,
) -> dict[str, Any]:
    stepper = ThetaScheme(
        theta=config.theta,
        lcp_solver_settings=ProjectedSORSolverSettings(
            tolerance=config.lcp_tolerance,
            max_iterations=config.max_iterations,
            relaxation=config.psor_relaxation,
        ),
    )
    started = perf_counter()
    values = stepper.solve(times, space, is_american=True)
    elapsed = perf_counter() - started
    return {
        "values": values,
        "diagnostics": _aggregate_diagnostics(stepper.last_lcp_diagnostics, elapsed),
    }


def _solve_petsc(
    config: PetscVIAssessmentConfig,
    space: SpaceSolver,
    times: np.ndarray,
) -> dict[str, Any]:
    solver = PetscVISolver(
        PetscVISolverSettings(
            tolerance=config.lcp_tolerance,
            max_iterations=config.max_iterations,
        )
    )
    stepper = ThetaScheme(theta=config.theta, lcp_solver=solver)
    started = perf_counter()
    values = stepper.solve(times, space, is_american=True)
    elapsed = perf_counter() - started
    return {
        "values": values,
        "diagnostics": _aggregate_diagnostics(stepper.last_lcp_diagnostics, elapsed),
        "backend": dict(solver.last_backend_evidence),
    }


def _aggregate_diagnostics(rows: list[Any], elapsed: float) -> dict[str, Any]:
    return {
        "success": bool(rows and all(row.success for row in rows)),
        "solver": rows[-1].solver,
        "solve_count": len(rows),
        "total_iterations": sum(row.iterations for row in rows),
        "max_iterations": max(row.iterations for row in rows),
        "total_linear_iterations": sum(row.linear_iterations for row in rows),
        "projected_residual_max": max(row.projected_residual_max for row in rows),
        "primal_violation_max": max(row.primal_violation_max for row in rows),
        "dual_violation_max": max(row.dual_violation_max for row in rows),
        "complementarity_max": max(row.complementarity_max for row in rows),
        "exercise_count_final": rows[-1].exercise_count,
        "elapsed_seconds": elapsed,
        "backend_reason_final": rows[-1].backend_reason,
    }


def _output_values(
    config: PetscVIAssessmentConfig, space: SpaceSolver, values: np.ndarray
) -> np.ndarray:
    points = np.array(
        [[config.spot - config.greek_bump, config.spot, config.spot + config.greek_bump]]
    )
    prices = np.asarray(space.Vh.probes(points) @ values, dtype=float)
    delta = (prices[2] - prices[0]) / (2.0 * config.greek_bump)
    gamma = (prices[2] - 2.0 * prices[1] + prices[0]) / config.greek_bump**2
    return np.array([prices[1], delta, gamma])


def _parity_errors(
    config: PetscVIAssessmentConfig,
    space: SpaceSolver,
    european: np.ndarray,
    psor: np.ndarray,
    petsc: np.ndarray,
) -> dict[str, float]:
    european_outputs = _output_values(config, space, european[-1])
    psor_outputs = _output_values(config, space, psor[-1])
    petsc_outputs = _output_values(config, space, petsc[-1])
    return {
        "grid_max_abs": float(np.max(np.abs(psor[-1] - petsc[-1]))),
        "price_abs": float(abs(psor_outputs[0] - petsc_outputs[0])),
        "delta_abs": float(abs(psor_outputs[1] - petsc_outputs[1])),
        "gamma_abs": float(abs(psor_outputs[2] - petsc_outputs[2])),
        "projected_sor_price": float(psor_outputs[0]),
        "petsc_price": float(petsc_outputs[0]),
        "projected_sor_delta": float(psor_outputs[1]),
        "petsc_delta": float(petsc_outputs[1]),
        "projected_sor_gamma": float(psor_outputs[2]),
        "petsc_gamma": float(petsc_outputs[2]),
        "european_price": float(european_outputs[0]),
        "projected_sor_early_exercise_premium": float(psor_outputs[0] - european_outputs[0]),
        "petsc_early_exercise_premium": float(petsc_outputs[0] - european_outputs[0]),
    }


def _timings(config: PetscVIAssessmentConfig) -> dict[str, Any]:
    psor_samples: list[float] = []
    petsc_samples: list[float] = []
    for repeat in range(config.repeats):
        pair = [("psor", psor_samples), ("petsc", petsc_samples)]
        if repeat % 2:
            pair.reverse()
        for backend, destination in pair:
            space = _space(config)
            times = np.linspace(0.0, config.maturity, config.time_steps + 1)
            started = perf_counter()
            if backend == "psor":
                _solve_psor(config, space, times)
            else:
                _solve_petsc(config, space, times)
            destination.append(perf_counter() - started)
    psor = _timing_summary(psor_samples)
    petsc = _timing_summary(petsc_samples)
    return {
        "clock": "time.perf_counter",
        "warmup_policy": "untimed initial parity solve for each backend",
        "backend_order": "alternated by repeat",
        "repeats": config.repeats,
        "projected_sor": psor,
        "petsc": petsc,
        "petsc_over_psor_runtime_ratio": petsc["median_seconds"] / psor["median_seconds"],
    }


def _timing_summary(samples: list[float]) -> dict[str, Any]:
    values = np.asarray(samples, dtype=float)
    center = float(np.median(values))
    return {
        "samples_seconds": samples,
        "median_seconds": center,
        "mad_seconds": float(np.median(np.abs(values - center))),
        "p05_seconds": float(np.quantile(values, 0.05)),
        "p95_seconds": float(np.quantile(values, 0.95)),
    }


def _failure_evidence(config: PetscVIAssessmentConfig) -> dict[str, Any]:
    problem = DiscreteLCP(
        matrix=sps.csr_matrix([[1.0, -1.0], [-1.0, 1.0]]),
        rhs=np.array([1.0, 1.0]),
        obstacle=np.array([0.0, 0.0]),
    )
    solver = PetscVISolver(
        PetscVISolverSettings(tolerance=min(config.lcp_tolerance, 1.0e-12), max_iterations=2)
    )
    result = solver.solve(problem, fail_on_nonconvergence=False)
    typed_exception_caught = False
    typed_exception_reason = ""
    try:
        solver.solve(problem)
    except LCPConvergenceError as exc:
        typed_exception_caught = True
        typed_exception_reason = exc.diagnostics.backend_reason
    return {
        "passed": (
            not result.success
            and result.diagnostics.backend_reason.startswith("SNES_DIVERGED")
            and typed_exception_caught
        ),
        "typed_exception": "LCPConvergenceError",
        "typed_exception_caught": typed_exception_caught,
        "typed_exception_backend_reason": typed_exception_reason,
        "backend_reason": result.diagnostics.backend_reason,
        "projected_residual_max": result.diagnostics.projected_residual_max,
    }


def _memory_evidence(space: SpaceSolver, backend: dict[str, object]) -> dict[str, Any]:
    matrices = [sps.csc_matrix(space.mass), sps.csc_matrix(space.stiffness)]
    scipy_bytes = sum(
        matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes for matrix in matrices
    )
    return {
        "scipy_assembled_mass_and_operator_bytes": int(scipy_bytes),
        "petsc_last_step_csr_transfer_bytes": cast(int, backend["matrix_csr_input_bytes"]),
        "petsc_last_step_mat_reported_memory_bytes": cast(float, backend["matrix_memory_bytes"]),
        "petsc_last_step_vector_payload_estimate_bytes": int(5 * space.Vh.N * 8),
        "measurement_scope": "allocated sparse buffers/PETSc Mat info/vector payload; not process RSS",
    }


def _predecessor(root: str | Path | None) -> dict[str, Any]:
    if root is None:
        return {
            "path": PREDECESSOR_PATH,
            "expected_sha256": PREDECESSOR_SHA256,
            "observed_sha256": None,
            "verified": False,
        }
    path = Path(root) / PREDECESSOR_PATH
    if not path.is_file():
        raise ValueError(f"missing predecessor artifact: {PREDECESSOR_PATH}")
    observed = file_sha256(path)
    if observed != PREDECESSOR_SHA256:
        raise ValueError("PETSc predecessor artifact hash mismatch")
    return {
        "path": PREDECESSOR_PATH,
        "expected_sha256": PREDECESSOR_SHA256,
        "observed_sha256": observed,
        "verified": True,
    }


__all__ = ["PetscVIAssessmentConfig", "run_petsc_vi_assessment"]
