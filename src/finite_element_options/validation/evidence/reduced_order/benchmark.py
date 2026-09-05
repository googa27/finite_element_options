"""Evidence-gated pyMOR promotion benchmark orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import scipy.sparse.linalg as spla  # type: ignore[import-untyped]

from finite_element_options.validation.evidence.serialization import (
    file_sha256,
    quantize_json_floats,
)

from .black_scholes import (
    AffineBlackScholesSystem,
    TrainedPymorROM,
    build_affine_black_scholes_system,
    train_pymor_rom,
)
from .contracts import OptionOutputs, PymorBlackScholesConfig, ROMEnvelopeError, SCHEMA_VERSION
from .performance import (
    TimingSummary,
    amortization_evidence,
    benchmark_online,
    environment_evidence,
    memory_evidence,
)


PREDECESSOR_PATH = "docs/evidence/regime_switching_quanto_openturns_uq_2026-09-04.json"
PREDECESSOR_SHA256 = "d488ea1d2300b3cd1da882479a5a475b22732145335ca3e4a3abd4393e80463f"
DECISION_POLICY = (
    "Promote pyMOR only as an optional POD/Galerkin adapter when an exact affine "
    "decomposition, holdout price/Greek tolerances, fail-closed envelope, median "
    "online speedup >=10x, and finite <=1000-query 10x amortization all pass."
)


@dataclass(frozen=True, slots=True)
class HoldoutEvaluation:
    """FOM, ROM, oracle, and error evidence at one unseen volatility."""

    volatility: float
    full_order: OptionOutputs
    reduced_order: OptionOutputs
    analytical_oracle: OptionOutputs
    rom_fom_errors: OptionOutputs
    fom_oracle_errors: OptionOutputs
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe holdout evidence."""

        return {
            "volatility": self.volatility,
            "full_order": self.full_order.to_dict(),
            "reduced_order": self.reduced_order.to_dict(),
            "analytical_oracle": self.analytical_oracle.to_dict(),
            "rom_fom_errors": self.rom_fom_errors.to_dict(),
            "fom_oracle_errors": self.fom_oracle_errors.to_dict(),
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class PymorBenchmarkReport:
    """Complete public-synthetic pyMOR adoption evidence."""

    config: PymorBlackScholesConfig
    predecessor: dict[str, Any]
    environment: dict[str, Any]
    decomposition: dict[str, Any]
    offline: dict[str, Any]
    holdouts: tuple[HoldoutEvaluation, ...]
    timing: dict[str, Any]
    memory: dict[str, int]
    envelope_refusals: tuple[dict[str, Any], ...]
    decision: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return normalized JSON evidence; timings remain observations, not golden bytes."""

        payload = {
            "schema_version": SCHEMA_VERSION,
            "privacy_class": "public_synthetic",
            "scope": (
                "Variance-parametric 1D European-call FEM POD/Galerkin spike; "
                "not a production capability or a multi-factor ROM validation."
            ),
            "study_input": self.config.to_input_dict(),
            "study_input_hash": self.config.input_hash,
            "predecessor": self.predecessor,
            "environment": self.environment,
            "decomposition": self.decomposition,
            "offline": self.offline,
            "holdouts": [row.to_dict() for row in self.holdouts],
            "timing": self.timing,
            "memory": self.memory,
            "envelope_refusals": list(self.envelope_refusals),
            "decision_policy": DECISION_POLICY,
            "decision": self.decision,
        }
        normalized = quantize_json_floats(payload, significant_digits=10)
        if not isinstance(normalized, dict):  # pragma: no cover - payload is a dict
            raise TypeError("benchmark serialization must return a mapping")
        return normalized


def run_pymor_benchmark(
    *,
    config: PymorBlackScholesConfig | None = None,
    root: str | Path | None = None,
) -> PymorBenchmarkReport:
    """Run affine validation, pyMOR training, holdout accuracy, and speed gates."""

    selected = config or PymorBlackScholesConfig()
    predecessor = _verify_predecessor(root)
    started = perf_counter()
    system = build_affine_black_scholes_system(selected)
    system_build_seconds = perf_counter() - started
    started = perf_counter()
    decomposition = _validate_affine_decomposition(system)
    affine_validation_seconds = perf_counter() - started
    trained = train_pymor_rom(system, selected)
    holdouts = _evaluate_holdouts(system, trained)
    timing = benchmark_online(system, trained)
    refusals = _verify_envelope_refusal(trained)
    offline_total = system_build_seconds + affine_validation_seconds + trained.offline_seconds
    amortization = amortization_evidence(timing, offline_total, selected)
    timing.update(amortization)
    offline = {
        "system_build_seconds": system_build_seconds,
        "affine_validation_seconds": affine_validation_seconds,
        "training_fom_seconds": trained.training_seconds,
        "adapter_setup_seconds": trained.projection.setup_seconds,
        "pod_seconds": trained.projection.pod_seconds,
        "projection_seconds": trained.projection.projection_seconds,
        "total_seconds": offline_total,
        "training_solves": len(selected.training_volatilities),
        "snapshot_count": trained.snapshot_count,
        "basis_size": trained.basis_size,
        "captured_snapshot_energy_fraction": trained.projection.captured_energy_fraction,
    }
    memory = memory_evidence(system, trained)
    checks = {
        "predecessor_verified": bool(predecessor["verified"]),
        "affine_decomposition": bool(decomposition["passed"]),
        "holdout_accuracy": all(row.passed for row in holdouts),
        "out_of_envelope_refusal": len(refusals) == 2 and all(row["passed"] for row in refusals),
        "median_online_speedup": timing["median_online_speedup"] >= selected.minimum_online_speedup,
        "ten_x_amortization": timing["ten_x_amortization_solve_count"] is not None
        and timing["ten_x_amortization_solve_count"] <= selected.maximum_ten_x_amortization_solves,
    }
    promoted = all(checks.values())
    decision = {
        "status": "promote_optional_adapter" if promoted else "reject_promotion",
        "promoted": promoted,
        "checks": checks,
        "fallback": "full_order_fem",
        "capability_matrix_upgrade": False,
    }
    return PymorBenchmarkReport(
        config=selected,
        predecessor=predecessor,
        environment=environment_evidence(trained),
        decomposition=decomposition,
        offline=offline,
        holdouts=holdouts,
        timing=timing,
        memory=memory,
        envelope_refusals=refusals,
        decision=decision,
    )


def verify_pymor_benchmark(
    reference: dict[str, Any],
    fresh: PymorBenchmarkReport,
) -> dict[str, Any]:
    """Verify semantic replay while treating timings as noisy measured evidence."""

    observed = fresh.to_dict()
    exact = {
        "schema_version": observed["schema_version"] == reference.get("schema_version"),
        "study_input_hash": observed["study_input_hash"] == reference.get("study_input_hash"),
        "decomposition_hash": observed["decomposition"]["hash"]
        == reference.get("decomposition", {}).get("hash"),
        "library_major_minor": _major_minor(observed["environment"]["pymor"])
        == _major_minor(reference.get("environment", {}).get("pymor", "")),
        "decision": observed["decision"]["status"] == reference.get("decision", {}).get("status"),
    }
    gates = {
        "fresh_decision_promoted": bool(observed["decision"]["promoted"]),
        "fresh_accuracy_passed": bool(observed["decision"]["checks"]["holdout_accuracy"]),
        "fresh_speed_passed": bool(observed["decision"]["checks"]["median_online_speedup"]),
        "fresh_amortization_passed": bool(observed["decision"]["checks"]["ten_x_amortization"]),
    }
    return {"passed": all(exact.values()) and all(gates.values()), "exact": exact, "gates": gates}


def _validate_affine_decomposition(system: AffineBlackScholesSystem) -> dict[str, Any]:
    config = system.config
    sample = tuple(
        sorted(
            {
                round(value, 12)
                for value in (
                    config.training_volatilities
                    + config.holdout_volatilities
                    + ((config.volatility_min + config.volatility_max) / 2.0,)
                )
            }
        )
    )
    errors: dict[str, float] = {}
    for volatility in sample:
        direct = system.assemble_direct_operator(volatility)
        affine = system.assemble_affine_operator(volatility)
        denominator = float(spla.norm(direct))
        errors[str(volatility)] = float(spla.norm(direct - affine) / denominator)
    maximum = max(errors.values())
    return {
        "parameterization": "eta=volatility**2",
        "operator_formula": "K(eta)=K_constant+eta*K_variance",
        "boundary_policy": system.boundary_policy,
        "fixed_mesh_and_time_grid": True,
        "sample_relative_errors": errors,
        "maximum_relative_error": maximum,
        "tolerance": config.affine_relative_tolerance,
        "passed": maximum <= config.affine_relative_tolerance,
        "hash": system.decomposition_hash,
    }


def _evaluate_holdouts(
    system: AffineBlackScholesSystem,
    trained: TrainedPymorROM,
) -> tuple[HoldoutEvaluation, ...]:
    config = system.config
    rows: list[HoldoutEvaluation] = []
    for volatility in config.holdout_volatilities:
        full = system.solve_full_order(volatility).outputs
        reduced = trained.solve(volatility)
        oracle = system.analytical_outputs(volatility)
        rom_errors = _absolute_errors(reduced, full)
        oracle_errors = _absolute_errors(full, oracle)
        passed = (
            rom_errors.price <= config.price_abs_tolerance
            and rom_errors.delta <= config.delta_abs_tolerance
            and rom_errors.gamma <= config.gamma_abs_tolerance
            and oracle_errors.price <= config.fom_oracle_price_tolerance
            and oracle_errors.delta <= config.fom_oracle_delta_tolerance
            and oracle_errors.gamma <= config.fom_oracle_gamma_tolerance
        )
        rows.append(
            HoldoutEvaluation(
                volatility=volatility,
                full_order=full,
                reduced_order=reduced,
                analytical_oracle=oracle,
                rom_fom_errors=rom_errors,
                fom_oracle_errors=oracle_errors,
                passed=passed,
            )
        )
    return tuple(rows)


def _verify_envelope_refusal(
    trained: TrainedPymorROM,
) -> tuple[dict[str, Any], ...]:
    config = trained.config
    values = (config.volatility_min - 0.01, config.volatility_max + 0.01)
    rows: list[dict[str, Any]] = []
    for volatility in values:
        try:
            trained.solve(volatility)
        except ROMEnvelopeError as exc:
            rows.append(
                {
                    "volatility": volatility,
                    "passed": True,
                    "reason": exc.reason,
                    "fallback": exc.fallback,
                }
            )
        else:
            rows.append({"volatility": volatility, "passed": False})
    return tuple(rows)


def _verify_predecessor(root: str | Path | None) -> dict[str, Any]:
    if root is None:
        return {
            "path": PREDECESSOR_PATH,
            "expected_sha256": PREDECESSOR_SHA256,
            "observed_sha256": None,
            "verification_mode": "declared_digest_only",
            "verified": False,
        }
    path = Path(root) / PREDECESSOR_PATH
    if not path.is_file():
        raise ValueError(f"missing predecessor artifact: {PREDECESSOR_PATH}")
    observed = file_sha256(path)
    if observed != PREDECESSOR_SHA256:
        raise ValueError("OpenTURNS predecessor artifact hash mismatch")
    return {
        "path": PREDECESSOR_PATH,
        "expected_sha256": PREDECESSOR_SHA256,
        "observed_sha256": observed,
        "verification_mode": "file_sha256",
        "verified": True,
    }


def _absolute_errors(left: OptionOutputs, right: OptionOutputs) -> OptionOutputs:
    return OptionOutputs(
        price=abs(left.price - right.price),
        delta=abs(left.delta - right.delta),
        gamma=abs(left.gamma - right.gamma),
    )


def _major_minor(value: str) -> tuple[int, int] | None:
    try:
        major, minor, *_ = value.split(".")
        return int(major), int(minor)
    except (TypeError, ValueError):
        return None


__all__ = [
    "DECISION_POLICY",
    "HoldoutEvaluation",
    "PymorBenchmarkReport",
    "TimingSummary",
    "run_pymor_benchmark",
    "verify_pymor_benchmark",
]
