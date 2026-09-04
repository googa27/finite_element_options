"""OpenTURNS FEM uncertainty-decomposition pilot orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..evidence_io import file_sha256
from .cases import (
    IMINUIT_ARTIFACT,
    IMINUIT_SHA256,
    QUANTLIB_ORACLE_ARTIFACT,
    QUANTLIB_ORACLE_SHA256,
    SCHEMA_VERSION,
    SCOPE_STATEMENT,
    build_components,
    calibrate_scales,
    canonical_study_input,
    canonical_uq_input_hash,
    evaluate_response,
    numpy_direct_sample,
    summarize_prices,
)
from .contracts import (
    COMPONENT_NAMES,
    ComponentName,
    UQPilotConfig,
    UQPilotResult,
    UQParityResult,
    UQPropagationResult,
)
from .openturns_adapter import (
    additive_sobol_recovery,
    component_variance_estimates,
    saltelli_indices,
    sample_normalized,
)

DECISION_POLICY = (
    "Retain OpenTURNS only as an optional uncertainty adapter when real FEM propagation, "
    "direct NumPy parity, additive Sobol recovery, raw Sobol finite-sample validation, "
    "component provenance, and RNG isolation pass; "
    "never promote to core/base or production maturity in this pilot."
)
SOBOL_SANITY_ENVELOPE = (-0.25, 1.25)


def verify_predecessor_hashes(root: Path | None = None) -> dict[str, dict[str, Any]]:
    """Verify predecessor evidence hashes required before canonical execution."""

    base = Path.cwd() if root is None else root
    checks = {
        "quantlib_oracle": (QUANTLIB_ORACLE_ARTIFACT, QUANTLIB_ORACLE_SHA256, True),
        "iminuit_identifiability": (IMINUIT_ARTIFACT, IMINUIT_SHA256, False),
    }
    out: dict[str, dict[str, Any]] = {}
    for name, (relative, expected, used) in checks.items():
        path = base / relative
        observed = file_sha256(path)
        if observed != expected:
            raise ValueError(
                f"hash mismatch for {relative}: expected {expected}, observed {observed}"
            )
        out[name] = {
            "artifact": relative,
            "expected_sha256": expected,
            "observed_sha256": observed,
            "verified": True,
            "used_as_parameter_source": used,
        }
    return out


def run_openturns_uq_pilot(
    config: UQPilotConfig | None = None, root: Path | None = None
) -> UQPilotResult:
    """Execute the evidence-gated OpenTURNS UQ pilot with real FEM evaluations."""

    controls = UQPilotConfig() if config is None else config
    predecessor_checks = verify_predecessor_hashes(root)
    calibration = calibrate_scales()
    components = build_components(calibration, controls)

    sample = sample_normalized(controls.sample_seed, controls.sample_size)
    ot_values = np.asarray([evaluate_response(row, calibration) for row in sample], dtype=float)
    price_summary = summarize_prices(ot_values)
    first, total, sobol_intervals, version, distribution_constructor = saltelli_indices(
        lambda row: evaluate_response(row, calibration),
        seed=controls.sobol_seed,
        base_size=controls.sobol_base_size,
    )
    sobol_validation = _sobol_validation(first, total, sobol_intervals)
    component_variances = component_variance_estimates(
        lambda row: evaluate_response(row, calibration), calibration, controls
    )
    propagation = UQPropagationResult(
        prices=price_summary,
        first_order_sobol=first,
        total_order_sobol=total,
        sobol_intervals=sobol_intervals,
        sobol_validation=sobol_validation,
        component_variance=component_variances,
        finite_count=int(np.isfinite(ot_values).sum()),
        sample_seed=controls.sample_seed,
        sample_size=controls.sample_size,
        sobol_seed=controls.sobol_seed,
        sobol_base_size=controls.sobol_base_size,
        openturns_version=version,
        distribution_constructor=distribution_constructor,
    )

    direct = _direct_reference(controls, calibration, propagation, ot_values)
    additive = additive_sobol_recovery(controls)
    attribution = _attribution_table(components, propagation)
    source_hashes_present = _all_source_hashes_present(components, calibration)
    passed = bool(
        direct.passed
        and additive.passed
        and source_hashes_present
        and propagation.sobol_validation["passed"]
        and propagation.finite_count == propagation.sample_size
    )
    decision = {
        "status": "retain_optional_adapter" if passed else "reject_adapter_until_gates_pass",
        "passed": passed,
        "maturity": "experimental_optional_non_production" if passed else "rejected",
        "policy": DECISION_POLICY,
        "why_numpy_remains_baseline": (
            "The NumPy route directly samples the five public normalized marginals and evaluates the "
            "same FEM response, so it remains the lightweight parity/reference baseline."
        ),
        "what_openturns_adds": (
            "OpenTURNS adds maintained composed-distribution sampling, SobolIndicesExperiment design "
            "generation, and raw finite-sample Saltelli first/total Sobol estimators with confidence "
            "interval diagnostics behind an optional extra."
        ),
        "non_production_limits": (
            "One-regime public-synthetic fixed-FX quanto diagnostic only; no PDP coupling, no capability "
            "matrix maturity upgrade, and no claim about intrinsic fair-value risk distribution."
        ),
    }
    provenance = {
        "predecessor_hash_verification": predecessor_checks,
        "canonical_study_input": canonical_study_input(controls),
        "privacy_class": "public-synthetic",
        "raw_samples_recorded": False,
        "local_paths_recorded": False,
        "openturns_dependency_evidence": {
            "requirement": "openturns>=1.27,<2",
            "observed_version": version,
            "license": "LGPLv3+ per PyPI project metadata/docs evidence",
            "api_used": [
                distribution_constructor,
                "RandomGenerator.SetSeed/GetState/SetState",
                "SobolIndicesExperiment(distribution, size, False)",
                "SaltelliSensitivityAlgorithm(inputDesign, outputDesign, size)",
                "SaltelliSensitivityAlgorithm.setUseAsymptoticDistribution(True)",
                "getFirstOrderIndicesInterval/getTotalOrderIndicesInterval",
            ],
        },
    }
    return UQPilotResult(
        schema_version=SCHEMA_VERSION,
        issue="https://github.com/googa27/finite_element_options/issues/134",
        scope=SCOPE_STATEMENT,
        decision=decision,
        study_input_hash=canonical_uq_input_hash(controls),
        component_names=COMPONENT_NAMES,
        components=list(components),
        calibration=calibration,
        propagation=propagation,
        direct_reference=direct,
        additive_sobol_recovery=additive,
        attribution_table=attribution,
        provenance=provenance,
    )


def _direct_reference(
    controls: UQPilotConfig,
    calibration: Any,
    propagation: UQPropagationResult,
    ot_values: np.ndarray,
) -> UQParityResult:
    direct_sample = numpy_direct_sample(controls.direct_seed, controls.direct_size)
    direct_values = np.asarray(
        [evaluate_response(row, calibration) for row in direct_sample], dtype=float
    )
    direct_summary = summarize_prices(direct_values)
    differences: dict[str, float] = {
        "mean": abs(float(propagation.prices["mean"]) - float(direct_summary["mean"])),
        "std": abs(float(propagation.prices["std"]) - float(direct_summary["std"])),
    }
    ot_quantiles = propagation.prices["quantiles"]
    direct_quantiles = direct_summary["quantiles"]
    for level, value in ot_quantiles.items():
        differences[f"quantile_{level}"] = abs(float(value) - float(direct_quantiles[level]))
    tolerances = _sampling_tolerances(ot_values, direct_values)
    passed = all(differences[key] <= tolerances[key] for key in differences)
    return UQParityResult(
        direct_seed=controls.direct_seed,
        direct_size=controls.direct_size,
        direct_prices=direct_summary,
        differences=differences,
        tolerances=tolerances,
        passed=bool(passed),
        tolerance_formula=(
            "Mean tolerance is 3*sqrt(var_ot/n_ot+var_np/n_np). Std tolerance uses the normal-theory "
            "3-sigma standard error of sample standard deviations. Quantile tolerances use a fixed-seed "
            "pooled-null bootstrap 99.5% envelope: pool the two empirical samples, resample two "
            "independent size-n samples from that null distribution, and add one combined standard "
            "error of means."
        ),
    )


def _sampling_tolerances(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    n_a = a.size
    n_b = b.size
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    mean_tol = 3.0 * float(np.sqrt(var_a / n_a + var_b / n_b))
    std_a = float(np.std(a, ddof=1))
    std_b = float(np.std(b, ddof=1))
    std_tol = 3.0 * float(np.sqrt(std_a**2 / (2 * (n_a - 1)) + std_b**2 / (2 * (n_b - 1))))
    mean_se = mean_tol / 3.0
    pooled = np.concatenate([np.asarray(a, dtype=float), np.asarray(b, dtype=float)])
    rng = np.random.default_rng(134_902)
    tolerances = {"mean": mean_tol, "std": std_tol}
    levels = (0.01, 0.05, 0.5, 0.95, 0.99)
    for level in levels:
        diffs = []
        for _ in range(600):
            ra = rng.choice(pooled, size=n_a, replace=True)
            rb = rng.choice(pooled, size=n_b, replace=True)
            diffs.append(abs(float(np.quantile(ra, level)) - float(np.quantile(rb, level))))
        tolerances[f"quantile_{level}"] = float(np.quantile(diffs, 0.995) + mean_se)
    return tolerances


def _attribution_table(
    components: tuple[Any, ...], propagation: UQPropagationResult
) -> dict[ComponentName, dict[str, Any]]:
    table: dict[ComponentName, dict[str, Any]] = {}
    for component in components:
        name = component.name
        table[name] = {
            "distribution": component.distribution,
            "role": component.role,
            "perturbs_fem_model": component.perturbs_fem_model,
            "additive_validation_estimator_error": component.additive_validation_estimator_error,
            "standalone_variance": propagation.component_variance[name]["variance"],
            "raw_first_order_sobol": propagation.first_order_sobol[name],
            "raw_total_order_sobol": propagation.total_order_sobol[name],
            "first_order_confidence_interval": propagation.sobol_intervals["first_order"][name],
            "total_order_confidence_interval": propagation.sobol_intervals["total_order"][name],
            "interpretation": (
                "Standalone one-at-a-time variance estimate; raw Saltelli first/total estimators "
                "are finite-sample diagnostics and may be slightly negative or out of [0, 1] due "
                "to sampling noise rather than physical Sobol values."
            ),
        }
    return table


def _all_source_hashes_present(components: tuple[Any, ...], calibration: Any) -> bool:
    hashes = [component.source_hash for component in components]
    hashes.extend(
        [
            calibration.fine_grid_hash,
            calibration.coarse_grid_hash,
            calibration.baseline_model_hash,
            calibration.payoff_hash,
        ]
    )
    return all(_is_lower_sha256_string(value) for value in hashes)


def _is_lower_sha256_string(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _sobol_validation(
    first: dict[ComponentName, float],
    total: dict[ComponentName, float],
    intervals: dict[str, dict[ComponentName, dict[str, float]]],
) -> dict[str, Any]:
    lower, upper = SOBOL_SANITY_ENVELOPE
    point_violations: list[dict[str, Any]] = []
    nonfinite_points: list[dict[str, str]] = []
    interval_violations: list[dict[str, Any]] = []
    interval_bound_failures: list[dict[str, Any]] = []

    for family, values in (("first_order", first), ("total_order", total)):
        for component, value in values.items():
            if not np.isfinite(value):
                nonfinite_points.append({"family": family, "component": component})
            elif not lower <= value <= upper:
                point_violations.append(
                    {"family": family, "component": component, "value": float(value)}
                )

    for family, components in intervals.items():
        for component, bounds in components.items():
            low = float(bounds["lower"])
            high = float(bounds["upper"])
            if not np.isfinite(low) or not np.isfinite(high):
                interval_bound_failures.append(
                    {
                        "family": family,
                        "component": component,
                        "lower": low,
                        "upper": high,
                        "reason": "nonfinite_bound",
                    }
                )
            elif low > high:
                interval_bound_failures.append(
                    {
                        "family": family,
                        "component": component,
                        "lower": low,
                        "upper": high,
                        "reason": "lower_greater_than_upper",
                    }
                )
            if low < lower or high > upper:
                interval_violations.append(
                    {
                        "family": family,
                        "component": component,
                        "lower": low,
                        "upper": high,
                    }
                )

    passed = not (nonfinite_points or point_violations or interval_bound_failures)
    return {
        "passed": bool(passed),
        "point_sanity_envelope": {"lower": lower, "upper": upper},
        "point_violations": point_violations,
        "nonfinite_points": nonfinite_points,
        "interval_bound_failures": interval_bound_failures,
        "interval_bounds_outside_point_envelope": interval_violations,
        "interval_bounds_outside_point_envelope_gate": "reported_only",
        "interpretation": (
            "Raw Saltelli finite-sample point estimates are not clipped. The gate requires finite "
            "points within the declared pilot sanity envelope and finite confidence-interval bounds "
            "with lower<=upper; small negative estimates or intervals crossing zero are sampling noise."
        ),
    }


__all__ = ["DECISION_POLICY", "run_openturns_uq_pilot", "verify_predecessor_hashes"]
