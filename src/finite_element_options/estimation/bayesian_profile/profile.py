"""Combined isolated-profile evidence and fail-closed adoption decision."""

from __future__ import annotations

from collections.abc import Mapping
from importlib.metadata import distribution, version
from pathlib import Path
import platform
import sys
from typing import Any

from finite_element_options.contracts.evidence_serialization import (
    canonical_json_sha256,
    distribution_install_mode,
    file_sha256,
    quantize_json_floats,
)

from .contracts import BayesianSmokeConfig, SCHEMA_VERSION
from .numpyro_smoke import jax_fem_differentiation_status, run_numpyro_smoke
from .oracle import exact_normal_posterior
from .pymc_smoke import run_pymc_smoke


PREDECESSOR_PATH = "docs/evidence/petsc_vi_assessment_2026-09-05.json"
PREDECESSOR_SHA256 = "b0ebd55b748c2c36382854ad6624f3f983b8a1ee25cdb1e34419df7fa9da5b35"
LOCK_PATH = "environments/bayesian-jax-py312/requirements.lock"
LOCK_SHA256 = "1f97148d8501965688e450aff6563abd0172c7098c622cf50bd9a0848d9e1f7f"
LOCKED_VERSION_KEYS = (
    "arviz",
    "finite_element_options",
    "jax",
    "jaxlib",
    "numpy",
    "numpyro",
    "pymc",
    "scipy",
)


def _python_minor(version: object) -> tuple[int, int] | None:
    try:
        major, minor, *_ = (int(part) for part in str(version).split("."))
    except (TypeError, ValueError):
        return None
    return major, minor


def stable_environment_checks(
    observed: Mapping[str, object], expected: Mapping[str, object]
) -> dict[str, bool]:
    """Compare locked semantics without coupling replay to host provenance."""

    return {
        "locked_package_versions": all(
            observed.get(key) == expected.get(key) for key in LOCKED_VERSION_KEYS
        ),
        "python_minor": _python_minor(observed.get("python"))
        == _python_minor(expected.get("python"))
        == (3, 12),
        "implementation": observed.get("implementation") == expected.get("implementation"),
    }


def run_bayesian_jax_profile(
    *,
    config: BayesianSmokeConfig | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Run native PyMC and JAX-native NumPyro evidence in one isolated profile."""

    selected = config or BayesianSmokeConfig()
    predecessor = _predecessor(root)
    lock = _lock_evidence(root)
    install_mode = distribution_install_mode("finite-element-options")
    profile_split = _profile_split_evidence()
    pymc = run_pymc_smoke(selected)
    numpyro = run_numpyro_smoke(selected)
    exact = exact_normal_posterior(selected)
    jax_fem = jax_fem_differentiation_status()
    cross_engine = {
        "posterior_mean_abs_difference": abs(pymc["posterior_mean"] - numpyro["posterior_mean"]),
        "posterior_sd_abs_difference": abs(pymc["posterior_sd"] - numpyro["posterior_sd"]),
        "posterior_predictive_mean_abs_difference": abs(
            pymc["posterior_predictive_mean"] - numpyro["posterior_predictive_mean"]
        ),
        "posterior_predictive_sd_abs_difference": abs(
            pymc["posterior_predictive_sd"] - numpyro["posterior_predictive_sd"]
        ),
    }
    checks = {
        "python_312": sys.version_info[:2] == (3, 12),
        "installed_wheel": install_mode == "wheel",
        "predecessor_verified": bool(predecessor["verified"]),
        "lock_verified": bool(lock["verified"]),
        "base_dependency_isolation": not profile_split["base_dependency_expansion"],
        "calibration_split": bool(profile_split["calibration_is_lightweight"]),
        "bayesian_extras_declared": bool(profile_split["bayesian_extras_declared"]),
        "pymc_posterior": bool(pymc["passed"]),
        "numpyro_posterior": bool(numpyro["passed"]),
        "cross_engine_mean": (
            cross_engine["posterior_mean_abs_difference"]
            <= selected.maximum_cross_engine_mean_difference
        ),
        "cross_engine_sd": (
            cross_engine["posterior_sd_abs_difference"]
            <= selected.maximum_cross_engine_sd_difference
        ),
        "cross_engine_predictive_mean": (
            cross_engine["posterior_predictive_mean_abs_difference"]
            <= selected.maximum_cross_engine_predictive_mean_difference
        ),
        "cross_engine_predictive_sd": (
            cross_engine["posterior_predictive_sd_abs_difference"]
            <= selected.maximum_cross_engine_predictive_sd_difference
        ),
        "jax_fem_fail_closed": jax_fem["supported"] is False,
    }
    promoted = all(checks.values())
    payload = {
        "schema_version": SCHEMA_VERSION,
        "privacy_class": "public_synthetic",
        "scope": (
            "Identifiable one-parameter known-variance synthetic posterior smoke; "
            "not market calibration and not automatic FEM differentiation."
        ),
        "input": selected.to_dict(),
        "input_hash": selected.input_hash,
        "data_hash": canonical_json_sha256(selected.observations),
        "seeds": {
            "pymc": selected.pymc_seed,
            "pymc_predictive": selected.pymc_predictive_seed,
            "numpyro": selected.numpyro_seed,
            "numpyro_predictive": selected.numpyro_predictive_seed,
        },
        "predecessor": predecessor,
        "environment_lock": lock,
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "finite_element_options_install_mode": install_mode,
            "finite_element_options": version("finite-element-options"),
            "numpy": version("numpy"),
            "scipy": version("scipy"),
            "pymc": version("pymc"),
            "arviz": version("arviz"),
            "jax": version("jax"),
            "jaxlib": version("jaxlib"),
            "numpyro": version("numpyro"),
        },
        "profile_split": profile_split,
        "exact_posterior": exact,
        "pymc": pymc,
        "numpyro": numpyro,
        "cross_engine": cross_engine,
        "jax_fem_differentiation": jax_fem,
        "decision": {
            "status": (
                "adopt_isolated_bayesian_profiles" if promoted else "reject_profile_adoption"
            ),
            "promoted": promoted,
            "checks": checks,
            "capability_matrix_upgrade": False,
            "base_wheel_change": profile_split["base_dependency_expansion"],
            "automatic_fem_differentiation": False,
        },
    }
    normalized = quantize_json_floats(payload, significant_digits=10)
    if not isinstance(normalized, dict):  # pragma: no cover - payload is a dict
        raise TypeError("Bayesian profile serialization must return a mapping")
    normalized["input"] = selected.to_dict()
    normalized["input_hash"] = selected.input_hash
    return normalized


def _profile_split_evidence() -> dict[str, Any]:
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    requirements = [
        Requirement(item) for item in (distribution("finite-element-options").requires or [])
    ]

    def applies(requirement: Requirement, extra: str) -> bool:
        return requirement.marker is None or requirement.marker.evaluate({"extra": extra})

    def has(extra: str, dependency: str) -> bool:
        expected = canonicalize_name(dependency)
        return any(
            canonicalize_name(requirement.name) == expected and applies(requirement, extra)
            for requirement in requirements
        )

    optional_stacks = {
        canonicalize_name(name)
        for name in ("arviz", "jax", "numpyro", "pandas", "pymc", "statsmodels")
    }
    base_leaks = sorted(
        str(requirement)
        for requirement in requirements
        if canonicalize_name(requirement.name) in optional_stacks and applies(requirement, "")
    )
    calibration_is_lightweight = (
        has("calibration", "pandas")
        and has("calibration", "statsmodels")
        and not has("calibration", "pymc")
        and not has("calibration", "arviz")
    )
    bayesian_extras_declared = (
        has("bayesian", "pymc")
        and has("bayesian", "arviz")
        and all(
            has("bayesian-jax", dependency) for dependency in ("pymc", "arviz", "jax", "numpyro")
        )
    )
    return {
        "calibration": ["pandas", "statsmodels"],
        "bayesian": ["arviz", "pymc"],
        "bayesian-jax": ["arviz", "jax", "numpyro", "pymc"],
        "base_dependency_expansion": bool(base_leaks),
        "base_optional_stack_leaks": base_leaks,
        "calibration_is_lightweight": calibration_is_lightweight,
        "bayesian_extras_declared": bayesian_extras_declared,
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
        raise ValueError("Bayesian profile predecessor hash mismatch")
    return {
        "path": PREDECESSOR_PATH,
        "expected_sha256": PREDECESSOR_SHA256,
        "observed_sha256": observed,
        "verified": True,
    }


def _lock_evidence(root: str | Path | None) -> dict[str, Any]:
    if root is None:
        return {
            "path": LOCK_PATH,
            "expected_sha256": LOCK_SHA256,
            "observed_sha256": None,
            "sha256": None,
            "verified": False,
        }
    path = Path(root) / LOCK_PATH
    if not path.is_file():
        raise ValueError(f"missing Bayesian profile lock: {LOCK_PATH}")
    digest = file_sha256(path)
    if digest != LOCK_SHA256:
        raise ValueError("Bayesian/JAX profile lock hash mismatch")
    return {
        "path": LOCK_PATH,
        "expected_sha256": LOCK_SHA256,
        "observed_sha256": digest,
        "sha256": digest,
        "verified": True,
    }


__all__ = [
    "LOCK_PATH",
    "LOCK_SHA256",
    "PREDECESSOR_PATH",
    "PREDECESSOR_SHA256",
    "run_bayesian_jax_profile",
    "stable_environment_checks",
]
