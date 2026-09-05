"""Static contract for the isolated Bayesian/JAX profile artifact."""

from __future__ import annotations

import json
from pathlib import Path

from finite_element_options.estimation.bayesian_profile import stable_environment_checks
from finite_element_options.validation.evidence.serialization import (
    canonical_json_sha256,
    file_sha256,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "docs/evidence/bayesian_jax_profile_2026-09-05.json"
LOCK = ROOT / "environments/bayesian-jax-py312/requirements.lock"
EXPECTED_SHA256 = "f00e3f73e043dcfcff0a12b0fb2edcac4a1d3418f968e118c8f545d9ff7026e7"
EXPECTED_LOCK_SHA256 = "1f97148d8501965688e450aff6563abd0172c7098c622cf50bd9a0848d9e1f7f"


def test_bayesian_jax_profile_artifact_is_hash_bound_and_passed() -> None:
    """Keep the wheel/lock, diagnostics, seeds, and fail-closed scope synchronized."""

    assert file_sha256(ARTIFACT) == EXPECTED_SHA256
    assert file_sha256(LOCK) == EXPECTED_LOCK_SHA256
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert canonical_json_sha256(payload["input"]) == payload["input_hash"]
    assert payload["environment_lock"]["sha256"] == EXPECTED_LOCK_SHA256
    assert payload["environment_lock"]["expected_sha256"] == EXPECTED_LOCK_SHA256
    assert payload["environment_lock"]["observed_sha256"] == EXPECTED_LOCK_SHA256
    assert payload["environment"]["python"].startswith("3.12")
    assert payload["environment"]["finite_element_options_install_mode"] == "wheel"
    assert payload["profile_split"]["base_dependency_expansion"] is False
    assert payload["profile_split"]["calibration_is_lightweight"] is True
    assert payload["profile_split"]["bayesian_extras_declared"] is True
    assert payload["environment"]["finite_element_options"] == "0.2.0"
    assert payload["pymc"]["passed"] is True
    assert payload["numpyro"]["passed"] is True
    assert payload["pymc"]["finite_log_density"] is True
    assert payload["numpyro"]["finite_log_density"] is True
    assert payload["pymc"]["rhat"] <= 1.05
    assert payload["numpyro"]["rhat"] <= 1.05
    assert payload["pymc"]["ess_bulk"] >= 100
    assert payload["numpyro"]["ess_bulk"] >= 100
    assert payload["cross_engine"]["posterior_sd_abs_difference"] <= 0.02
    assert payload["cross_engine"]["posterior_predictive_sd_abs_difference"] <= 0.04
    assert payload["pymc"]["checks"]["posterior_predictive_sd"] is True
    assert payload["numpyro"]["checks"]["posterior_predictive_sd"] is True
    assert payload["jax_fem_differentiation"]["supported"] is False
    assert payload["decision"]["automatic_fem_differentiation"] is False
    assert payload["decision"]["status"] == "adopt_isolated_bayesian_profiles"
    assert all(payload["decision"]["checks"].values())
    assert "/home/" not in ARTIFACT.read_text(encoding="utf-8")


def test_semantic_replay_ignores_host_and_python_patch_provenance() -> None:
    """Equivalent locked Python 3.12 environments replay across CI hosts."""

    expected = {
        "python": "3.12.3",
        "implementation": "CPython",
        "platform": "Linux-host-a",
        "arviz": "0.23.4",
        "finite_element_options": "0.2.0",
        "jax": "0.9.2",
        "jaxlib": "0.9.2",
        "numpy": "2.5.1",
        "numpyro": "0.21.0",
        "pymc": "5.26.1",
        "scipy": "1.18.0",
    }
    observed = {**expected, "python": "3.12.12", "platform": "Linux-host-b"}
    assert all(stable_environment_checks(observed, expected).values())

    observed["python"] = "3.13.0"
    assert stable_environment_checks(observed, expected)["python_minor"] is False

    observed = {**expected, "finite_element_options": "0.3.0"}
    assert stable_environment_checks(observed, expected)["locked_package_versions"] is False
