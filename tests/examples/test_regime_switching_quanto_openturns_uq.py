"""OpenTURNS FEM uncertainty-decomposition pilot tests for issue #134."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
ADOPTION = "finite_element_options.examples.regime_switching_quanto.adoption"
UNCERTAINTY = f"{ADOPTION}.uncertainty"
ARTIFACT = ROOT / "docs" / "evidence" / "regime_switching_quanto_openturns_uq_2026-09-04.json"
EXPECTED_ARTIFACT_SHA256 = "c7e90d2857b9d43da5ed65221c2b85969aecfb62a2be45966bade82cd336b6a3"
_VALID_SHA = "0" * 64


def _minimal_calibration():
    from finite_element_options.examples.regime_switching_quanto.adoption.uncertainty import (
        UQCalibration,
    )

    return UQCalibration(
        baseline_price_fine=1.0,
        baseline_price_coarse=0.9,
        baseline_price_oracle=0.8,
        fine_oracle_abs_error=0.2,
        coarse_oracle_abs_error=0.1,
        oracle_identity="test analytical oracle",
        domain_error_grid={"points": 8},
        domain_error_grid_hash=_VALID_SHA,
        domain_max_fine_oracle_abs_error=0.15,
        domain_max_error_input={"spot": 100.0, "sigma": 0.2, "correlation_weight": 0.5},
        domain_error_safety_factor=1.1,
        numerical_half_width=0.2,
        numerical_formula="test",
        mc_price=1.1,
        mc_standard_error=0.3,
        mc_seed=1,
        mc_paths=8,
        mc_steps=8,
        mc_steps_per_year=8,
        fine_grid={},
        coarse_grid={},
        fine_grid_hash=_VALID_SHA,
        coarse_grid_hash=_VALID_SHA,
        baseline_model_hash=_VALID_SHA,
        payoff_hash=_VALID_SHA,
        oracle_hash=_VALID_SHA,
    )


@pytest.fixture(scope="module")
def pilot_result():
    pytest.importorskip("openturns")
    from finite_element_options.examples.regime_switching_quanto.adoption.uncertainty import (
        run_openturns_uq_pilot,
    )

    return run_openturns_uq_pilot(root=ROOT)


def _run_import_probe(code: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_contracts_enforce_exactly_five_components_and_no_model_risk(pilot_result: object) -> None:
    """The public contract separates all five required components."""

    from finite_element_options.examples.regime_switching_quanto.adoption.uncertainty import (
        COMPONENT_NAMES,
    )

    artifact = pilot_result.to_dict()
    assert tuple(artifact["component_names"]) == COMPONENT_NAMES
    assert [component["name"] for component in artifact["components"]] == list(COMPONENT_NAMES)
    assert "model_risk" not in json.dumps(artifact)
    for component in artifact["components"]:
        assert component["source_hash"]
        assert len(component["source_hash"]) == 64
        assert component["distribution"]
        assert component["units"]
        for value in component["scale_or_range"].values():
            if isinstance(value, (int, float)):
                assert np.isfinite(value)
    numerical = next(item for item in artifact["components"] if item["name"] == "numerical")
    assert numerical["scale_or_range"]["mode"] == "independent_domain_screened_oracle_envelope"
    assert "domain_half_width" in numerical["distribution"]


def test_custom_config_component_sources_match_custom_study_hash(pilot_result: Any) -> None:
    """Non-default controls must propagate into study-input-sourced component hashes."""

    from finite_element_options.examples.regime_switching_quanto.adoption.uncertainty import (
        UQPilotConfig,
        build_components,
        canonical_uq_input_hash,
    )

    custom = UQPilotConfig(sample_seed=999_134)
    expected = canonical_uq_input_hash(custom)
    components = build_components(pilot_result.calibration, custom)

    assert expected != pilot_result.study_input_hash
    assert all(component.source_hash == expected for component in components[:3])
    assert all(component.source_hash != expected for component in components[3:])


def test_monte_carlo_source_hash_binds_calibrated_outputs(pilot_result: Any) -> None:
    """Changing MC price/error evidence must change the component provenance digest."""

    from dataclasses import replace

    from finite_element_options.examples.regime_switching_quanto.adoption.uncertainty import (
        build_components,
    )

    original = build_components(pilot_result.calibration)[4].source_hash
    changed_calibration = replace(
        pilot_result.calibration,
        mc_price=pilot_result.calibration.mc_price + 1.0,
        mc_standard_error=pilot_result.calibration.mc_standard_error + 0.5,
    )
    changed = build_components(changed_calibration)[4].source_hash

    assert changed != original


def test_runtime_provenance_names_actual_distribution_constructor(pilot_result: Any) -> None:
    """Artifact API provenance must report the constructor selected by the installed OpenTURNS."""

    propagation = pilot_result.propagation.to_dict()
    api_used = pilot_result.provenance["openturns_dependency_evidence"]["api_used"]

    assert propagation["distribution_constructor"] in {
        "ComposedDistribution",
        "JointDistribution",
    }
    assert api_used[0] == propagation["distribution_constructor"]


def test_runner_works_outside_checkout_and_fails_closed_on_unverified_predecessors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Installed-wheel execution must not require repository-only docs/evidence files."""

    from finite_element_options.examples.regime_switching_quanto.adoption.uncertainty import (
        UQPilotConfig,
        run_openturns_uq_pilot,
    )

    pytest.importorskip("openturns")
    monkeypatch.chdir(tmp_path)
    result = run_openturns_uq_pilot(
        UQPilotConfig(
            sample_size=8,
            sobol_base_size=8,
            direct_size=8,
            component_size=8,
            additive_sobol_base_size=64,
        )
    )

    checks = result.provenance["predecessor_hash_verification"]
    assert result.decision["passed"] is False
    assert result.decision["status"] == "reject_adapter_until_gates_pass"
    assert result.decision["predecessor_hashes_verified"] is False
    assert all(check["verification_mode"] == "declared_digest_only" for check in checks.values())
    assert all(check["observed_sha256"] is None for check in checks.values())


def test_explicit_root_reports_missing_predecessor_as_typed_error(tmp_path: Path) -> None:
    """Partial checkout errors must name the missing relative artifact without a raw path."""

    from finite_element_options.examples.regime_switching_quanto.adoption.uncertainty import (
        verify_predecessor_hashes,
    )

    with pytest.raises(ValueError, match="missing predecessor artifact: docs/evidence/"):
        verify_predecessor_hashes(tmp_path)


def test_real_fem_response_calls_existing_solver_and_records_separate_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scale calibration and response evaluation route through the existing FEM solver."""

    from finite_element_options.examples.regime_switching_quanto import fem
    from finite_element_options.examples.regime_switching_quanto.adoption.uncertainty import cases

    original = fem.price_contract_fem
    calls: list[object] = []

    def wrapped(*args: object, **kwargs: object):
        calls.append(kwargs.get("grid"))
        return original(*args, **kwargs)

    monkeypatch.setattr(fem, "price_contract_fem", wrapped)
    calibration = cases.calibrate_scales()
    value = cases.evaluate_response(np.zeros(5), calibration)
    assert len(calls) >= 3
    assert np.isfinite(value)
    assert calibration.numerical_half_width >= calibration.fine_oracle_abs_error > 0.0
    assert calibration.numerical_half_width >= calibration.coarse_oracle_abs_error > 0.0
    assert calibration.numerical_half_width >= (
        calibration.domain_error_safety_factor * calibration.domain_max_fine_oracle_abs_error
    )
    assert calibration.domain_error_grid["spot_levels"] == 11
    assert calibration.domain_max_error_input == {
        "spot": 100.0,
        "sigma": 0.17,
        "correlation_weight": 0.0,
    }
    assert calibration.mc_standard_error > 0.0
    assert "analytical_oracle_price" in calibration.numerical_formula
    assert calibration.baseline_price_oracle == pytest.approx(5615.513349, rel=1.0e-7)
    assert calibration.mc_seed == 134_011
    assert calibration.fine_grid_hash != calibration.coarse_grid_hash
    assert calibration.oracle_hash != calibration.fine_grid_hash


def test_numerical_error_envelope_covers_supported_domain_screening_case() -> None:
    """The independent domain envelope covers the reviewer's worst screened input."""

    from finite_element_options.examples.regime_switching_quanto.adoption.uncertainty import cases

    calibration = cases.calibrate_scales()
    zero_numerical = np.asarray([0.0, -1.0, -1.0, 0.0, 0.0])
    positive_numerical = np.asarray([0.0, -1.0, -1.0, 1.0, 0.0])
    fem_price = cases.evaluate_response(zero_numerical, calibration)
    perturbed_price = cases.evaluate_response(positive_numerical, calibration)
    oracle_price = cases.analytical_price(
        spot=100.0,
        sigma=0.17,
        correlation_weight=0.0,
    )
    off_baseline_error = abs(fem_price - oracle_price)

    assert off_baseline_error == pytest.approx(calibration.domain_max_fine_oracle_abs_error)
    assert calibration.numerical_half_width >= (
        calibration.domain_error_safety_factor * off_baseline_error
    )
    assert perturbed_price - fem_price == pytest.approx(calibration.numerical_half_width)


def test_mapping_and_baseline_model_fail_closed_without_clipping() -> None:
    """Normalized coordinates and mapped model parameters reject unlawful values."""

    from finite_element_options.examples.regime_switching_quanto.adoption.uncertainty import (
        baseline_model,
        map_normalized_inputs,
    )

    calibration = _minimal_calibration()
    mapped = map_normalized_inputs(np.asarray([1.0, -1.0, 0.0, 1.0, 8.0]), calibration)
    assert mapped["spot"] == 105.0
    assert mapped["sigma"] == pytest.approx(0.17)
    assert mapped["correlation_weight"] == 0.5
    assert mapped["numerical_error"] == pytest.approx(0.2)
    assert mapped["monte_carlo_error"] == pytest.approx(2.4)

    bad_inputs = (
        [1.01, 0.0, 0.0, 0.0, 0.0],
        [0.0, -1.01, 0.0, 0.0, 0.0],
        [0.0, 0.0, np.nan, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.01, 0.0],
        [0.0, 0.0, 0.0, 0.0, np.inf],
        [0.0, 0.0, 0.0, 0.0],
    )
    for values in bad_inputs:
        with pytest.raises(ValueError):
            map_normalized_inputs(np.asarray(values), calibration)

    with pytest.raises(ValueError, match="correlation_weight"):
        baseline_model(correlation_weight=1.01)
    with pytest.raises(ValueError, match="sigma"):
        baseline_model(sigma=np.inf)


def test_calibration_hashes_require_lowercase_sha256_hex() -> None:
    """Calibration provenance hashes reject uppercase, non-hex, and short strings."""

    from finite_element_options.examples.regime_switching_quanto.adoption.uncertainty import (
        UQCalibration,
    )

    baseline = _minimal_calibration().to_dict()
    for field, bad_hash in (
        ("fine_grid_hash", "A" * 64),
        ("coarse_grid_hash", "g" * 64),
        ("baseline_model_hash", "0" * 63),
        ("oracle_hash", "Z" * 64),
        ("domain_error_grid_hash", "x" * 64),
    ):
        payload = dict(baseline)
        payload[field] = bad_hash
        with pytest.raises(ValueError, match="lowercase SHA-256 hex"):
            UQCalibration(**payload)

    undercovered = dict(baseline)
    undercovered["numerical_half_width"] = 0.19
    with pytest.raises(ValueError, match="must cover baseline and domain analytical errors"):
        UQCalibration(**undercovered)


def test_sobol_raw_estimates_are_not_clipped_and_validation_reports_envelope() -> None:
    """Raw Saltelli serialization preserves negatives and values above one."""

    from finite_element_options.examples.regime_switching_quanto.adoption.uncertainty import (
        COMPONENT_NAMES,
    )
    from finite_element_options.examples.regime_switching_quanto.adoption.uncertainty.openturns_adapter import (
        _raw_indices,
    )
    from finite_element_options.examples.regime_switching_quanto.adoption.uncertainty.pilot import (
        _sobol_validation,
    )

    raw = _raw_indices([1.3349, 0.525, -0.1538, -0.0405, -0.000007])
    assert raw["data"] == pytest.approx(1.3349)
    assert raw["model_form"] == pytest.approx(-0.1538)
    assert sum(raw.values()) > 1.5

    intervals = {
        family: {name: {"lower": -0.2, "upper": 1.2} for name in COMPONENT_NAMES}
        for family in ("first_order", "total_order")
    }
    total = {name: 0.0 for name in COMPONENT_NAMES}
    validation = _sobol_validation(raw, total, intervals)
    assert validation["passed"] is False
    assert validation["point_sanity_envelopes"] == {
        "first_order": {"lower": -0.05, "upper": 1.0},
        "total_order": {"lower": -0.05, "upper": 1.05},
    }
    assert validation["point_violations"][0]["family"] == "first_order"
    assert validation["point_violations"][0]["component"] == "data"
    assert validation["point_violations"][0]["value"] == pytest.approx(1.3349)


def test_seeded_openturns_reproducibility_and_json_serialization(pilot_result: object) -> None:
    """OpenTURNS samples and public results are deterministic and JSON serializable."""

    from finite_element_options.examples.regime_switching_quanto.adoption.evidence_io import (
        canonical_json,
    )
    from finite_element_options.examples.regime_switching_quanto.adoption.uncertainty.openturns_adapter import (
        sample_normalized,
    )

    first = sample_normalized(1234, 12)
    second = sample_normalized(1234, 12)
    assert np.array_equal(first, second)
    text = canonical_json(pilot_result.to_dict())
    assert "openturns" in text.lower()
    assert "RandomVector" not in text


def test_direct_numpy_reference_parity_with_declared_sampling_tolerances(
    pilot_result: Any,
) -> None:
    """Independent NumPy sequences agree with OpenTURNS propagation within statistical tolerances."""

    parity = pilot_result.direct_reference.to_dict()
    assert parity["passed"] is True
    for key, difference in parity["differences"].items():
        assert difference <= parity["tolerances"][key]
    assert "pooled-null bootstrap 99.5% envelope" in parity["tolerance_formula"]
    assert "pool the two empirical samples" in parity["tolerance_formula"]


def test_real_sobol_estimates_intervals_and_sanity_gate(pilot_result: Any) -> None:
    """Real FEM Sobol output records raw finite estimates and confidence intervals."""

    from finite_element_options.examples.regime_switching_quanto.adoption.uncertainty import (
        COMPONENT_NAMES,
    )

    propagation = pilot_result.to_dict()["propagation"]
    assert tuple(propagation["first_order_sobol"]) == COMPONENT_NAMES
    assert tuple(propagation["total_order_sobol"]) == COMPONENT_NAMES
    validation = propagation["sobol_validation"]
    assert validation["passed"] is True
    assert validation["point_sanity_envelopes"] == {
        "first_order": {"lower": -0.05, "upper": 1.0},
        "total_order": {"lower": -0.05, "upper": 1.05},
    }
    assert validation["point_violations"] == []
    assert validation["nonfinite_points"] == []
    assert validation["interval_bound_failures"] == []
    for family in ("first_order", "total_order"):
        assert tuple(propagation["sobol_intervals"][family]) == COMPONENT_NAMES
        for name, bounds in propagation["sobol_intervals"][family].items():
            point = propagation[f"{family}_sobol"][name]
            assert np.isfinite(point)
            assert np.isfinite(bounds["lower"])
            assert np.isfinite(bounds["upper"])
            assert bounds["lower"] <= bounds["upper"]


def test_synthetic_additive_variance_and_sobol_recovery(pilot_result: object) -> None:
    """A known additive model validates maintained OpenTURNS Saltelli first/total indices."""

    recovery = pilot_result.additive_sobol_recovery.to_dict()
    assert recovery["passed"] is True
    assert recovery["max_abs_error_first"] <= recovery["tolerance"]
    assert recovery["max_abs_error_total"] <= recovery["tolerance"]
    assert recovery["expected_first"]["monte_carlo"] > recovery["expected_first"]["data"]


def test_numerical_absent_from_parameter_and_mc_is_estimator_only(pilot_result: object) -> None:
    """Numerical and MC additive validation errors are not folded into parameter uncertainty."""

    components = {component.name: component for component in pilot_result.components}
    assert "numerical error is excluded" in components["parameter"].description.lower()
    assert "equity volatility only" in components["parameter"].description.lower()
    assert components["numerical"].additive_validation_estimator_error is True
    assert components["numerical"].perturbs_fem_model is False
    assert components["monte_carlo"].additive_validation_estimator_error is True
    assert components["monte_carlo"].perturbs_fem_model is False
    assert "not intrinsic fair-value uncertainty" in components["monte_carlo"].description


def test_openturns_rng_success_failure_and_coordinated_concurrency_restoration() -> None:
    """Shared public context restores RNG state for success, failure, and coordinated calls."""

    openturns = pytest.importorskip("openturns")
    from finite_element_options.examples.regime_switching_quanto.adoption import openturns_seeded

    def next_draw_after_seed(seed: int) -> list[list[float]]:
        openturns.RandomGenerator.SetSeed(seed)
        return list(openturns.Normal().getSample(3))

    expected = next_draw_after_seed(9001)
    openturns.RandomGenerator.SetSeed(9001)
    with openturns_seeded(1):
        assert len(openturns.Normal().getSample(4)) == 4
    assert list(openturns.Normal().getSample(3)) == expected

    openturns.RandomGenerator.SetSeed(9002)
    expected_failure = list(openturns.Normal().getSample(3))
    openturns.RandomGenerator.SetSeed(9002)
    with pytest.raises(RuntimeError, match="forced OpenTURNS failure"):
        with openturns_seeded(2):
            raise RuntimeError("forced OpenTURNS failure")
    assert list(openturns.Normal().getSample(3)) == expected_failure

    openturns.RandomGenerator.SetSeed(9003)
    expected_concurrent = list(openturns.Normal().getSample(3))
    openturns.RandomGenerator.SetSeed(9003)
    entered_first = threading.Event()
    release_first = threading.Event()
    events: list[str] = []
    failures: list[BaseException] = []

    def worker_one() -> None:
        try:
            with openturns_seeded(10):
                events.append("first_enter")
                entered_first.set()
                assert release_first.wait(5.0)
                events.append("first_exit")
        except BaseException as exc:  # pragma: no cover - thread handoff
            failures.append(exc)

    def worker_two() -> None:
        try:
            assert entered_first.wait(5.0)
            with openturns_seeded(11):
                events.append("second_enter")
                events.append("second_exit")
        except BaseException as exc:  # pragma: no cover - thread handoff
            failures.append(exc)

    first_thread = threading.Thread(target=worker_one)
    second_thread = threading.Thread(target=worker_two)
    first_thread.start()
    second_thread.start()
    entered_first.wait(5.0)
    time.sleep(0.05)
    assert events == ["first_enter"]
    release_first.set()
    first_thread.join(5.0)
    second_thread.join(5.0)
    assert not failures
    assert events == ["first_enter", "first_exit", "second_enter", "second_exit"]
    assert list(openturns.Normal().getSample(3)) == expected_concurrent


def test_missing_extra_hint_and_base_facade_contract_imports_with_openturns_blocked() -> None:
    """Base/facade/contracts import without OpenTURNS; execution advertises exact extra."""

    probe = _run_import_probe(
        f"""
        import importlib
        import importlib.abc
        import sys

        class BlockOpenTURNS(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname == 'openturns' or fullname.startswith('openturns.'):
                    raise ModuleNotFoundError('blocked optional dependency', name='openturns')
                return None

        sys.modules.pop('openturns', None)
        sys.meta_path.insert(0, BlockOpenTURNS())
        for module_name in (
            'finite_element_options',
            {ADOPTION!r},
            {UNCERTAINTY!r},
            {UNCERTAINTY + ".contracts"!r},
        ):
            importlib.import_module(module_name)
        assert 'openturns' not in sys.modules
        from {ADOPTION} import openturns_seeded
        assert callable(openturns_seeded)
        assert 'openturns' not in sys.modules
        from {UNCERTAINTY}.openturns_adapter import sample_normalized
        try:
            sample_normalized(1, 8)
        except ImportError as exc:
            message = str(exc)
            assert 'finite-element-options[uncertainty]' in message
            assert 'openturns>=1.27,<2' in message
        else:
            raise AssertionError('expected missing OpenTURNS extra')
        """
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr


def test_canonical_artifact_sha_input_scope_and_decision_static() -> None:
    """Committed canonical artifact remains hash-bound and active without OpenTURNS."""

    from finite_element_options.examples.regime_switching_quanto.adoption.evidence_io import (
        canonical_json,
    )
    from finite_element_options.examples.regime_switching_quanto.adoption.uncertainty import (
        SCOPE_STATEMENT,
        canonical_uq_input_hash,
    )

    text = ARTIFACT.read_text(encoding="utf-8")
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == EXPECTED_ARTIFACT_SHA256
    artifact = json.loads(text)
    assert text == canonical_json(artifact) + "\n"
    assert artifact["study_input_hash"] == canonical_uq_input_hash()
    assert artifact["scope"] == SCOPE_STATEMENT
    assert artifact["decision"]["status"] == "retain_optional_adapter"
    assert artifact["decision"]["maturity"] == "experimental_optional_non_production"
    assert artifact["direct_reference"]["passed"] is True
    assert artifact["additive_sobol_recovery"]["passed"] is True
    assert artifact["propagation"]["sobol_validation"]["passed"] is True
    assert "sobol_intervals" in artifact["propagation"]
    assert "bounded to" not in text.lower()
    assert sorted(artifact["attribution_table"]) == [
        "data",
        "model_form",
        "monte_carlo",
        "numerical",
        "parameter",
    ]
    assert artifact["provenance"]["privacy_class"] == "public-synthetic"
    assert artifact["provenance"]["raw_samples_recorded"] is False
    assert (
        artifact["provenance"]["predecessor_hash_verification"]["quantlib_oracle"]["verified"]
        is True
    )
