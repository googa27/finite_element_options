"""Iminuit profile-likelihood identifiability tests for quanto adoption."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from finite_element_options.examples.regime_switching_quanto.adoption.evidence_io import (
    canonical_json,
    canonical_json_sha256,
)
from finite_element_options.examples.regime_switching_quanto.adoption.identifiability import (
    CalibrationCase,
    ParameterBounds,
    ProfileGrid,
    WeightedQuantoCalibrationObjective,
    canonical_identifiability_input_hash,
    default_identifiability_cases,
    run_iminuit_identifiability,
    run_iminuit_identifiability_study,
)

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_iminuit_identifiability.py"
ARTIFACT = (
    ROOT / "docs" / "evidence" / "regime_switching_quanto_iminuit_identifiability_2026-09-04.json"
)
ARTIFACT_SHA256 = "6294b52e9d6aa26aeda39a1809486272223d41ecc7a00e42e670f5dcbba39a3b"


def _case(case_id: str) -> CalibrationCase:
    cases = {case.case_id: case for case in default_identifiability_cases()}
    return cases[case_id]


def test_identified_case_passes_all_gates_and_estimates_truth() -> None:
    """The nonzero-FX-vol surface is identified and estimates synthetic truth."""

    pytest.importorskip("iminuit")
    result = run_iminuit_identifiability(_case("identified_quanto_surface")).to_dict()

    assert result["identification"]["identified"] is True
    assert result["identification"]["reasons"] == []
    values = result["minimum"]["values"]
    truth = result["case"]["synthetic_truth_metadata"]
    assert values["equity_vol"] == pytest.approx(truth["equity_vol"], abs=2.0e-4)
    assert values["correlation"] == pytest.approx(truth["correlation"], abs=2.0e-3)
    assert result["minimum"]["fmin_flags"]["has_accurate_covar"] is True
    assert result["hesse"]["covariance_quality"]["positive_definite"] is True
    for parameter in ("equity_vol", "correlation"):
        assert result["minos"]["parameters"][parameter]["is_valid"] is True
        evidence = result["profiles"]["parameters"][parameter]["evidence"]
        assert evidence["finite_stable"] is True
        assert evidence["lower_crosses_delta_chi2_1"] is True
        assert evidence["upper_crosses_delta_chi2_1"] is True


def test_weak_rho_case_has_point_estimate_but_is_not_identified() -> None:
    """The fx_vol=0 case can estimate sigma while rejecting rho identification."""

    pytest.importorskip("iminuit")
    result = run_iminuit_identifiability(_case("weak_rho_fxvol_zero")).to_dict()

    assert result["identification"]["identified"] is False
    assert result["minimum"]["values"]["equity_vol"] == pytest.approx(0.23, abs=2.0e-4)
    assert any("HESSE" in reason for reason in result["identification"]["reasons"])
    assert any("MINOS" in reason for reason in result["identification"]["reasons"])
    rho_profile = result["profiles"]["parameters"]["correlation"]["evidence"]
    assert rho_profile["max_delta_chi2"] == pytest.approx(0.0, abs=1.0e-12)
    assert rho_profile["lower_crosses_delta_chi2_1"] is False
    assert rho_profile["upper_crosses_delta_chi2_1"] is False


def test_finite_difference_objective_checks_identified_and_weak_truth() -> None:
    """Objective gradients/curvatures distinguish identified and structural-rho cases."""

    identified = _case("identified_quanto_surface")
    ident_obj = WeightedQuantoCalibrationObjective(identified)
    ident_fd = _central_difference(
        ident_obj,
        identified.synthetic_truth["equity_vol"],
        identified.synthetic_truth["correlation"],
        identified.finite_difference_step,
    )
    assert abs(ident_fd["equity_vol"]["gradient"]) < 1.0e-2
    assert abs(ident_fd["correlation"]["gradient"]) < 1.0e-2
    assert ident_fd["equity_vol"]["curvature"] > 0.0
    assert ident_fd["correlation"]["curvature"] > 0.0

    weak = _case("weak_rho_fxvol_zero")
    weak_obj = WeightedQuantoCalibrationObjective(weak)
    weak_fd = _central_difference(
        weak_obj,
        weak.synthetic_truth["equity_vol"],
        weak.synthetic_truth["correlation"],
        weak.finite_difference_step,
    )
    assert abs(weak_fd["equity_vol"]["gradient"]) < 1.0e-2
    assert weak_fd["equity_vol"]["curvature"] > 0.0
    assert weak_fd["correlation"]["gradient"] == pytest.approx(0.0, abs=1.0e-12)
    assert weak_fd["correlation"]["curvature"] == pytest.approx(0.0, abs=1.0e-12)


def test_out_of_bounds_nonfinite_and_invalid_targets_fail_closed() -> None:
    """Invalid inputs raise early and invalid evaluations return infinity diagnostics."""

    case = _case("identified_quanto_surface")
    objective = WeightedQuantoCalibrationObjective(case)

    out = objective.evaluate(0.01, -0.4)
    assert math.isinf(float(out.chi2))
    assert out.finite is False
    assert out.diagnostics["reason"] == "out_of_bounds_parameter"
    assert out.to_dict()["chi2"] is None

    nonfinite = objective.evaluate(math.nan, -0.4)
    assert math.isinf(float(nonfinite.chi2))
    assert nonfinite.finite is False
    assert nonfinite.diagnostics["reason"] == "nonfinite_parameter"
    assert "nan" not in canonical_json(nonfinite.to_dict()).lower()

    target = case.targets[0]
    with pytest.raises(ValueError, match="price_std must be positive"):
        replace(target, price_std=0.0)
    with pytest.raises(ValueError, match="target_price must be finite"):
        replace(target, target_price=math.nan)


def test_bound_contact_case_is_rejected() -> None:
    """A minimum close to an explicit bound is non-identified even with a point estimate."""

    pytest.importorskip("iminuit")
    case = _case("identified_quanto_surface")
    bounded = replace(
        case,
        bounds=ParameterBounds(equity_vol=(0.05, 0.23005), correlation=(-0.95, 0.95)),
        profile_grids=(
            ProfileGrid("equity_vol", 0.18, 0.23005, 21),
            ProfileGrid("correlation", -0.80, 0.05, 21),
        ),
    )

    result = run_iminuit_identifiability(bounded).to_dict()

    assert result["identification"]["identified"] is False
    assert result["boundary_contact"]["any_near_bound"] is True
    assert any("bound" in reason for reason in result["identification"]["reasons"])


def test_result_profiles_typed_failures_and_json_are_deterministic() -> None:
    """Contracts serialize canonically without raw iminuit object representations."""

    pytest.importorskip("iminuit")
    result = run_iminuit_identifiability(_case("weak_rho_fxvol_zero")).to_dict()
    encoded = canonical_json(result)

    assert encoded == canonical_json(result)
    assert canonical_json_sha256(result) == canonical_json_sha256(result)
    assert "<iminuit" not in encoded
    assert "Minuit(" not in encoded
    assert "FMin(" not in encoded
    assert "nan" not in encoded.lower()
    assert result["minos"]["parameters"]["correlation"]["status"] in {
        "available",
        "missing",
    }
    if result["optimizer"]["failure"] is not None:
        assert set(result["optimizer"]["failure"]) == {"type", "message"}


def test_profile_failures_suppress_raw_optimizer_paths_and_nonfinite_text() -> None:
    """Exceptional profile traces must remain deterministic and privacy-safe."""

    from finite_element_options.examples.regime_switching_quanto.adoption.identifiability.adapter import (
        _profile_diagnostics,
    )

    class FailingProfile:
        def mnprofile(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("FMin(/home/example/private.py) nan")

    profiles = _profile_diagnostics(
        _case("identified_quanto_surface"),
        FailingProfile(),
        {"values": {"equity_vol": 0.23, "correlation": -0.40}},
    )
    encoded = canonical_json(profiles)

    assert "FMin(" not in encoded
    assert "/home/" not in encoded
    assert "nan" not in encoded.lower()
    for parameter in ("equity_vol", "correlation"):
        failure = profiles["parameters"][parameter]["failure"]
        assert failure == {
            "type": "RuntimeError",
            "message": "optimizer exception captured; raw implementation details suppressed",
        }


def test_base_facade_and_contracts_import_with_iminuit_blocked() -> None:
    """Facades and contracts must not import iminuit in the base profile."""

    code = """
    import builtins
    import importlib
    import sys

    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name.split('.')[0] == 'iminuit':
            raise ModuleNotFoundError('blocked optional dependency: ' + name, name='iminuit')
        return original_import(name, *args, **kwargs)

    builtins.__import__ = guarded_import
    for name in (
        'finite_element_options',
        'finite_element_options.examples.regime_switching_quanto.adoption',
        'finite_element_options.examples.regime_switching_quanto.adoption.identifiability',
        'finite_element_options.examples.regime_switching_quanto.adoption.identifiability.contracts',
    ):
        importlib.import_module(name)
    assert not any(name.split('.')[0] == 'iminuit' for name in sys.modules)
    """
    probe = _run_python(code)
    assert probe.returncode == 0, probe.stdout + probe.stderr


def test_execution_missing_extra_names_exact_install_hint_and_version() -> None:
    """The script fail-closed hint names finite-element-options[identifiability]."""

    code = f"""
    import importlib.abc
    import importlib.util
    import sys

    class BlockIminuit(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname == 'iminuit' or fullname.startswith('iminuit.'):
                raise ModuleNotFoundError('blocked optional dependency: ' + fullname, name='iminuit')
            return None

    sys.meta_path.insert(0, BlockIminuit())
    spec = importlib.util.spec_from_file_location('run_iminuit_identifiability', {str(SCRIPT)!r})
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raise SystemExit(module.main(['--output', '/tmp/blocked-iminuit.json']))
    """
    probe = _run_python(code)
    assert probe.returncode == 2
    assert "finite-element-options[identifiability]" in probe.stderr
    assert "iminuit>=2.32,<3" in probe.stderr


def test_committed_artifact_is_canonical_hash_bound_and_scoped() -> None:
    """Static evidence remains guarded even when iminuit is absent from the base profile."""

    raw = ARTIFACT.read_bytes()
    payload = json.loads(raw)

    assert hashlib.sha256(raw).hexdigest() == ARTIFACT_SHA256
    assert raw.decode("utf-8") == canonical_json(payload) + "\n"
    assert payload["study_input_hash"] == canonical_identifiability_input_hash()
    assert payload["summary"]["decisions"] == {
        "identified_quanto_surface": True,
        "weak_rho_fxvol_zero": False,
    }
    assert "not observed/live market" in payload["scope"]
    assert "production calibration" in payload["scope"]


def test_runtime_study_is_deterministic_and_matches_expected_decisions() -> None:
    """The live optional study records both expected public-synthetic decisions."""

    pytest.importorskip("iminuit")
    study = run_iminuit_identifiability_study().to_dict()
    artifact_hash = canonical_json_sha256(study)

    assert artifact_hash == canonical_json_sha256(study)
    assert study["summary"]["all_expected_decisions_passed"] is True
    assert study["summary"]["decisions"] == {
        "identified_quanto_surface": True,
        "weak_rho_fxvol_zero": False,
    }
    assert "Public-synthetic" in study["scope"]
    assert "not observed/live market" in study["scope"]
    assert "production calibration" in study["scope"]
    assert set(study["summary"]["case_input_hashes"]) == set(study["summary"]["decisions"])


def _central_difference(
    objective: WeightedQuantoCalibrationObjective,
    equity_vol: float,
    correlation: float,
    step: float,
) -> dict[str, dict[str, float]]:
    base = {"equity_vol": equity_vol, "correlation": correlation}
    base_value = objective(**base)
    result: dict[str, dict[str, float]] = {}
    for parameter in ("equity_vol", "correlation"):
        lower = dict(base)
        upper = dict(base)
        lower[parameter] -= step
        upper[parameter] += step
        f_minus = objective(**lower)
        f_plus = objective(**upper)
        result[parameter] = {
            "gradient": (f_plus - f_minus) / (2.0 * step),
            "curvature": (f_plus - 2.0 * base_value + f_minus) / (step * step),
        }
    return result


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
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
