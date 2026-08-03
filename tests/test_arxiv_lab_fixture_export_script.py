"""Tests for the arxiv-lab FEM fixture export script."""

from __future__ import annotations

from hashlib import sha256
import json
from math import isclose
from numbers import Real
from pathlib import Path
import subprocess
import sys

import pytest

from finite_element_options.validation.black_scholes_parity import (
    FEM_BS_001_PROBLEM_SPEC_PATH,
    FEM_BS_001_RESULT_EXPORT_PATH,
    FEMParityConvergenceRow,
    build_fixture_config_hash,
)


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _assert_structural_exact_numerical_close(left: object, right: object) -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_structural_exact_numerical_close(left[key], right[key])
        return
    if isinstance(left, list) and isinstance(right, list):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_structural_exact_numerical_close(left_item, right_item)
        return
    if (
        isinstance(left, Real)
        and not isinstance(left, bool)
        and isinstance(right, Real)
        and not isinstance(right, bool)
    ):
        assert isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-12)
        return
    assert left == right


def test_arxiv_lab_fixture_export_script_regenerates_public_fixture_paths(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/export_arxiv_lab_black_scholes_fixture.py",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    output = json.loads(completed.stdout)
    spec_path = tmp_path / "fem_bs_001" / "problem_spec.json"
    result_path = tmp_path / "fem_bs_001" / "result_export.json"

    assert output["benchmark_id"] == "fem-bs-001"
    assert output["status"] == "converged"
    assert output["generated"] == [
        {"path": str(spec_path), "sha256": _file_sha256(spec_path)},
        {"path": str(result_path), "sha256": _file_sha256(result_path)},
    ]
    spec_payload = json.loads(spec_path.read_text())
    result_payload = json.loads(result_path.read_text())
    checked_in_spec = json.loads(FEM_BS_001_PROBLEM_SPEC_PATH.read_text())
    checked_in_result = json.loads(FEM_BS_001_RESULT_EXPORT_PATH.read_text())

    assert (spec_path.parent / spec_payload["result_export_uri"]).resolve() == (
        result_path.resolve()
    )
    comparable_spec = {
        key: value
        for key, value in spec_payload.items()
        if key not in {"contract_id", "result_export_uri"}
    }
    comparable_checked_in_spec = {
        key: value
        for key, value in checked_in_spec.items()
        if key not in {"contract_id", "result_export_uri"}
    }
    _assert_structural_exact_numerical_close(
        comparable_spec, comparable_checked_in_spec
    )
    _assert_structural_exact_numerical_close(result_payload, checked_in_result)


def test_arxiv_lab_numerical_rows_are_canonicalized_before_hashing() -> None:
    linux_scipy_118 = FEMParityConvergenceRow(
        refinement_level=6,
        time_steps=80,
        degrees_of_freedom=129,
        observed_price=10.450662854138187,
        expected_price=10.450583572185565,
        absolute_error=7.928195262252302e-05,
        relative_error=7.586366069884701e-06,
        observed_delta=0.6359943247917581,
        expected_delta=0.6368306511756192,
        delta_absolute_error=0.0008363263838611079,
        observed_gamma=0.018753215754558593,
        expected_gamma=0.01876201734584689,
        gamma_absolute_error=8.801591288298827e-06,
    )
    alternate_superlu_roundoff = FEMParityConvergenceRow(
        refinement_level=6,
        time_steps=80,
        degrees_of_freedom=129,
        observed_price=10.45066285413824,
        expected_price=10.450583572185565,
        absolute_error=7.928195267581373e-05,
        relative_error=7.586366074984005e-06,
        observed_delta=0.6359943247917601,
        expected_delta=0.6368306511756192,
        delta_absolute_error=0.0008363263838591095,
        observed_gamma=0.018753215754558,
        expected_gamma=0.01876201734584689,
        gamma_absolute_error=8.801591288892102e-06,
    )

    assert (
        linux_scipy_118.to_public_dict() == alternate_superlu_roundoff.to_public_dict()
    )
    assert linux_scipy_118.gamma_absolute_error != pytest.approx(
        alternate_superlu_roundoff.gamma_absolute_error, abs=1e-16
    )


def test_arxiv_lab_fixture_export_contains_recomputable_scientific_provenance(
    tmp_path: Path,
) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/export_arxiv_lab_black_scholes_fixture.py",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    spec_payload = json.loads(
        (tmp_path / "fem_bs_001" / "problem_spec.json").read_text()
    )
    result_payload = json.loads(
        (tmp_path / "fem_bs_001" / "result_export.json").read_text()
    )
    result_hash = result_payload.pop("result_hash")

    assert spec_payload["contract_version"] == "fem-parity-contract/v1"
    assert result_payload["format_version"] == "fem-bs-oracle-result-v1"
    assert result_hash == build_fixture_config_hash(result_payload)
    assert result_payload["provenance"] == spec_payload["provenance"]
    assert result_payload["provenance"]["consumer"] == "arxiv-implementation-lab"
    assert (
        result_payload["provenance"]["fixture_owner"]
        == "googa27/finite_element_options"
    )
    assert result_payload["privacy_class"] == "public_synthetic"
    assert result_payload["measure"] == "risk_neutral"
    assert result_payload["numeraire"] == "money_market_account"
    assert result_payload["pde_convention"]["operator_sign"] == (
        "forward_tau_generator_minus_discount"
    )
    assert result_payload["pde_convention"]["initial_condition_tau_zero"] == (
        "u(S,0)=max(S-K,0)"
    )
    assert result_payload["boundaries"] == spec_payload["boundaries"]
    assert result_payload["boundaries"] == [
        {
            "condition_type": "dirichlet",
            "enforced_nodes": 1,
            "expression": "0",
            "location": "S=0",
        },
        {
            "condition_type": "dirichlet",
            "enforced_nodes": 1,
            "expression": "linear_growth",
            "location": "S=S_max",
        },
    ]
    assert result_payload["units"] == {
        "delta": "value_per_underlying",
        "gamma": "value_per_underlying_squared",
        "rate": "1/year",
        "spot": "CLP",
        "strike": "CLP",
        "time": "year",
        "underlying": "CLP",
        "value": "CLP",
        "volatility": "annualized_decimal",
    }
    assert (
        result_payload["summary"]["observed_price"]
        == result_payload["rows"][-1]["observed_price"]
    )
    assert (
        result_payload["summary"]["price_absolute_error"]
        <= result_payload["summary"]["price_tolerance_absolute"]
    )
    assert (
        result_payload["summary"]["delta_absolute_error"]
        <= result_payload["summary"]["delta_tolerance_absolute"]
    )
    assert (
        result_payload["summary"]["gamma_absolute_error"]
        <= result_payload["summary"]["gamma_tolerance_absolute"]
    )
