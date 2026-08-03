"""Tests for the arxiv-lab FEM fixture export script."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

from finite_element_options.validation.black_scholes_parity import (
    FEM_BS_001_PROBLEM_SPEC_PATH,
    FEM_BS_001_RESULT_EXPORT_PATH,
    build_fixture_config_hash,
)


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


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
    assert json.loads(spec_path.read_text()) == json.loads(
        FEM_BS_001_PROBLEM_SPEC_PATH.read_text()
    )
    assert json.loads(result_path.read_text()) == json.loads(
        FEM_BS_001_RESULT_EXPORT_PATH.read_text()
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

    spec_payload = json.loads((tmp_path / "fem_bs_001" / "problem_spec.json").read_text())
    result_payload = json.loads((tmp_path / "fem_bs_001" / "result_export.json").read_text())
    result_hash = result_payload.pop("result_hash")

    assert spec_payload["contract_version"] == "fem-parity-contract/v1"
    assert result_payload["format_version"] == "fem-bs-oracle-result-v1"
    assert result_hash == build_fixture_config_hash(result_payload)
    assert result_payload["provenance"] == spec_payload["provenance"]
    assert result_payload["provenance"]["consumer"] == "arxiv-implementation-lab"
    assert result_payload["provenance"]["fixture_owner"] == "googa27/finite_element_options"
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
    assert result_payload["summary"]["observed_price"] == result_payload["rows"][-1][
        "observed_price"
    ]
    assert result_payload["summary"]["price_absolute_error"] <= result_payload["summary"][
        "price_tolerance_absolute"
    ]
    assert result_payload["summary"]["delta_absolute_error"] <= result_payload["summary"][
        "delta_tolerance_absolute"
    ]
    assert result_payload["summary"]["gamma_absolute_error"] <= result_payload["summary"][
        "gamma_tolerance_absolute"
    ]
