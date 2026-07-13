from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKER = REPO_ROOT / "scripts" / "check_portfolio_architecture.py"
CONTRACT = REPO_ROOT / "docs" / "ARCHITECTURE.yaml"


def load_checker() -> Any:
    spec = importlib.util.spec_from_file_location(
        "check_portfolio_architecture", CHECKER
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_portfolio_architecture_contract(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(CHECKER)],
        check=False,
        text=True,
        capture_output=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_portfolio_architecture_contract_rejects_invalid_limit_contract() -> None:
    checker = load_checker()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    del contract["limits"]["max_immediate_runtime_entries"]

    errors = checker.validate_contract(contract)

    assert errors == ["limits.max_immediate_runtime_entries is required"]


def test_portfolio_architecture_contract_wraps_os_errors(tmp_path: Path) -> None:
    checker = load_checker()
    unreadable_contract = tmp_path / "ARCHITECTURE.yaml"
    unreadable_contract.mkdir()
    checker.ROOT = tmp_path
    checker.CONTRACT = unreadable_contract

    try:
        checker.load_contract()
    except checker.ContractLoadError as exc:
        assert "could not read ARCHITECTURE.yaml" in str(exc)
    else:  # pragma: no cover - clearer assertion message than bare failure
        raise AssertionError("expected ContractLoadError")
