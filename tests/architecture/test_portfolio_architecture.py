from __future__ import annotations

import subprocess
import sys


def test_portfolio_architecture_contract() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_portfolio_architecture.py"],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
