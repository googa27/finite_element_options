from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


def test_ai_and_semantic_hierarchy_policy() -> None:
    if importlib.util.find_spec("yaml") is None:
        pytest.skip(
            "Dedicated policy workflow installs and executes PyYAML-backed gate"
        )
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "check_ai_hierarchy_policy.py")],
        check=False,
        text=True,
        capture_output=True,
        cwd=root,
    )
    assert result.returncode == 0, result.stdout + result.stderr
