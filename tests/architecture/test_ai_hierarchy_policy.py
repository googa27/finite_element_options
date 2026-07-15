from __future__ import annotations

import importlib.util
import subprocess
import sys

import pytest


def test_ai_and_semantic_hierarchy_policy() -> None:
    if importlib.util.find_spec("yaml") is None:
        pytest.skip(
            "Dedicated policy workflow installs and executes PyYAML-backed gate"
        )
    result = subprocess.run(
        [sys.executable, "scripts/check_ai_hierarchy_policy.py"],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
