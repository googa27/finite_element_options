from __future__ import annotations

import subprocess
import sys


def test_ai_and_semantic_hierarchy_policy() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_ai_hierarchy_policy.py"],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
