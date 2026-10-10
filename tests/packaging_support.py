"""Checked child execution shared by packaging tests, never runtime code."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def run_checked(
    command: list[str], *, cwd: Path, env: dict[str, str] | None = None
) -> str:
    """Isolate source/home selection and preserve genuine child failures."""
    child_env = dict(os.environ if env is None else env)
    for name in ("PYTHONPATH", "PYTHONHOME"):
        child_env.pop(name, None)
    result = subprocess.run(
        command,
        cwd=cwd,
        env=child_env,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode:
        raise AssertionError(
            json.dumps(
                {
                    "argv": command,
                    "cwd": str(cwd),
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                },
                sort_keys=True,
            )
        )
    return result.stdout + result.stderr
