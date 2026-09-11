"""Preserve the pure grid owner and existing private stepper aliases."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from finite_element_options.time_integration import stepper, time_grid

ROOT = Path(__file__).resolve().parents[2]
GRID = ROOT / "src/finite_element_options/time_integration/time_grid.py"


def test_time_grid_is_an_inward_dependency_with_compatible_aliases() -> None:
    """Keep pure grid validation independent of its numerical consumers."""
    policy = json.loads((ROOT / "docs/ARCHITECTURE.yaml").read_text())
    assert policy["theta_startup_counts"]["dependency_direction"] == (
        "stepper -> time_grid; pure validators never import numerical consumers"
    )
    for node in ast.walk(ast.parse(GRID.read_text())):
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0
            assert not (node.module or "").startswith("finite_element_options")
        elif isinstance(node, ast.Import):
            assert all(
                not alias.name.startswith("finite_element_options")
                for alias in node.names
            )
    for name in (
        "_validate_theta",
        "_validate_startup_count",
        "_validate_time_grid",
        "_canonical_local_steps",
    ):
        assert getattr(stepper, name) is getattr(time_grid, name)
