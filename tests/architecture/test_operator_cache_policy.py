"""Keep the declared numerical retention policy aligned with both public owners."""

import inspect
import json
from pathlib import Path

from finite_element_options.space.solver import SpaceSolver
from finite_element_options.time_integration.stepper import ThetaScheme


def test_architecture_default_capacities_and_executable_owners_match():
    root = Path(__file__).resolve().parents[2]
    policy = json.loads((root / "docs/ARCHITECTURE.yaml").read_text())["architecture"][
        "operator_cache_policy"
    ]
    assert policy["owner"] == "finite_element_options.core.operator_cache"
    assert (
        policy["operator_cache_default_entries"]
        == inspect.signature(SpaceSolver).parameters["operator_cache_size"].default
        == 2
    )
    assert (
        policy["factorization_cache_default_entries"]
        == inspect.signature(ThetaScheme).parameters["factorization_cache_size"].default
        == 2
    )
    assert callable(SpaceSolver.invalidate_operator_cache)
    assert callable(SpaceSolver.operator_cache_info)
    assert all((root / path).is_file() for path in policy["fitness_tests"])
    assert (root / policy["benchmark"]).is_file()
