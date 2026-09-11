"""Compare actual retained FEM payloads and complete histories against an exact Git baseline."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from types import SimpleNamespace
import weakref

import numpy as np
import scipy
import scipy.sparse.linalg as spla
import skfem as fem

from finite_element_options.core.config import Config
from finite_element_options.space.solver import SpaceSolver
from finite_element_options.time_integration.stepper import ThetaScheme

BASELINE = "c875010ac1e995fbaa00f575868cd81988fe22b7"


class ReactionForms:
    """P1 weak reaction operator: (du/dt,v) = -(c(t)u,v), u(0)=1."""

    def __init__(self, varying):
        self.varying = varying
        self.assemblies = 0

    def id_bil(self):
        return fem.BilinearForm(lambda u, v, w: u * v)

    def operator_form(self, time):
        self.assemblies += 1
        coefficient = 0.2 + 0.2 * time if self.varying else 0.2
        return fem.BilinearForm(lambda u, v, w: -coefficient * u * v)

    def b_lin(self):
        return fem.LinearForm(lambda v, w: 0 * v)

    def source_lin(self, time):
        return fem.LinearForm(lambda v, w: 0 * v)


def sparse_bytes(matrix):
    return sum(getattr(matrix, name).nbytes for name in ("data", "indices", "indptr"))


def represented_factor_bytes(factor):
    return (
        sparse_bytes(factor.L)
        + sparse_bytes(factor.U)
        + factor.perm_c.nbytes
        + factor.perm_r.nbytes
    )


def baseline_module(repo, output, relative, name):
    content = subprocess.check_output(
        ["git", "-C", str(repo), "show", f"{BASELINE}:{relative}"]
    )
    snapshot = output / "baseline-source" / relative
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes(content)
    spec = importlib.util.spec_from_file_location(name, snapshot)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, hashlib.sha256(content).hexdigest()


def resident_values(cache):
    # Evidence instrumentation reads the real containers; runtime uses public APIs.
    return cache.values() if isinstance(cache, dict) else cache._entries.values()


def measure(space_type, stepper_type, grid, *, varying=True, startup=False):
    payoff = SimpleNamespace(
        call_payoff=lambda x: np.ones_like(x),
        put_payoff=lambda x: np.ones_like(x),
        call=lambda *args: None,
        put=lambda *args: None,
    )
    spatial = space_type(
        fem.MeshLine(np.linspace(0, 1, 1025)),
        object(),
        payoff,
        True,
        forms=ReactionForms(varying),
        config=Config(elem=fem.ElementLineP1()),
    )
    options = (
        dict(startup_theta=1.0, startup_steps=2, startup_substeps=2) if startup else {}
    )
    solver = stepper_type(**options)
    original = spla.splu
    refs, samples = [], []

    class MeasuredFactor:
        def __init__(self, matrix):
            self.factor = original(matrix)
            self.represented_bytes = represented_factor_bytes(self.factor)

        def solve(self, rhs):
            frame = inspect.currentframe().f_back
            try:
                cached = list(resident_values(frame.f_locals["factorized_solvers"]))
                samples.append(
                    {
                        "resident_factors": len(cached),
                        "represented_factor_bytes": sum(
                            value.__self__.represented_bytes for value in cached
                        ),
                    }
                )
            finally:
                del frame
            return self.factor.solve(rhs)

    def factor(matrix):
        value = MeasuredFactor(matrix)
        refs.append(weakref.ref(value))
        return value

    spla.splu = factor
    try:
        values = solver.solve(grid, spatial)
    finally:
        spla.splu = original
    assert all(ref() is None for ref in refs), "factor escaped per-solve lifetime"
    expected = 1.0
    for step in solver._internal_steps(tuple(grid)):
        start = 0.2 + 0.2 * step.start if varying else 0.2
        end = 0.2 + 0.2 * step.end if varying else 0.2
        expected *= (1 - (1 - step.theta) * step.dt * start) / (
            1 + step.theta * step.dt * end
        )
    np.testing.assert_allclose(values[-1], expected, rtol=2e-12, atol=2e-12)
    matrices = list(resident_values(spatial._operator_matrix_cache))
    result = {
        "dofs": int(spatial.Vh.N),
        "output_steps": len(grid) - 1,
        "internal_steps": len(solver._internal_steps(tuple(grid))),
        "varying_coefficient": varying,
        "rannacher_startup": startup,
        "operator_assemblies": spatial.forms.assemblies,
        "resident_operators_after_return": len(matrices),
        "resident_operator_csr_bytes": sum(sparse_bytes(matrix) for matrix in matrices),
        "peak_resident_factors": max(row["resident_factors"] for row in samples),
        "peak_resident_factor_representation_bytes": max(
            row["represented_factor_bytes"] for row in samples
        ),
        "live_factors_after_return": sum(ref() is not None for ref in refs),
        "factorizations": solver.last_solve_diagnostics.factorization_count,
        "reuse_hits": solver.last_solve_diagnostics.factorization_reuse_count,
        "solution_history_bytes": values.nbytes,
        "max_residual": solver.last_solve_diagnostics.max_linear_residual_abs,
        "discrete_oracle_max_abs_error": float(np.max(np.abs(values[-1] - expected))),
        "continuous_oracle_max_abs_error": float(
            np.max(np.abs(values[-1] - np.exp(-(0.3 if varying else 0.2))))
        ),
        "history_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
    }
    if hasattr(spatial, "operator_cache_info"):
        result["operator_cache_info"] = asdict(spatial.operator_cache_info())
        result["linear_solve_cache_info"] = {
            key: value
            for key, value in solver.last_solve_diagnostics.to_public_dict().items()
            if key.startswith("factorization_cache_")
        }
    return result, values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    baseline_space, space_hash = baseline_module(
        repo,
        output,
        "src/finite_element_options/space/solver.py",
        "finite_element_options.space._cache_baseline",
    )
    baseline_stepper, stepper_hash = baseline_module(
        repo,
        output,
        "src/finite_element_options/time_integration/stepper.py",
        "finite_element_options.time_integration._cache_baseline",
    )
    cases = [(np.linspace(0, 1, steps + 1), True, False) for steps in (50, 200, 400)]
    cases += [
        (np.linspace(0, 1, 201), False, False),
        (np.array([0, 0.25, 0.75, 1]), False, False),
        (np.linspace(0, 1, 21), False, True),
    ]
    rows = []
    for grid, varying, startup in cases:
        base, before = measure(
            baseline_space.SpaceSolver,
            baseline_stepper.ThetaScheme,
            grid,
            varying=varying,
            startup=startup,
        )
        current, after = measure(
            SpaceSolver, ThetaScheme, grid, varying=varying, startup=startup
        )
        np.testing.assert_array_equal(before, after)
        assert current["resident_operators_after_return"] <= 2
        assert current["peak_resident_factors"] <= 2
        rows.append(
            {"baseline": base, "bounded": current, "full_history_bitwise_equal": True}
        )
    report = {
        "baseline_ref": BASELINE,
        "baseline_space_sha256": space_hash,
        "baseline_stepper_sha256": stepper_hash,
        "source_head": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_hashes": {
            str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                repo / "src/finite_element_options/space/solver.py",
                repo / "src/finite_element_options/time_integration/stepper.py",
                repo / "src/finite_element_options/core/operator_cache.py",
            ]
        },
        "versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_fem": fem.__version__,
        },
        "platform": platform.platform(),
        "dtype": "float64",
        "device": "CPU",
        "jit": False,
        "threads": {
            key: os.environ.get(key)
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "rows": rows,
        "limits": [
            "Resource instrumentation only; no speedup or RSS claim.",
            "Factor bytes are exported L/U/permutation representation, not native allocator usage.",
            "Mass, initial stiffness, active matrices/factor, solution history and scalar diagnostics are outside resident-cache counts.",
            "Endpoint assembly and distinct LU counts are unchanged for these workloads.",
            "Default cache size 2; other working sets may refactor after eviction.",
        ],
    }
    (output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "cases": len(rows),
                "all_histories_bitwise_equal": True,
                "output": str(output / "benchmark.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
