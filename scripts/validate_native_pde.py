#!/usr/bin/env python3
"""Execute maintained native PDE demonstrations and fail closed on bad evidence.

These serial public-synthetic fixtures validate DOLFINx/Dedalus demonstrations,
not the package's optional pricing backend or market calibration. Native imports
are mandatory only for ``solve``; metadata unit tests are not native run proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


VERSIONS = {"fenicsx": "0.11.0", "dedalus": "3.0.5"}
ERROR_LIMITS = {"fenicsx": 1e-3, "dedalus": 2e-6}
REPLAY_COORDINATE_ATOL = 1e-12
REPLAY_FIELD_ATOL = 1e-12
REPLAY_FIELD_RTOL = 1e-10


def _finite(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def _count(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_report(report: dict[str, Any]) -> None:
    """Check measurements and identity; this alone is not actor attestation."""
    backend = report["backend"]
    if (
        type(report["schema"]) is not int
        or report["schema"] != 1
        or backend not in VERSIONS
    ):
        raise ValueError("unknown native evidence schema/backend")
    if report["phase"] not in ("original", "relocated"):
        raise ValueError("unknown execution phase")
    prefix = Path(report["prefix"])
    if not prefix.is_absolute() or prefix == Path("/"):
        raise ValueError("native prefix must be a specific absolute directory")
    if (
        report["script_sha256"]
        != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    ):
        raise ValueError("report is not bound to this reviewed script")
    if report["runtime"] != {backend: VERSIONS[backend]}:
        raise ValueError("native runtime differs from the reviewed profile")
    origins = report["origins"]
    if not {backend, "numpy"}.issubset(origins):
        raise ValueError("native and NumPy module origins are required")
    for origin in origins.values():
        if not Path(origin).resolve().is_relative_to(prefix.resolve()):
            raise ValueError("module origin escapes the declared native prefix")
    levels = report["levels"]
    resolutions = [16, 32, 64] if backend == "fenicsx" else [10, 20, 40]
    if len(levels) != 3:
        raise ValueError("exactly three executed refinement levels are required")
    errors = []
    scales = []
    for row, resolution in zip(levels, resolutions, strict=True):
        if _count(row["resolution"], "resolution") != resolution:
            raise ValueError("missing, duplicate or unexpected refinement level")
        scale = _finite(row["scale"], "refinement scale")
        expected = 1 / resolution if backend == "fenicsx" else 0.5 / resolution
        if not math.isclose(scale, expected, rel_tol=0, abs_tol=1e-15):
            raise ValueError("refinement scale disagrees with the executed resolution")
        error = _finite(row["error_l2"], "L2 error")
        if error <= 0:
            raise ValueError("positive measured errors are needed to establish order")
        _count(row["iterations"], "iterations")
        if backend == "fenicsx":
            _count(row["ksp_reason"], "KSP converged reason")
            if not 0 <= _finite(row["boundary_max"], "boundary error") <= 1e-12:
                raise ValueError("Dirichlet boundary acceptance failed")
        else:
            if row["iterations"] != resolution:
                raise ValueError("heat timestep count differs from the executed grid")
            if not math.isclose(
                _finite(row["final_time"], "final time"), 0.5, rel_tol=0, abs_tol=1e-12
            ):
                raise ValueError("heat solve did not reach the analytical final time")
            if abs(_finite(row["mean"], "heat mean")) > 1e-12:
                raise ValueError("periodic heat mean is not conserved")
            if not 0 < _finite(row["energy_ratio"], "heat energy ratio") <= 1:
                raise ValueError("positive-diffusivity heat energy must decay")
        errors.append(error)
        scales.append(scale)
    for index in (0, 1):
        order = math.log(errors[index] / errors[index + 1]) / math.log(
            scales[index] / scales[index + 1]
        )
        if not math.isfinite(order) or order < 1.8:
            raise ValueError(f"analytical refinement order {order:.6g} is below 1.8")
    if errors[-1] > ERROR_LIMITS[backend]:
        raise ValueError("finest analytical L2 error exceeds the reviewed ceiling")


def validate_pair(original: dict[str, Any], relocated: dict[str, Any]) -> None:
    """Require an independently located consumer with matching measured output."""
    validate_report(original)
    validate_report(relocated)
    if original["phase"] != "original" or relocated["phase"] != "relocated":
        raise ValueError("producer/consumer phases must be original/relocated")
    if Path(original["prefix"]).resolve() == Path(relocated["prefix"]).resolve():
        raise ValueError("relocated consumer must have a different absolute prefix")
    if original["backend"] != relocated["backend"]:
        raise ValueError("producer/consumer backend mismatch")
    for before, after in zip(original["levels"], relocated["levels"], strict=True):
        if before.keys() != after.keys():
            raise ValueError("relocated measurement fields changed")
        for name, value in before.items():
            if not math.isclose(value, after[name], rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError(f"relocated measurement changed: {name}")
    if "measurements" not in original or "measurements" not in relocated:
        raise ValueError("validated measurement arrays are required for replay")
    import numpy as np

    arrays = [report["measurements"] for report in (original, relocated)]
    # DOLFINx may number equivalent DOFs differently at a relocated prefix.
    # Sort complete points, applying each point permutation to every field.
    orders = [np.lexsort((values["y"], values["x"])) for values in arrays]
    for name in ("x", "y", "exact", "u"):
        before, after = [
            values[name][order] for values, order in zip(arrays, orders, strict=True)
        ]
        rtol = REPLAY_FIELD_RTOL if name == "u" else 0
        atol = REPLAY_FIELD_ATOL if name == "u" else REPLAY_COORDINATE_ATOL
        if before.shape != after.shape or not np.allclose(
            before, after, rtol=rtol, atol=atol
        ):
            raise ValueError(f"relocated replay field differs: {name}")


def _origins(backend: str, modules: dict[str, Any]) -> dict[str, str]:
    origins = {
        name: str(Path(module.__file__).resolve()) for name, module in modules.items()
    }
    required = (
        ("libdolfinx", "libpetsc", "libmpi")
        if backend == "fenicsx"
        else ("libfftw", "libmpi")
    )
    loaded = [
        line.split()[-1]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if line.split()[-1].startswith("/")
    ]
    for library in required:
        matches = sorted(
            {
                str(Path(path).resolve())
                for path in loaded
                if Path(path).name.startswith(library)
            }
        )
        if not matches:
            raise ValueError(f"required native shared library is not loaded: {library}")
        for index, path in enumerate(matches):
            origins[f"{library}:{index}"] = path
    return origins


def _poisson() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    import dolfinx
    from dolfinx import fem, mesh
    from dolfinx.fem.petsc import LinearProblem
    import mpi4py
    from mpi4py import MPI
    import numpy as np
    import petsc4py
    from petsc4py import PETSc
    import ufl

    if dolfinx.__version__ != VERSIONS["fenicsx"]:
        raise ValueError("DOLFINx version is not the reviewed 0.11.0")
    levels = []
    arrays = {}
    for cells in (16, 32, 64):
        domain = mesh.create_unit_square(MPI.COMM_SELF, cells, cells)
        space = fem.functionspace(domain, ("Lagrange", 1))
        trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
        x = ufl.SpatialCoordinate(domain)
        exact = ufl.sin(math.pi * x[0]) * ufl.sin(math.pi * x[1])
        dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 8})
        fdim = domain.topology.dim - 1
        domain.topology.create_connectivity(fdim, domain.topology.dim)
        facets = mesh.exterior_facet_indices(domain.topology)
        dofs = fem.locate_dofs_topological(space, fdim, facets)
        bc = fem.dirichletbc(PETSc.ScalarType(0), dofs, space)
        problem = LinearProblem(
            ufl.inner(ufl.grad(trial), ufl.grad(test)) * dx,
            2 * math.pi**2 * exact * test * dx,
            bcs=[bc],
            petsc_options_prefix=f"native_poisson_{cells}_",
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "ksp_error_if_not_converged": True,
            },
        )
        solution = problem.solve()
        solution.x.scatter_forward()
        reason = int(problem.solver.getConvergedReason())
        if reason <= 0:
            raise ValueError(f"PETSc KSP failed with reason {reason}")
        error_sq = domain.comm.allreduce(
            fem.assemble_scalar(fem.form((solution - exact) ** 2 * dx)), op=MPI.SUM
        )
        values = solution.x.array.copy()
        if np.iscomplexobj(values):
            raise ValueError("this public fixture requires a real PETSc profile")
        coordinates = space.tabulate_dof_coordinates()
        levels.append(
            {
                "resolution": cells,
                "scale": 1 / cells,
                "error_l2": math.sqrt(float(error_sq)),
                "iterations": int(problem.solver.getIterationNumber()),
                "ksp_reason": reason,
                "boundary_max": float(np.max(np.abs(values[dofs]))),
            }
        )
        arrays = {
            "x": coordinates[:, 0],
            "y": coordinates[:, 1],
            "u": values,
            "exact": np.sin(math.pi * coordinates[:, 0])
            * np.sin(math.pi * coordinates[:, 1]),
        }
    return (
        levels,
        arrays,
        _origins(
            "fenicsx",
            {
                "fenicsx": dolfinx,
                "numpy": np,
                "mpi4py": mpi4py,
                "petsc4py": petsc4py,
                "ufl": ufl,
            },
        ),
    )


def _heat() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    import dedalus
    import dedalus.public as d3
    import mpi4py
    from mpi4py import MPI
    import numpy as np

    if dedalus.__version__ != VERSIONS["dedalus"]:
        raise ValueError("Dedalus version is not the reviewed 3.0.5")
    levels = []
    arrays = {}
    for steps in (10, 20, 40):
        coordinate = d3.Coordinate("x")
        distributor = d3.Distributor(coordinate, dtype=np.float64, comm=MPI.COMM_SELF)
        basis = d3.RealFourier(coordinate, size=64, bounds=(0, 2 * math.pi))
        field = distributor.Field(name="u", bases=basis)
        x = distributor.local_grid(basis)
        initial = np.sin(x) + 0.25 * np.cos(2 * x)
        field["g"] = initial

        def derivative(value):
            return d3.Differentiate(value, coordinate)

        problem = d3.IVP([field], namespace={"u": field, "D": 0.1, "dx": derivative})
        problem.add_equation("dt(u) - D*dx(dx(u)) = 0")
        solver = problem.build_solver(d3.RK222)
        for _ in range(steps):
            solver.step(0.5 / steps)
        field.change_scales(1)
        values = np.asarray(field["g"]).reshape(-1).copy()
        points = np.asarray(x).reshape(-1)
        exact = np.exp(-0.05) * np.sin(points) + 0.25 * np.exp(-0.2) * np.cos(
            2 * points
        )
        levels.append(
            {
                "resolution": steps,
                "scale": 0.5 / steps,
                "error_l2": float(np.sqrt(np.mean((values - exact) ** 2))),
                "iterations": int(solver.iteration),
                "final_time": float(solver.sim_time),
                "mean": float(np.mean(values)),
                "energy_ratio": float(np.mean(values**2) / np.mean(initial**2)),
            }
        )
        arrays = {"x": points, "y": np.zeros_like(points), "u": values, "exact": exact}
    return (
        levels,
        arrays,
        _origins("dedalus", {"dedalus": dedalus, "numpy": np, "mpi4py": mpi4py}),
    )


def solve(backend: str, output: Path, phase: str, expected_prefix: Path) -> None:
    """Run the actual optional backend; absence/error never becomes a skip."""
    if output.exists() or output.is_symlink():
        raise ValueError("output already exists; refusing to overwrite evidence")
    if Path(sys.prefix).resolve() != expected_prefix.resolve():
        raise ValueError("interpreter prefix differs from the requested native profile")
    levels, arrays, origins = _poisson() if backend == "fenicsx" else _heat()
    report = {
        "schema": 1,
        "backend": backend,
        "phase": phase,
        "prefix": str(Path(sys.prefix).resolve()),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime": {backend: VERSIONS[backend]},
        "origins": origins,
        "levels": levels,
    }
    validate_report(report)
    import numpy as np

    if any(np.iscomplexobj(a) or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("native solve produced a complex or nonfinite array")
    output.mkdir(parents=True, exist_ok=False)
    data = output / "values.npz"
    np.savez_compressed(data, **arrays)
    report["data_sha256"] = hashlib.sha256(data.read_bytes()).hexdigest()
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "backend": backend,
                "phase": phase,
                "executed_levels": 3,
                "native_skips": 0,
                "finest_error_l2": levels[-1]["error_l2"],
            }
        )
    )


def read_report(path: Path) -> dict[str, Any]:
    """Validate measured arrays as well as report metadata before acceptance."""
    report = json.loads(path.read_text())
    validate_report(report)
    data = path.parent / "values.npz"
    if hashlib.sha256(data.read_bytes()).hexdigest() != report["data_sha256"]:
        raise ValueError("measured NPZ digest differs from the native report")
    import numpy as np

    with np.load(data, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in ("x", "y", "u", "exact")}
    if any(
        a.ndim != 1 or not a.size or np.iscomplexobj(a) or not np.isfinite(a).all()
        for a in arrays.values()
    ):
        raise ValueError("measured arrays must be nonempty finite real vectors")
    if len({a.shape for a in arrays.values()}) != 1:
        raise ValueError("measured array shapes disagree")
    x, y = arrays["x"], arrays["y"]
    if report["backend"] == "fenicsx":
        grid = np.linspace(0, 1, 65)
        if (
            x.size != 65**2
            or np.unique(np.column_stack((x, y)), axis=0).shape[0] != 65**2
            or np.unique(x).size != 65
            or np.unique(y).size != 65
            or not np.allclose(np.unique(x), grid, rtol=0, atol=1e-12)
            or not np.allclose(np.unique(y), grid, rtol=0, atol=1e-12)
        ):
            raise ValueError("measured Poisson grid is incomplete or changed")
    elif (
        x.size != 64
        or not np.allclose(np.sort(x), np.arange(64) * math.pi / 32, rtol=0, atol=1e-12)
        or not np.all(y == 0)
    ):
        raise ValueError("measured periodic heat grid is incomplete or changed")
    exact = (
        np.sin(math.pi * x) * np.sin(math.pi * y)
        if report["backend"] == "fenicsx"
        else np.exp(-0.05) * np.sin(x) + 0.25 * np.exp(-0.2) * np.cos(2 * x)
    )
    if not np.allclose(arrays["exact"], exact, rtol=0, atol=1e-12):
        raise ValueError(
            "stored exact values disagree with the independent analytical formula"
        )
    if (
        float(np.sqrt(np.mean((arrays["u"] - exact) ** 2)))
        > ERROR_LIMITS[report["backend"]]
    ):
        raise ValueError("measured finest values violate the analytical ceiling")
    return {**report, "measurements": arrays}


def main() -> int:
    """Expose execution, artifact validation and relocated replay comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("solve")
    run.add_argument("--backend", choices=tuple(VERSIONS), required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--phase", choices=("original", "relocated"), required=True)
    run.add_argument("--expected-prefix", type=Path, required=True)
    check = commands.add_parser("validate")
    check.add_argument("report", type=Path)
    compare = commands.add_parser("compare")
    compare.add_argument("original", type=Path)
    compare.add_argument("relocated", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "solve":
            solve(args.backend, args.output, args.phase, args.expected_prefix)
        elif args.command == "validate":
            read_report(args.report)
        else:
            validate_pair(read_report(args.original), read_report(args.relocated))
    except (ValueError, KeyError, TypeError, ImportError, OSError) as error:
        print(f"native validation refused: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
