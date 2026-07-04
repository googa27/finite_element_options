"""Installed-package example demonstrating adaptive mesh refinement."""

from __future__ import annotations

import numpy as np
import skfem as fem
import skfem.helpers as fh

import finite_element_options.plots as plots
from finite_element_options.space.adaptive import AdaptiveMesh

_ELEMENT = fem.ElementTriP1()
_X_MAX = 1
_Y_MAX = 1
_BOUNDARIES = {
    "x_min": lambda x: x[0] == 0,
    "x_max": lambda x: x[0] == _X_MAX,
    "y_min": lambda x: x[1] == 0,
    "y_max": lambda x: x[1] == _Y_MAX,
}


def forcing(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the right-hand-side forcing term."""
    return np.exp((x - 0.3) ** 2 + (y - 0.2) ** 2)


@fem.LinearForm
def _linear_form(v, w):
    x, y = w.x
    return forcing(x, y) * v


@fem.BilinearForm
def _bilinear_form(u, v, _):
    return fh.dot(u.grad, v.grad) + 0.1 * u * v


def solve_system(mesh: fem.Mesh) -> np.ndarray:
    """Solve the model problem on ``mesh``."""
    basis = fem.CellBasis(mesh, _ELEMENT)
    matrix = _bilinear_form.assemble(basis)
    rhs = _linear_form.assemble(basis)
    matrix, rhs = fem.enforce(matrix, rhs, D=basis.get_dofs(["y_min", "y_max"]))
    return fem.solve(matrix, rhs)


def main() -> None:
    """Run five adaptive refinements and plot the final solution."""
    mesh = (
        fem.MeshTri()
        .init_tensor(x=np.linspace(0, _X_MAX, 3), y=np.linspace(0, _Y_MAX, 3))
        .with_boundaries(_BOUNDARIES)
    )
    adapter = AdaptiveMesh(_ELEMENT, criterion="residual", boundaries=_BOUNDARIES)

    solution = solve_system(mesh)
    for _ in range(5):
        mesh = adapter.refine(mesh, solution)
        solution = solve_system(mesh)

    basis = fem.CellBasis(mesh, _ELEMENT)
    plots.plot_2d(basis, solution, title="Solution")


if __name__ == "__main__":
    main()
