"""Example demonstrating adaptive mesh refinement."""

import numpy as np
import skfem as fem
import skfem.helpers as fh

import src.plots as splt
from src.space.adaptive import AdaptiveMesh

# Finite element and domain size
ELEM = fem.ElementTriP1()
X_MAX = 1
Y_MAX = 1


def f(x, y):
    """Right-hand side forcing term."""
    return np.exp((x - 0.3) ** 2 + (y - 0.2) ** 2)


@fem.LinearForm
def b_lin(v, w):
    x, y = w.x
    return f(x, y) * v


@fem.BilinearForm
def a_bil(u, v, _):
    return fh.dot(u.grad, v.grad) + 0.1 * u * v


def solve_system(mesh: fem.Mesh) -> np.ndarray:
    """Solve the model problem on ``mesh``."""
    Vh = fem.CellBasis(mesh, ELEM)
    A = a_bil.assemble(Vh)
    b = b_lin.assemble(Vh)
    A, b = fem.enforce(A, b, D=Vh.get_dofs(["y_min", "y_max"]))
    return fem.solve(A, b)


BOUNDARIES = {
    "x_min": lambda x: x[0] == 0,
    "x_max": lambda x: x[0] == X_MAX,
    "y_min": lambda x: x[1] == 0,
    "y_max": lambda x: x[1] == Y_MAX,
}

# Initialise mesh and adaptive handler
mesh = (
    fem.MeshTri()
    .init_tensor(x=np.linspace(0, X_MAX, 3), y=np.linspace(0, Y_MAX, 3))
    .with_boundaries(BOUNDARIES)
)
adapt = AdaptiveMesh(ELEM, criterion="residual", boundaries=BOUNDARIES)

# Initial solve and refinement loop
u = solve_system(mesh)
for _ in range(5):
    mesh = adapt.refine(mesh, u)
    u = solve_system(mesh)

# Visualise final solution
Vh = fem.CellBasis(mesh, ELEM)
splt.plot_2d(Vh, u, title="Solution")
