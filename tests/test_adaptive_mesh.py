import os
import sys
import numpy as np
import skfem as fem
import skfem.helpers as fh

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.space.adaptive import AdaptiveMesh

ELEM = fem.ElementTriP1()
BOUNDARIES = {
    "x_min": lambda x: x[0] == 0,
    "x_max": lambda x: x[0] == 1,
    "y_min": lambda x: x[1] == 0,
    "y_max": lambda x: x[1] == 1,
}


@fem.LinearForm
def b_lin(v, w):
    x, y = w.x
    return np.exp((x - 0.3) ** 2 + (y - 0.2) ** 2) * v


@fem.BilinearForm
def a_bil(u, v, _):
    return fh.dot(u.grad, v.grad) + 0.1 * u * v


def solve(mesh: fem.Mesh) -> np.ndarray:
    Vh = fem.CellBasis(mesh, ELEM)
    A = a_bil.assemble(Vh)
    b = b_lin.assemble(Vh)
    A, b = fem.enforce(A, b, D=Vh.get_dofs(["y_min", "y_max"]))
    return fem.solve(A, b)


def initial_mesh() -> fem.Mesh:
    return (
        fem.MeshTri()
        .init_tensor(x=np.linspace(0, 1, 3), y=np.linspace(0, 1, 3))
        .with_boundaries(BOUNDARIES)
    )


def test_residual_refine_and_coarsen():
    mesh = initial_mesh()
    adapt = AdaptiveMesh(ELEM, criterion="residual", boundaries=BOUNDARIES)
    u = solve(mesh)
    refined = adapt.refine(mesh, u)
    assert refined.nelements > mesh.nelements
    u_ref = solve(refined)
    coarse = adapt.coarsen(refined, u_ref)
    assert coarse.nelements < refined.nelements


def test_gradient_refinement_increases_elements():
    mesh = initial_mesh()
    adapt = AdaptiveMesh(ELEM, criterion="gradient", boundaries=BOUNDARIES)
    u = solve(mesh)
    refined = adapt.refine(mesh, u)
    assert refined.nelements > mesh.nelements
