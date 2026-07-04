import os
import sys

import numpy as np
import pytest
import skfem as fem
import skfem.helpers as fh

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from finite_element_options.space.adaptive import AdaptiveMesh, mesh_measure

ELEM = fem.ElementTriP1()
BOUNDARIES = {
    "x_min": lambda x: np.isclose(x[0], 0.0),
    "x_max": lambda x: np.isclose(x[0], 1.0),
    "y_min": lambda x: np.isclose(x[1], 0.0),
    "y_max": lambda x: np.isclose(x[1], 1.0),
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


def test_residual_refinement_preserves_domain_and_reports_transfer() -> None:
    mesh = initial_mesh()
    adapt = AdaptiveMesh(ELEM, criterion="residual", boundaries=BOUNDARIES)
    u = solve(mesh)

    result = adapt.refine_with_transfer(mesh, u)

    assert result.mesh.nelements > mesh.nelements
    assert set(result.mesh.boundaries) == set(BOUNDARIES)
    assert mesh_measure(result.mesh) == pytest.approx(mesh_measure(mesh))
    assert np.all(result.diagnostics.element_measures > 0.0)
    assert result.diagnostics.old_elements == mesh.nelements
    assert result.diagnostics.new_elements == result.mesh.nelements
    assert result.diagnostics.marked_elements >= 1
    assert result.diagnostics.transfer_l2_change >= 0.0
    assert result.values.shape == (fem.CellBasis(result.mesh, ELEM).N,)


def test_coarsening_is_disabled_until_hierarchy_and_transfer_are_proven() -> None:
    mesh = initial_mesh()
    adapt = AdaptiveMesh(ELEM, criterion="residual", boundaries=BOUNDARIES)
    u = solve(mesh)
    refined = adapt.refine(mesh, u)
    u_ref = solve(refined)

    with pytest.raises(NotImplementedError, match="coarsening is disabled"):
        adapt.coarsen(refined, u_ref)


def test_gradient_refinement_increases_elements_and_preserves_measure() -> None:
    mesh = initial_mesh()
    adapt = AdaptiveMesh(ELEM, criterion="gradient", boundaries=BOUNDARIES)
    u = solve(mesh)
    refined = adapt.refine(mesh, u)
    assert refined.nelements > mesh.nelements
    assert mesh_measure(refined) == pytest.approx(mesh_measure(mesh))


def test_solution_transfer_preserves_constant_and_linear_1d_functions() -> None:
    element = fem.ElementLineP1()
    mesh = fem.MeshLine(np.linspace(0.0, 1.0, 5)).with_boundaries(
        {
            "left": lambda x: np.isclose(x[0], 0.0),
            "right": lambda x: np.isclose(x[0], 1.0),
        }
    )
    adapt = AdaptiveMesh(element, criterion="gradient")
    refined = mesh.refined(np.asarray([1, 2], dtype=np.int32))

    constant = np.full(fem.CellBasis(mesh, element).N, 7.0)
    transferred_constant = adapt.transfer_solution(mesh, refined, constant)
    assert np.allclose(transferred_constant, 7.0)

    linear = fem.CellBasis(mesh, element).doflocs[0]
    transferred_linear = adapt.transfer_solution(mesh, refined, linear)
    assert np.allclose(transferred_linear, fem.CellBasis(refined, element).doflocs[0])


def test_residual_estimator_supports_1d_meshes() -> None:
    element = fem.ElementLineP1()
    mesh = fem.MeshLine(np.linspace(0.0, 1.0, 5))
    basis = fem.CellBasis(mesh, element)
    u = np.sin(np.pi * basis.doflocs[0])
    adapt = AdaptiveMesh(element, criterion="residual", theta=0.5)

    result = adapt.refine_with_transfer(mesh, u)

    assert result.mesh.nelements > mesh.nelements
    assert mesh_measure(result.mesh) == pytest.approx(mesh_measure(mesh))
    assert result.values.shape == (fem.CellBasis(result.mesh, element).N,)
