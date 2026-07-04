"""Adaptive mesh refinement utilities with topology and transfer guards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
import skfem as fem
import skfem.helpers as fh


@dataclass(frozen=True)
class AdaptiveDiagnostics:
    """Diagnostics emitted by one adaptive mesh operation."""

    criterion: str
    marking_theta: float
    marked_elements: int
    old_elements: int
    new_elements: int
    old_measure: float
    new_measure: float
    element_measures: np.ndarray
    estimator: np.ndarray
    transfer_operator: str
    transfer_l2_change: float


@dataclass(frozen=True)
class AdaptiveResult:
    """Mesh, transferred values, and diagnostics after adaptation."""

    mesh: fem.Mesh
    values: np.ndarray
    diagnostics: AdaptiveDiagnostics


def element_measures(mesh: fem.Mesh) -> np.ndarray:
    """Return positive simplex measure for every mesh element."""

    p = np.asarray(mesh.p, dtype=float)
    t = np.asarray(mesh.t, dtype=int)
    dim = int(mesh.dim())
    if dim == 1:
        return np.abs(p[0, t[1]] - p[0, t[0]])
    if dim == 2:
        a = p[:, t[1]] - p[:, t[0]]
        b = p[:, t[2]] - p[:, t[0]]
        return 0.5 * np.abs(a[0] * b[1] - a[1] * b[0])
    if dim == 3:
        a = p[:, t[1]] - p[:, t[0]]
        b = p[:, t[2]] - p[:, t[0]]
        c = p[:, t[3]] - p[:, t[0]]
        cross = np.cross(a.T, b.T)
        return np.abs(np.einsum("ij,ij->i", cross, c.T)) / 6.0
    msg = f"adaptive topology checks support simplex meshes in 1D/2D/3D, got {dim}D"
    raise NotImplementedError(msg)


def mesh_measure(mesh: fem.Mesh) -> float:
    """Return the total geometric measure covered by ``mesh``."""

    return float(np.sum(element_measures(mesh)))


class AdaptiveMesh:
    """Adaptive mesh refinement based on element-wise indicators.

    Parameters
    ----------
    element:
        Finite element used for basis functions.
    criterion:
        Refinement criterion to use. Supported values are ``"residual"``
        (facet/edge jump residual estimator) and ``"gradient"`` (gradient
        magnitude indicator).
    theta:
        Parameter passed to :func:`skfem.adaptive_theta` controlling how many
        elements are refined. Values closer to ``1`` refine fewer elements.
    boundaries:
        Optional boundary definition dictionary passed to
        :meth:`skfem.Mesh.with_boundaries` after each refinement step.
    """

    def __init__(
        self,
        element: fem.Element,
        *,
        criterion: str = "residual",
        theta: float = 0.5,
        boundaries: (
            Dict[str, Callable[[np.ndarray], np.ndarray]] | None
        ) = None,
    ) -> None:
        """Store mesh refinement configuration."""

        self.element = element
        self.criterion = criterion
        self.theta = theta
        self.boundaries = boundaries or {}

    # ------------------------------------------------------------------
    # Estimators
    # ------------------------------------------------------------------
    def _residual_estimator(self, mesh: fem.Mesh, u: np.ndarray) -> np.ndarray:
        """Return residual-based error indicators for every element."""

        dim = int(mesh.dim())
        if dim == 1:
            return self._line_residual_estimator(mesh, u)
        return self._facet_jump_residual_estimator(mesh, u)

    def _line_residual_estimator(self, mesh: fem.Mesh, u: np.ndarray) -> np.ndarray:
        """Return 1D slope-jump residual indicators."""

        p = np.asarray(mesh.p, dtype=float)
        t = np.asarray(mesh.t, dtype=int)
        values = np.asarray(u, dtype=float)
        lengths = element_measures(mesh)
        dx = p[0, t[1]] - p[0, t[0]]
        slopes = (values[t[1]] - values[t[0]]) / dx
        eta = np.zeros(mesh.nelements, dtype=float)
        point_to_elements: dict[int, list[int]] = {}
        for elem, nodes in enumerate(t.T):
            for node in nodes:
                point_to_elements.setdefault(int(node), []).append(elem)
        for elems in point_to_elements.values():
            if len(elems) < 2:
                continue
            local_slopes = slopes[elems]
            jump = float(np.max(local_slopes) - np.min(local_slopes))
            scale = float(np.mean(lengths[elems]))
            eta[elems] += 0.5 * scale * jump**2
        return eta

    def _facet_jump_residual_estimator(
        self, mesh: fem.Mesh, u: np.ndarray
    ) -> np.ndarray:
        """Return normal-gradient jump indicators on interior facets."""

        dim = int(mesh.dim())
        fbasis = [
            fem.InteriorFacetBasis(mesh, self.element, side=i) for i in [0, 1]
        ]
        w = {f"u{i + 1}": fbasis[i].interpolate(u) for i in [0, 1]}

        @fem.Functional
        def edge_jump(w):
            h = w.h
            n = w.n
            dw1 = fh.grad(w["u1"])
            dw2 = fh.grad(w["u2"])
            jump = sum((dw1[axis] - dw2[axis]) * n[axis] for axis in range(dim))
            return h * jump**2

        eta_e = edge_jump.elemental(fbasis[0], **w)
        facet_eta = np.zeros(mesh.facets.shape[1])
        np.add.at(facet_eta, fbasis[0].find, eta_e)
        return np.sum(0.5 * facet_eta[mesh.t2f], axis=0)

    def _gradient_estimator(self, mesh: fem.Mesh, u: np.ndarray) -> np.ndarray:
        """Return gradient-energy indicators for every element."""

        basis = fem.CellBasis(mesh, self.element)
        w = basis.interpolate(u)

        @fem.Functional
        def grad_energy(w):
            return fh.dot(w["u"].grad, w["u"].grad)

        return np.asarray(grad_energy.elemental(basis, u=w), dtype=float)

    def _estimate(self, mesh: fem.Mesh, u: np.ndarray) -> np.ndarray:
        """Return finite non-negative element indicators."""

        if self.criterion == "gradient":
            eta = self._gradient_estimator(mesh, u)
        elif self.criterion == "residual":
            eta = self._residual_estimator(mesh, u)
        else:
            msg = f"unsupported adaptive criterion: {self.criterion!r}"
            raise ValueError(msg)
        eta = np.asarray(eta, dtype=float)
        if eta.shape != (mesh.nelements,):
            msg = (
                "adaptive estimator must return one indicator per element; "
                f"got shape {eta.shape} for {mesh.nelements} elements"
            )
            raise ValueError(msg)
        if not np.all(np.isfinite(eta)):
            msg = "adaptive estimator returned non-finite indicators"
            raise ValueError(msg)
        return np.maximum(eta, 0.0)

    def _marked_elements(self, eta: np.ndarray) -> np.ndarray:
        """Return a non-empty set of element indices selected for refinement."""

        marked = np.asarray(fem.adaptive_theta(eta, theta=self.theta), dtype=np.int32)
        if marked.size == 0:
            marked = np.asarray([int(np.argmax(eta))], dtype=np.int32)
        return np.unique(marked)

    # ------------------------------------------------------------------
    # Topology and transfer
    # ------------------------------------------------------------------
    def _validate_topology(
        self, mesh: fem.Mesh, *, reference_measure: float | None = None
    ) -> np.ndarray:
        """Validate element orientation, measure, and domain coverage."""

        measures = element_measures(mesh)
        if measures.shape != (mesh.nelements,):
            msg = "mesh measure calculation did not produce one value per element"
            raise ValueError(msg)
        if not np.all(np.isfinite(measures)) or not np.all(measures > 0.0):
            msg = "adaptive mesh contains inverted, degenerate, or non-finite elements"
            raise ValueError(msg)
        if reference_measure is not None and not np.isclose(
            float(np.sum(measures)), reference_measure, rtol=1.0e-10, atol=1.0e-12
        ):
            msg = "adaptive refinement changed total domain measure"
            raise ValueError(msg)
        return measures

    def transfer_solution(
        self, old_mesh: fem.Mesh, new_mesh: fem.Mesh, values: np.ndarray
    ) -> np.ndarray:
        """Interpolate nodal solution values from ``old_mesh`` to ``new_mesh``."""

        old_basis = fem.CellBasis(old_mesh, self.element)
        new_basis = fem.CellBasis(new_mesh, self.element)
        old_values = np.asarray(values, dtype=float)
        if old_values.shape != (old_basis.N,):
            msg = (
                "solution transfer expected one value per old basis dof; "
                f"got {old_values.shape} for {old_basis.N} dofs"
            )
            raise ValueError(msg)
        interpolator = old_basis.interpolator(old_values)
        transferred = np.asarray(interpolator(new_basis.doflocs), dtype=float)
        if transferred.shape != (new_basis.N,):
            msg = "solution transfer did not return one value per new basis dof"
            raise ValueError(msg)
        if not np.all(np.isfinite(transferred)):
            msg = "solution transfer produced non-finite values"
            raise ValueError(msg)
        return transferred

    def _transfer_l2_change(
        self, old_mesh: fem.Mesh, new_mesh: fem.Mesh, old_values: np.ndarray,
        new_values: np.ndarray,
    ) -> float:
        """Return round-trip RMS transfer change at old degrees of freedom."""

        old_basis = fem.CellBasis(old_mesh, self.element)
        new_basis = fem.CellBasis(new_mesh, self.element)
        roundtrip = new_basis.interpolator(new_values)(old_basis.doflocs)
        delta = np.asarray(roundtrip, dtype=float) - np.asarray(old_values, dtype=float)
        return float(np.linalg.norm(delta) / np.sqrt(delta.size))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def refine_with_transfer(self, mesh: fem.Mesh, u: np.ndarray) -> AdaptiveResult:
        """Refine ``mesh`` and transfer solution values to the new basis."""

        old_measure = mesh_measure(mesh)
        self._validate_topology(mesh)
        eta = self._estimate(mesh, u)
        marked = self._marked_elements(eta)
        refined = mesh.refined(marked)
        if int(refined.dim()) > 1:
            refined = refined.smoothed()
        if self.boundaries:
            refined = refined.with_boundaries(self.boundaries)
        measures = self._validate_topology(refined, reference_measure=old_measure)
        transferred = self.transfer_solution(mesh, refined, u)
        diagnostics = AdaptiveDiagnostics(
            criterion=self.criterion,
            marking_theta=float(self.theta),
            marked_elements=int(marked.size),
            old_elements=int(mesh.nelements),
            new_elements=int(refined.nelements),
            old_measure=old_measure,
            new_measure=float(np.sum(measures)),
            element_measures=measures,
            estimator=eta,
            transfer_operator="nodal_interpolation",
            transfer_l2_change=self._transfer_l2_change(
                mesh, refined, np.asarray(u, dtype=float), transferred
            ),
        )
        return AdaptiveResult(refined, transferred, diagnostics)

    def refine(self, mesh: fem.Mesh, u: np.ndarray) -> fem.Mesh:
        """Refine ``mesh`` based on solution ``u`` and configured criterion."""

        return self.refine_with_transfer(mesh, u).mesh

    def coarsen(self, mesh: fem.Mesh, u: np.ndarray) -> fem.Mesh:
        """Reject coarsening until reversible hierarchy guards are implemented."""

        _ = (mesh, u)
        msg = (
            "adaptive coarsening is disabled until a reversible hierarchy, "
            "coverage proof, and solution-transfer invariant are implemented"
        )
        raise NotImplementedError(msg)
