"""Adaptive mesh refinement utilities."""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import skfem as fem
import skfem.helpers as fh


class AdaptiveMesh:
    """Adaptive mesh refinement based on element-wise indicators.

    Parameters
    ----------
    element:
        Finite element used for basis functions.
    criterion:
        Refinement criterion to use. Supported values are ``"residual"``
        (edge jump residual estimator) and ``"gradient"`` (gradient magnitude
        indicator).
    theta:
        Parameter passed to :func:`skfem.adaptive_theta` controlling how many
        elements are refined. Values closer to ``1`` refine fewer elements.
    boundaries:
        Optional boundary definition dictionary passed to
        :meth:`skfem.Mesh.with_boundaries` after each refinement or
        coarsening step.
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
        """Return residual-based error indicator."""
        fbasis = [
            fem.InteriorFacetBasis(mesh, self.element, side=i) for i in [0, 1]
        ]
        w = {f"u{i+1}": fbasis[i].interpolate(u) for i in [0, 1]}

        @fem.Functional
        def edge_jump(w):
            h = w.h
            n = w.n
            dw1 = fh.grad(w["u1"])
            dw2 = fh.grad(w["u2"])
            jump = (dw1[0] - dw2[0]) * n[0] + (
                dw1[1] - dw2[1]
            ) * n[1]
            return h * jump**2

        eta_E = edge_jump.elemental(fbasis[0], **w)
        tmp = np.zeros(mesh.facets.shape[1])
        np.add.at(tmp, fbasis[0].find, eta_E)
        return np.sum(0.5 * tmp[mesh.t2f], axis=0)

    def _gradient_estimator(self, mesh: fem.Mesh, u: np.ndarray) -> np.ndarray:
        """Return gradient-based error indicator."""
        basis = fem.CellBasis(mesh, self.element)
        w = basis.interpolate(u)

        @fem.Functional
        def grad_energy(w):
            return fh.dot(w["u"].grad, w["u"].grad)

        return grad_energy.elemental(basis, u=w)

    def _estimate(self, mesh: fem.Mesh, u: np.ndarray) -> np.ndarray:
        if self.criterion == "gradient":
            return self._gradient_estimator(mesh, u)
        return self._residual_estimator(mesh, u)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def refine(self, mesh: fem.Mesh, u: np.ndarray) -> fem.Mesh:
        """Refine ``mesh`` based on solution ``u`` and configured criterion."""
        eta = self._estimate(mesh, u)
        mesh = mesh.refined(
            fem.adaptive_theta(eta, theta=self.theta)
        ).smoothed()
        if self.boundaries:
            mesh = mesh.with_boundaries(self.boundaries)
        return mesh

    def coarsen(self, mesh: fem.Mesh, u: np.ndarray) -> fem.Mesh:
        """Coarsen ``mesh`` by removing elements with the smallest error."""
        eta = self._estimate(mesh, u)
        remove = np.argsort(eta)[: len(eta) // 2]
        mesh = mesh.remove_elements(remove)
        if self.boundaries:
            mesh = mesh.with_boundaries(self.boundaries)
        return mesh
