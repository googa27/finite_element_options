"""Spatial solver assembling finite element operators."""

from __future__ import annotations

import numpy as np
import skfem as fem

from .forms import Forms, PDEForms
from .boundary import apply_dirichlet
from .adaptive import AdaptiveMesh
from finite_element_options.transform import CoordinateTransform
from finite_element_options.core.config import Config


class SpaceSolver:
    """Assemble spatial operators and boundary terms for the PDE."""

    def __init__(
        self,
        mesh: fem.Mesh,
        dynamics,
        payoff,
        is_call: bool,
        transform: CoordinateTransform | None = None,
        *,
        forms: Forms | None = None,
        adaptive_criterion: str | None = None,
        config: Config | None = None,
    ):
        """Initialize the spatial solver and assemble static operators.

        Parameters
        ----------
        mesh:
            Spatial discretisation mesh.
        dynamics:
            Stochastic dynamics describing the underlying process.
        payoff:
            Option payoff object.
        is_call:
            ``True`` for call options, ``False`` for puts.
        transform:
            Optional coordinate transformation.
        forms:
            Preassembled bilinear and linear forms.
        adaptive_criterion:
            Name of adaptive refinement criterion.
        config:
            Numerical configuration specifying finite element type.
        """
        self.mesh = mesh
        self.dynamics = dynamics
        self.payoff = payoff
        self.is_call = is_call
        self.transform = transform or CoordinateTransform()
        self.config = config or Config()
        self.Vh = fem.CellBasis(mesh, self.config.elem)
        self.dVh = fem.FacetBasis(mesh, self.config.elem)
        self.adapt = (
            AdaptiveMesh(self.config.elem, criterion=adaptive_criterion)
            if adaptive_criterion is not None
            else None
        )
        self.forms = forms or PDEForms(
            is_call=is_call,
            payoff=payoff,
            dynamics=dynamics,
            transform=self.transform,
        )
        self._mean_variance_supports_config = (
            "config" in dynamics.mean_variance.__code__.co_varnames
        )
        self.mass = self.forms.id_bil().assemble(self.Vh)
        self.stiffness = self.forms.l_bil().assemble(self.Vh)

    def _projected_payoff(
        self, x: np.ndarray, *, th_phys: float | None = None
    ) -> np.ndarray:
        """Return projected payoff or boundary values for coordinates ``x``.

        The helper performs :meth:`~finite_element_options.transform.CoordinateTransform.untransform_state` once and reuses
        the extracted variance inputs when computing mean-variance terms.
        """

        state = self.transform.untransform_state(x)
        spot = state[0]
        variance_inputs = state[1] if state.shape[0] > 1 else None

        if self.is_call:
            payoff_fn = self.payoff.call_payoff
            price_fn = self.payoff.call
        else:
            payoff_fn = self.payoff.put_payoff
            price_fn = self.payoff.put

        if th_phys is None:
            return payoff_fn(spot)

        variance_seed = (
            variance_inputs if variance_inputs is not None else np.zeros_like(spot)
        )
        kwargs = {"config": self.config} if self._mean_variance_supports_config else {}
        mean_variance = self.dynamics.mean_variance(
            th_phys,
            variance_seed,
            **kwargs,
        )
        return price_fn(th_phys, spot, mean_variance)

    def initial_condition(self) -> np.ndarray:
        """Initial spatial values from the payoff."""
        return self.Vh.project(lambda x: self._projected_payoff(x))

    def matrices(self, theta: float, dt: float):
        """Return the system matrices for the θ-scheme."""
        A = self.mass - theta * dt * self.stiffness
        B = self.mass + (1 - theta) * dt * self.stiffness
        return A, B

    def boundary_term(self, th: float) -> np.ndarray:
        """Assemble the natural boundary contribution at time ``th``."""
        th_phys = self.transform.untransform_time(th)
        return self.forms.b_lin().assemble(self.dVh, th=th_phys)

    def dirichlet(self, th: float) -> np.ndarray:
        """Return Dirichlet values at time ``th``."""
        th_phys = self.transform.untransform_time(th)

        return self.Vh.project(lambda x: self._projected_payoff(x, th_phys=th_phys))

    def apply_dirichlet(self, A, b, dirichlet_bcs, u_dirichlet):
        """Apply Dirichlet boundary conditions to ``A`` and ``b``."""
        return apply_dirichlet(A, b, self.Vh, dirichlet_bcs, u_dirichlet)

    def refine_mesh(self, u: np.ndarray) -> fem.Mesh:
        """Refine the internal mesh using adaptive criterion.

        Parameters
        ----------
        u:
            Current solution vector used to compute refinement indicators.
        """
        if self.adapt is None:
            raise ValueError("Adaptive mesh not configured for this solver.")
        self.mesh = self.adapt.refine(self.mesh, u)
        self.Vh = fem.CellBasis(self.mesh, self.config.elem)
        self.dVh = fem.FacetBasis(self.mesh, self.config.elem)
        self.mass = self.forms.id_bil().assemble(self.Vh)
        self.stiffness = self.forms.l_bil().assemble(self.Vh)
        return self.mesh
