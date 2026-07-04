"""Spatial solver assembling finite element operators."""

from __future__ import annotations

import numpy as np
import skfem as fem

from .forms import Forms, PDEForms
from .boundary import apply_dirichlet
from .adaptive import AdaptiveMesh
from finite_element_options.space.domain import DomainSpec
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
        if forms is None:
            self.transform.validate_transformed_state_domain(mesh.p)
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

    def domain_diagnostics(
        self, *, horizon: float, tail_mass: float = 1.0e-6
    ) -> dict[str, object]:
        """Return public domain, boundary, and model-tail diagnostics."""

        domain = getattr(self.mesh, "domain_spec", None)
        if isinstance(domain, DomainSpec):
            state_domain = [axis.to_public_dict() for axis in domain.axes]
        else:
            state_domain = [
                {
                    "name": f"x{axis}",
                    "lower": float(np.min(self.mesh.p[axis])),
                    "upper": float(np.max(self.mesh.p[axis])),
                    "scale": "unknown",
                    "truncation_policy": "mesh-extents",
                    "tail_mass": None,
                }
                for axis in range(int(self.mesh.dim()))
            ]
        diagnostics: dict[str, object] = {
            "horizon": float(horizon),
            "coordinate_system": getattr(domain, "coordinate_system", "unknown"),
            "state_domain": state_domain,
            "boundary_facets": tuple((getattr(self.mesh, "boundaries", None) or {})),
            "mesh_dimension": int(self.mesh.dim()),
            "mesh_elements": int(self.mesh.nelements),
        }
        diagnostics["variance_domain"] = self.variance_domain_diagnostics(
            horizon=horizon,
            tail_mass=tail_mass,
        )
        return diagnostics

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
        th_phys = float(np.asarray(self.transform.untransform_time(th)))
        return self.forms.b_lin().assemble(self.dVh, th=th_phys)

    def dirichlet(self, th: float) -> np.ndarray:
        """Return nodal Dirichlet values at time ``th``.

        Dirichlet elimination consumes one value per degree of freedom, so this
        evaluates the boundary oracle at ``Vh.doflocs`` instead of using an
        ``L2`` projection that can smear endpoint values.
        """

        th_phys = float(np.asarray(self.transform.untransform_time(th)))
        return np.asarray(
            self._projected_payoff(self.Vh.doflocs, th_phys=th_phys),
            dtype=float,
        )

    def variance_domain_diagnostics(
        self, *, horizon: float, tail_mass: float = 1.0e-6
    ) -> dict:
        """Return model/domain diagnostics for the variance coordinate.

        Heston-like dynamics expose exact CIR moment diagnostics.  The spatial
        solver enriches those model diagnostics with the current mesh variance
        extent so downstream evidence can tell whether the numerical domain is
        a model-driven tail bound or only an ad-hoc mesh size.
        """

        if not hasattr(self.dynamics, "variance_domain_diagnostics"):
            return {
                "policy": "no-stochastic-variance-coordinate",
                "horizon": float(horizon),
                "mesh_dimension": int(self.mesh.dim()),
            }

        state = self.transform.untransform_state(self.mesh.p)
        variance_seed = state[1] if state.shape[0] > 1 else np.zeros_like(state[0])
        diagnostics = dict(
            self.dynamics.variance_domain_diagnostics(
                horizon=horizon,
                initial_variance=variance_seed,
                tail_mass=tail_mass,
            )
        )
        if state.shape[0] > 1:
            diagnostics["mesh_variance_min"] = float(np.min(variance_seed))
            diagnostics["mesh_variance_max"] = float(np.max(variance_seed))
            diagnostics["mesh_contains_tail_bound"] = bool(
                diagnostics["mesh_variance_min"] <= diagnostics["domain_lower"]
                and diagnostics["domain_upper"] <= diagnostics["mesh_variance_max"]
            )
        diagnostics["mesh_dimension"] = int(self.mesh.dim())
        diagnostics["mesh_elements"] = int(self.mesh.nelements)
        return diagnostics

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
