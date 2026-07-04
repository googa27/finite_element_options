"""Spatial solver assembling finite element operators."""

from __future__ import annotations

import numpy as np
import skfem as fem

from .forms import Forms, PDEForms
from .boundary import apply_dirichlet
from .adaptive import AdaptiveDiagnostics, AdaptiveMesh, AdaptiveResult
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
        self.last_adaptive_diagnostics: AdaptiveDiagnostics | None = None
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
        self._operator_matrix_cache: dict[float, object] = {}
        self.matrix_time_calls: list[tuple[float, float]] = []
        self.last_coefficient_diagnostics: dict[str, str] = {}
        self.stiffness = self.operator_matrix(0.0)

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

    def operator_matrix(self, th: float = 0.0):
        """Assemble or reuse the spatial PDE operator at time ``th``.

        ``th`` is supplied in the solver time coordinate and transformed back
        before coefficient fields are evaluated.  The resulting matrix includes
        diffusion, advection and the quadrature-evaluated reaction/discount
        field.
        """

        time_key = float(np.asarray(th, dtype=float))
        if not np.isfinite(time_key):
            raise ValueError("operator assembly time must be finite")
        if time_key not in self._operator_matrix_cache:
            th_phys = float(np.asarray(self.transform.untransform_time(time_key)))
            matrix = self.forms.operator_form(th_phys).assemble(self.Vh, th=th_phys)
            self._operator_matrix_cache[time_key] = matrix
            diagnostics = getattr(self.forms, "coefficient_diagnostics", {})
            self.last_coefficient_diagnostics = dict(diagnostics)
        return self._operator_matrix_cache[time_key]

    def matrices(
        self,
        theta: float,
        dt: float,
        *,
        start: float | None = None,
        end: float | None = None,
    ):
        """Return the endpoint-refreshed system matrices for the θ-scheme."""

        start_time = 0.0 if start is None else float(start)
        end_time = start_time if end is None else float(end)
        self.matrix_time_calls.append((start_time, end_time))
        start_operator = self.operator_matrix(start_time)
        end_operator = self.operator_matrix(end_time)
        A = self.mass - theta * dt * end_operator
        B = self.mass + (1 - theta) * dt * start_operator
        return A, B

    def boundary_term(self, th: float) -> np.ndarray:
        """Assemble natural-boundary and source contributions at time ``th``."""
        th_phys = float(np.asarray(self.transform.untransform_time(th)))
        natural = self.forms.b_lin().assemble(self.dVh, th=th_phys)
        source = self.forms.source_lin(th_phys).assemble(self.Vh, th=th_phys)
        diagnostics = getattr(self.forms, "coefficient_diagnostics", {})
        self.last_coefficient_diagnostics = dict(diagnostics)
        return np.asarray(natural, dtype=float) + np.asarray(source, dtype=float)

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

    def apply_dirichlet(self, A, b, boundaries, u_dirichlet):
        """Apply Dirichlet boundary conditions to ``A`` and ``b``."""
        return apply_dirichlet(A, b, self.Vh, boundaries, u_dirichlet)

    def refine_with_transfer(self, u: np.ndarray) -> AdaptiveResult:
        """Refine the internal mesh and transfer ``u`` to the new basis."""

        if self.adapt is None:
            raise ValueError("Adaptive mesh not configured for this solver.")
        result = self.adapt.refine_with_transfer(self.mesh, u)
        self.mesh = result.mesh
        self.Vh = fem.CellBasis(self.mesh, self.config.elem)
        self.dVh = fem.FacetBasis(self.mesh, self.config.elem)
        self.mass = self.forms.id_bil().assemble(self.Vh)
        self._operator_matrix_cache.clear()
        self.stiffness = self.operator_matrix(0.0)
        self.last_adaptive_diagnostics = result.diagnostics
        return result

    def refine_mesh(self, u: np.ndarray) -> fem.Mesh:
        """Refine the internal mesh using adaptive criterion.

        Parameters
        ----------
        u:
            Current solution vector used to compute refinement indicators.
        """
        return self.refine_with_transfer(u).mesh
