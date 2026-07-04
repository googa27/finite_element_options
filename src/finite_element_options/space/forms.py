"""Finite element forms expressed as injectable strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import skfem as fem
import skfem.helpers as fhl

from finite_element_options.core.interfaces import DynamicsModel, Payoff
from finite_element_options.transform import CoordinateTransform


class Forms(ABC):
    """Interface defining the variational forms used by the solver."""

    @abstractmethod
    def id_bil(self):
        r"""Return the mass bilinear form ``\int u v\,dx``."""

    @abstractmethod
    def l_bil(self, th: float = 0.0):
        """Return the bilinear form representing the PDE operator."""

    def operator_form(self, th: float = 0.0):
        """Return the PDE operator form at physical time ``th``."""

        return self.l_bil(th=th)

    @abstractmethod
    def b_lin(self):
        """Return the linear form associated with natural boundaries."""

    def source_lin(self, th: float = 0.0):
        """Return the cell source/load form at physical time ``th``."""

        del th

        @fem.LinearForm
        def zero(_v, _w):  # pylint: disable=unused-argument
            return 0.0

        return zero


class PDEForms(Forms):
    """Default forms constructed from a dynamics model and payoff."""

    def __init__(
        self,
        *,
        is_call: bool,
        payoff: Payoff,
        dynamics: DynamicsModel,
        transform: CoordinateTransform | None = None,
    ):
        """Store option type, payoff and dynamics descriptors.

        Parameters
        ----------
        is_call:
            ``True`` for call options and ``False`` for puts.
        payoff:
            Payoff function describing the terminal condition.
        dynamics:
            Model supplying drift, diffusion and coefficient fields.
        transform:
            Optional coordinate transform applied before evaluation.
        """

        self.is_call = is_call
        self.payoff = payoff
        self.dynamics = dynamics
        self.transform = transform or CoordinateTransform()
        self.coefficient_diagnostics: dict[str, str] = {
            "discount_field": "constant-attribute",
            "source_field": "zero",
        }

    @staticmethod
    def id_bil():
        r"""Return the mass bilinear form ``\int u v\,dx``."""

        @fem.BilinearForm
        def _id(u, v, _):
            return u * v

        return _id

    def l_bil(self, th: float = 0.0):
        """Return the bilinear form representing the PDE operator."""

        physical_time = _validate_time(th)

        @fem.BilinearForm
        def _l(u, v, w):
            state = self.transform.untransform_state(w.x)
            time = _form_time(w, physical_time)
            A, dA, b = self.transform.transformed_coefficients(self.dynamics, w.x)
            mu = [b_i - dA_i / 2 for b_i, dA_i in zip(b, dA)]
            discount = self._scalar_field(
                name="discount",
                state=state,
                time=time,
                default=getattr(self.dynamics, "r", 0.0),
            )
            return (
                -(1 / 2) * fhl.dot(fhl.grad(v), fhl.mul(A, fhl.grad(u)))
                + v * fhl.dot(mu, fhl.grad(u))
                - discount * v * u
            )

        return _l

    def source_lin(self, th: float = 0.0):
        """Return the cell source/load form at physical time ``th``."""

        physical_time = _validate_time(th)

        @fem.LinearForm
        def _source(v, w):
            state = self.transform.untransform_state(w.x)
            time = _form_time(w, physical_time)
            source = self._scalar_field(
                name="source",
                state=state,
                time=time,
                default=0.0,
            )
            return source * v

        return _source

    def b_lin(self):
        """Return the linear form associated with natural boundaries."""

        if not hasattr(self.dynamics, "boundary_term"):

            @fem.LinearForm
            def zero(v, w):  # pylint: disable=unused-argument
                return 0.0

            return zero

        return self.dynamics.boundary_term(self.is_call, self.payoff)

    def _scalar_field(self, *, name: str, state: np.ndarray, time: float, default: float):
        """Evaluate a finite scalar field on quadrature state/time arrays."""

        field = getattr(self.dynamics, name, None)
        diagnostic_key = f"{name}_field"
        if callable(field):
            value = field(state, time)
            self.coefficient_diagnostics[diagnostic_key] = "callable"
        else:
            value = default
            self.coefficient_diagnostics[diagnostic_key] = (
                "constant-attribute" if name == "discount" else "zero"
            )

        array = np.asarray(value, dtype=float)
        target_shape = np.asarray(state[0], dtype=float).shape
        if array.ndim == 0:
            array = np.full(target_shape, float(array), dtype=float)
        else:
            try:
                array = np.broadcast_to(array, target_shape).astype(float, copy=False)
            except ValueError as exc:
                msg = (
                    f"{name} field returned shape {array.shape}, expected scalar or "
                    f"broadcastable to quadrature shape {target_shape}"
                )
                raise ValueError(msg) from exc
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} field must return finite values")
        return array


def _validate_time(th: float) -> float:
    """Return a finite scalar time value."""

    time = float(np.asarray(th, dtype=float))
    if not np.isfinite(time):
        raise ValueError("coefficient time must be finite")
    return time


def _form_time(w, default: float) -> float:
    """Return the scalar time attached to a scikit-fem form assembly."""

    raw = getattr(w, "th", default)
    time = float(np.asarray(raw, dtype=float))
    if not np.isfinite(time):
        raise ValueError("form coefficient time must be finite")
    return time
