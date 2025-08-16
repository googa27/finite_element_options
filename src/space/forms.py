"""Finite element forms for the option pricing PDE."""

import skfem as fem
import skfem.helpers as fhl

from src.transform import CoordinateTransform


class Forms:
    """Collection of variational forms used in the solver."""

    def __init__(
        self,
        is_call,
        bsopt,
        dynh,
        transform: CoordinateTransform | None = None,
    ):
        self.is_call = is_call
        self.bsopt = bsopt
        self.dynh = dynh
        self.transform = transform or CoordinateTransform()

    @staticmethod
    def id_bil():
        @fem.BilinearForm
        def Id_bil(u, v, _):
            return u * v

        return Id_bil

    def l_bil(self):
        @fem.BilinearForm
        def L_bil(u, v, w):
            coords = self.transform.untransform_state(w.x)
            A = self.dynh.A(*coords)
            dA = self.dynh.dA(*coords)
            b = self.dynh.b(*coords)
            mu = [b_i - dA_i / 2 for b_i, dA_i in zip(b, dA)]
            return (
                -(1 / 2) * fhl.dot(fhl.grad(v), fhl.mul(A, fhl.grad(u)))
                + v * fhl.dot(mu, fhl.grad(u))
                - self.dynh.r * v * u
            )

        return L_bil

    def b_lin(self):
        if not hasattr(self.dynh, "boundary_term"):
            @fem.LinearForm
            def zero(v, w):
                return 0.0

            return zero

        return self.dynh.boundary_term(self.is_call, self.bsopt)
