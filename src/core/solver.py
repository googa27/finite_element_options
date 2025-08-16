"""Core PDE solver for option pricing."""

from typing import Sequence

import numpy as np
import skfem as fem

from .forms import Forms
import CONFIG as CFG


def initialize_value_array(t: Sequence[float], Vh: fem.CellBasis, bsopt, is_call: bool) -> np.ndarray:
    """Initialise the time-space-value array with the payoff."""
    v_tsv = np.empty((len(t), Vh.N))
    v_tsv[0] = Vh.project(
        lambda x: bsopt.call_payoff(x[0]) * is_call
        + bsopt.put_payoff(x[0]) * (not is_call)
    )
    return v_tsv


def solve_pde(
    t: Sequence[float],
    mesh: fem.MeshTri,
    dynh,
    bsopt,
    is_call: bool,
    dirichlet_bcs=None,
    lam: float = 0.5,
    is_american: bool = False,
):
    """Solve the option pricing PDE returning the time/space solution array."""
    dt = t[1] - t[0]
    Vh = fem.CellBasis(mesh, CFG.ELEM)
    dVh = fem.FacetBasis(mesh, CFG.ELEM)

    v_tsv = initialize_value_array(t, Vh, bsopt, is_call)

    forms = Forms(is_call=is_call, bsopt=bsopt, dynh=dynh)
    I = forms.id_bil().assemble(Vh)
    L = forms.l_bil().assemble(Vh)
    A = I - lam * dt * L
    B = A + dt * L

    for i, th_i in enumerate(t[:-1]):
        b_previous = B @ v_tsv[i]
        b_inhom = lam * forms.b_lin().assemble(dVh, th=th_i + dt) + (1 - lam) * forms.b_lin().assemble(
            dVh, th=th_i
        )
        b = b_previous + dt * b_inhom

        if dirichlet_bcs:
            u_dirichlet = Vh.project(
                lambda x: bsopt.call(th_i, x[0], dynh.mean_variance(th_i, x[1]))
                if is_call
                else bsopt.put(th_i, x[0], dynh.mean_variance(th_i, x[1]))
            )
            A_enf, b_enf = fem.enforce(
                A, b, x=u_dirichlet, D=Vh.get_dofs(dirichlet_bcs)
            )
            v_tsv[i + 1] = fem.solve(A_enf, b_enf)
        else:
            v_tsv[i + 1] = fem.solve(A, b)

        if is_american:
            v_tsv[i + 1] = np.maximum(v_tsv[i + 1], v_tsv[0])

    return v_tsv
