"""scikit-fem research pricer for coupled two-factor regime-switching contracts."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse import block_diag, csr_matrix, kron
from scipy.sparse.linalg import splu
from skfem import Basis, BilinearForm, ElementTriP1, MeshTri, asm
from skfem.helpers import grad

from finite_element_options.examples.regime_switching_quanto.contracts import (
    ContractSpec,
    FEMGridSpec,
    FEMPriceResult,
    TwoFactorRegimeModel,
)

_BOUNDARY_DESCRIPTION = (
    "Strong all-boundary Dirichlet values use discounted deterministic/"
    "frozen-diffusion continuation: at each boundary node and regime, the payoff "
    "is evaluated at CLP-Q forward equity and FX levels with regime-frozen "
    "coefficients."
)


def price_contract_fem(
    model: TwoFactorRegimeModel,
    contract: ContractSpec,
    *,
    maturity: float,
    equity_spot: float,
    fx_spot: float,
    grid: FEMGridSpec,
    return_surface: bool = False,
) -> FEMPriceResult:
    """Price a two-factor finite-state regime-switching contract by P1 FEM.

    The backward pricing problem is advanced in time-to-maturity ``tau`` from the
    terminal payoff.  For each regime the weak generator is assembled as

    ``int v L_i u = -int grad(v)^T D_i grad(u) + int v a_i.grad(u) - rd int vu``.

    The cross derivative sign follows ``D_xy = 0.5*rho*sigS*sigF``: since
    ``div(D grad u)`` contributes ``2*D_xy*u_xy`` for constant ``D``, this yields
    exactly ``rho*sigS*sigF*u_xy`` in the strong form, without doubling the rho
    term.  CTMC coupling adds block rows ``sum_j Q_ij M u_j``.  Theta stepping
    solves ``(M - theta dt G) u_new = (M + (1-theta) dt G) u_old`` with essential
    boundary algebra ``B_ID g_old - A_ID g_new``.
    """

    if maturity <= 0.0 or not np.isfinite(maturity):
        raise ValueError("maturity must be positive and finite")
    if equity_spot <= 0.0 or fx_spot <= 0.0:
        raise ValueError("equity_spot and fx_spot must be positive")

    xs = np.linspace(grid.x_domain[0], grid.x_domain[1], grid.nx)
    ys = np.linspace(grid.y_domain[0], grid.y_domain[1], grid.ny)
    mesh = MeshTri.init_tensor(xs, ys)
    basis = Basis(mesh, ElementTriP1())
    mass = asm(_mass_form(), basis).tocsr()
    generator = _assemble_block_generator(model, basis, mass)
    mass_block = block_diag([mass] * model.n_regimes, format="csr")

    n_nodes = int(basis.N)
    n_total = n_nodes * model.n_regimes
    boundary = _block_boundary_dofs(basis, model.n_regimes)
    interior = np.setdiff1d(np.arange(n_total), boundary, assume_unique=True)
    doflocs = basis.doflocs

    values = np.tile(
        contract.payoff(doflocs[0], doflocs[1], equity_spot=equity_spot, fx_spot=fx_spot),
        model.n_regimes,
    ).astype(float)
    values[boundary] = _boundary_values(
        model,
        contract,
        doflocs,
        tau=0.0,
        equity_spot=equity_spot,
        fx_spot=fx_spot,
    )[boundary]

    residual = 0.0
    nnz = 0
    schedule = _time_schedule(maturity, grid)
    systems = {}
    for tau_old, tau_new, dt, theta in schedule:
        key = (round(float(dt), 15), float(theta))
        if key not in systems:
            left = (mass_block - theta * dt * generator).tocsr()
            right = (mass_block + (1.0 - theta) * dt * generator).tocsr()
            left_ii = left[interior, :][:, interior].tocsc()
            systems[key] = (left, right, left_ii, splu(left_ii))
        left, right, left_ii, factorization = systems[key]
        g_old = _boundary_values(
            model,
            contract,
            doflocs,
            tau=tau_old,
            equity_spot=equity_spot,
            fx_spot=fx_spot,
        )
        g_new = _boundary_values(
            model,
            contract,
            doflocs,
            tau=tau_new,
            equity_spot=equity_spot,
            fx_spot=fx_spot,
        )
        rhs = right[interior, :][:, interior] @ values[interior]
        rhs += right[interior, :][:, boundary] @ g_old[boundary]
        rhs -= left[interior, :][:, boundary] @ g_new[boundary]
        solved = np.asarray(factorization.solve(rhs), dtype=float)
        residual_vec = left_ii @ solved - rhs
        residual = float(np.linalg.norm(residual_vec) / (1.0 + np.linalg.norm(rhs)))
        values[interior] = solved
        values[boundary] = g_new[boundary]
        nnz = int(left.nnz)

    probe = basis.probes(np.array([[0.0], [0.0]]))
    per_regime = [
        float((probe @ values[r * n_nodes : (r + 1) * n_nodes])[0]) for r in range(model.n_regimes)
    ]
    probs = np.asarray(model.current_probabilities, dtype=float)
    mixture = float(np.dot(probs, per_regime))
    surface = None
    if return_surface:
        nodal = np.zeros(n_nodes)
        for regime, probability in enumerate(probs):
            nodal += probability * values[regime * n_nodes : (regime + 1) * n_nodes]
        surface = {"x": doflocs[0].copy(), "y": doflocs[1].copy(), "value": nodal}

    return FEMPriceResult(
        per_regime_prices=per_regime,
        mixture_price=mixture,
        degrees_of_freedom=n_total,
        nnz=nnz,
        time_steps=len(schedule),
        boundary_description=_BOUNDARY_DESCRIPTION,
        residual=residual,
        factorizations=len(systems),
        factorization_reuses=len(schedule) - len(systems),
        nodal_mixture_surface=surface,
    )


@BilinearForm
def _mass(u: Any, v: Any, w: Any) -> Any:
    return u * v


def _mass_form() -> BilinearForm:
    return _mass


def _assemble_block_generator(
    model: TwoFactorRegimeModel, basis: Basis, mass: csr_matrix
) -> csr_matrix:
    sig_s = model.scaled_equity_vol
    sig_f = model.scaled_fx_vol
    rho = np.asarray(model.correlation, dtype=float)
    a_s, a_f = model.drifts()
    blocks = []
    for regime in range(model.n_regimes):
        blocks.append(
            asm(
                _regime_generator_form(
                    sig_s=float(sig_s[regime]),
                    sig_f=float(sig_f[regime]),
                    rho=float(rho[regime]),
                    a_s=float(a_s[regime]),
                    a_f=float(a_f[regime]),
                    rd=float(model.domestic_rate),
                ),
                basis,
            ).tocsr()
        )
    q = csr_matrix(np.asarray(model.generator, dtype=float))
    return (block_diag(blocks, format="csr") + kron(q, mass, format="csr")).tocsr()


def _regime_generator_form(
    *, sig_s: float, sig_f: float, rho: float, a_s: float, a_f: float, rd: float
) -> BilinearForm:
    d_xx = 0.5 * sig_s * sig_s
    d_xy = 0.5 * rho * sig_s * sig_f
    d_yy = 0.5 * sig_f * sig_f

    @BilinearForm
    def generator(u: Any, v: Any, w: Any) -> Any:
        gu = grad(u)
        gv = grad(v)
        diffusion = -(
            d_xx * gv[0] * gu[0] + d_xy * (gv[0] * gu[1] + gv[1] * gu[0]) + d_yy * gv[1] * gu[1]
        )
        advection = v * (a_s * gu[0] + a_f * gu[1])
        reaction = -rd * u * v
        return diffusion + advection + reaction

    return generator


def _block_boundary_dofs(basis: Basis, n_regimes: int) -> np.ndarray:
    base = np.asarray(basis.get_dofs().all(), dtype=int)
    return np.concatenate([base + regime * basis.N for regime in range(n_regimes)])


def _time_schedule(maturity: float, grid: FEMGridSpec) -> list[tuple[float, float, float, float]]:
    dt = maturity / grid.time_steps
    if not grid.rannacher or grid.rannacher_steps == 0:
        tau = 0.0
        out = []
        for _ in range(grid.time_steps):
            out.append((tau, tau + dt, dt, 0.5))
            tau += dt
        return out
    out = []
    tau = 0.0
    half = 0.5 * dt
    for _ in range(4):
        out.append((tau, tau + half, half, 1.0))
        tau += half
    for _ in range(grid.time_steps - 2):
        out.append((tau, tau + dt, dt, 0.5))
        tau += dt
    return out


def _boundary_values(
    model: TwoFactorRegimeModel,
    contract: ContractSpec,
    doflocs: np.ndarray,
    *,
    tau: float,
    equity_spot: float,
    fx_spot: float,
) -> np.ndarray:
    values = np.empty(model.n_regimes * doflocs.shape[1], dtype=float)
    sig_s = model.scaled_equity_vol
    sig_f = model.scaled_fx_vol
    rho = np.asarray(model.correlation, dtype=float)
    discount = float(np.exp(-model.domestic_rate * tau))
    for regime in range(model.n_regimes):
        # Domestic-Q forward levels.  The equity forward includes the quanto
        # adjustment; the FX forward is the USDCLP domestic/foreign carry.
        x_forward = (
            doflocs[0]
            + (
                model.foreign_rate
                - model.dividend_yield
                - rho[regime] * sig_s[regime] * sig_f[regime]
            )
            * tau
        )
        y_forward = doflocs[1] + (model.domestic_rate - model.foreign_rate) * tau
        start = regime * doflocs.shape[1]
        stop = start + doflocs.shape[1]
        values[start:stop] = discount * contract.payoff(
            x_forward,
            y_forward,
            equity_spot=equity_spot,
            fx_spot=fx_spot,
        )
    return values
