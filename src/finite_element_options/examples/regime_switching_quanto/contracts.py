"""Typed contracts for research-only two-factor regime-switching quanto pricing.

The records in this module deliberately live under ``examples``: they encode the
CLP-domestic-measure assumptions needed by the research prototype and are not a
promoted core FEM capability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from finite_element_options.examples.regime_switching_quanto._types import json_safe

ContractKind = Literal[
    "composite_call",
    "composite_put",
    "composite_digital",
    "quanto_call",
    "dual_trigger_protection",
]
SUPPORTED_CONTRACT_KINDS = (
    "composite_call",
    "composite_put",
    "composite_digital",
    "quanto_call",
    "dual_trigger_protection",
)


@dataclass(frozen=True)
class TwoFactorRegimeModel:
    """Finite-state CLP domestic-``Q`` two-factor log-diffusion model.

    For regime ``i`` and log states ``x=log(S/S0)``, ``y=log(F/F0)`` the pricing
    PDE uses

    ``u_tau = 0.5 sigS_i^2 u_xx + rho_i sigS_i sigF_i u_xy
             + 0.5 sigF_i^2 u_yy + aS_i u_x + aF_i u_y
             - rd u_i + sum_j Q_ij u_j``.

    The generator is treated as a risk-neutral annual CTMC generator only under
    the explicit research assumptions documented by ``measure_note``: the fitted
    historical generator is reused under ``Q`` and volatilities may be rescaled by
    ``volatility_scale``.
    """

    equity_vol: list[float]
    fx_vol: list[float]
    correlation: list[float]
    generator: list[list[float]]
    current_probabilities: list[float]
    domestic_rate: float
    foreign_rate: float
    dividend_yield: float = 0.0
    volatility_scale: float = 1.0
    measure_note: str = (
        "Research assumption: use the same Q generator as fitted P; "
        "volatility_scale is a configurable stress multiplier."
    )

    def __post_init__(self) -> None:
        """Validate regime dimensions, stochastic laws, and finite parameters."""

        vols_s = np.asarray(self.equity_vol, dtype=float)
        vols_f = np.asarray(self.fx_vol, dtype=float)
        corr = np.asarray(self.correlation, dtype=float)
        q = np.asarray(self.generator, dtype=float)
        probs = np.asarray(self.current_probabilities, dtype=float)
        n_regimes = vols_s.size

        if n_regimes < 1:
            raise ValueError("at least one regime is required")
        if vols_f.shape != (n_regimes,) or corr.shape != (n_regimes,):
            raise ValueError("volatility and correlation arrays must have one value per regime")
        if q.shape != (n_regimes, n_regimes):
            raise ValueError("generator must be a square matrix matching regimes")
        if probs.shape != (n_regimes,):
            raise ValueError("current_probabilities must match regimes")
        if np.any(~np.isfinite(vols_s)) or np.any(~np.isfinite(vols_f)):
            raise ValueError("volatilities must be finite")
        if np.any(vols_s < 0.0) or np.any(vols_f < 0.0):
            raise ValueError("volatilities must be non-negative")
        if np.any(~np.isfinite(corr)) or np.any(np.abs(corr) > 1.0):
            raise ValueError("correlations must lie in [-1, 1]")
        if self.volatility_scale <= 0.0 or not np.isfinite(self.volatility_scale):
            raise ValueError("volatility_scale must be positive and finite")
        if np.any(~np.isfinite(q)):
            raise ValueError("generator entries must be finite")
        off_diag = q[~np.eye(n_regimes, dtype=bool)]
        if np.any(off_diag < -1.0e-12):
            raise ValueError("generator off-diagonal entries must be non-negative")
        if np.any(np.diag(q) > 1.0e-12):
            raise ValueError("generator diagonal entries must be non-positive")
        if not np.allclose(q.sum(axis=1), 0.0, atol=1.0e-10):
            raise ValueError("generator rows must sum to zero")
        if np.any(probs < -1.0e-12) or not np.isclose(probs.sum(), 1.0, atol=1.0e-10):
            raise ValueError("current_probabilities must be non-negative and sum to one")
        for name, value in (
            ("domestic_rate", self.domestic_rate),
            ("foreign_rate", self.foreign_rate),
            ("dividend_yield", self.dividend_yield),
        ):
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")

    @property
    def n_regimes(self) -> int:
        """Number of finite CTMC regimes."""

        return len(self.equity_vol)

    @property
    def scaled_equity_vol(self) -> np.ndarray:
        """Equity volatilities after applying the research stress scale."""

        return np.asarray(self.equity_vol, dtype=float) * float(self.volatility_scale)

    @property
    def scaled_fx_vol(self) -> np.ndarray:
        """FX volatilities after applying the research stress scale."""

        return np.asarray(self.fx_vol, dtype=float) * float(self.volatility_scale)

    def drifts(self) -> tuple[np.ndarray, np.ndarray]:
        """Return regime log drifts ``(aS, aF)`` under the CLP domestic measure."""

        sig_s = self.scaled_equity_vol
        sig_f = self.scaled_fx_vol
        rho = np.asarray(self.correlation, dtype=float)
        a_s = self.foreign_rate - self.dividend_yield - rho * sig_s * sig_f - 0.5 * sig_s * sig_s
        a_f = self.domestic_rate - self.foreign_rate - 0.5 * sig_f * sig_f
        return a_s, a_f

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation with no NumPy leakage."""

        payload = asdict(self)
        payload["scaled_equity_vol"] = self.scaled_equity_vol.tolist()
        payload["scaled_fx_vol"] = self.scaled_fx_vol.tolist()
        a_s, a_f = self.drifts()
        payload["equity_log_drift"] = a_s.tolist()
        payload["fx_log_drift"] = a_f.tolist()
        return json_safe(payload)

    @classmethod
    def from_calibration_result(
        cls,
        result: Any,
        *,
        domestic_rate: float,
        foreign_rate: float,
        dividend_yield: float = 0.0,
        volatility_scale: float = 1.0,
    ) -> "TwoFactorRegimeModel":
        """Build a pricing model from the local calibration result shape."""

        regimes = list(result.regimes)
        return cls(
            equity_vol=[float(regime.annual_volatility[0]) for regime in regimes],
            fx_vol=[float(regime.annual_volatility[1]) for regime in regimes],
            correlation=[float(regime.correlation[0][1]) for regime in regimes],
            generator=json_safe(result.continuous_time_generator),
            current_probabilities=json_safe(result.current_probabilities),
            domestic_rate=domestic_rate,
            foreign_rate=foreign_rate,
            dividend_yield=dividend_yield,
            volatility_scale=volatility_scale,
        )


@dataclass(frozen=True)
class ContractSpec:
    """Vectorized payoff contract for composite and quanto research products."""

    kind: ContractKind
    strike: float | None = None
    payout: float = 1.0
    fixed_fx: float | None = None
    equity_barrier: float | None = None
    fx_barrier: float | None = None

    def __post_init__(self) -> None:
        """Validate the parameters required by the selected payoff kind."""

        if self.kind not in SUPPORTED_CONTRACT_KINDS:
            raise ValueError(f"unsupported contract kind: {self.kind}")
        if self.kind in {"composite_call", "composite_put", "composite_digital", "quanto_call"}:
            _require_positive(self.strike, "strike")
        if self.kind in {"composite_digital", "dual_trigger_protection"}:
            _require_positive(self.payout, "payout")
        if self.kind == "quanto_call":
            _require_positive(self.fixed_fx, "fixed_fx")
        if self.kind == "dual_trigger_protection":
            _require_positive(self.equity_barrier, "equity_barrier")
            _require_positive(self.fx_barrier, "fx_barrier")

    def payoff(
        self,
        x: np.ndarray | float,
        y: np.ndarray | float,
        *,
        equity_spot: float,
        fx_spot: float,
    ) -> np.ndarray:
        """Evaluate the terminal payoff at log states, vectorized by NumPy."""

        if equity_spot <= 0.0 or fx_spot <= 0.0:
            raise ValueError("equity_spot and fx_spot must be positive")
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        equity = equity_spot * np.exp(x_arr)
        fx = fx_spot * np.exp(y_arr)
        composite = equity * fx
        strike = 0.0 if self.strike is None else float(self.strike)
        if self.kind == "composite_call":
            return np.maximum(composite - strike, 0.0)
        if self.kind == "composite_put":
            return np.maximum(strike - composite, 0.0)
        if self.kind == "composite_digital":
            return float(self.payout) * (composite >= strike)
        if self.kind == "quanto_call":
            fixed_fx = self.fixed_fx
            if fixed_fx is None:  # defensive; __post_init__ validates public construction
                raise ValueError("fixed_fx must be positive")
            return float(fixed_fx) * np.maximum(equity - strike, 0.0)
        equity_barrier = self.equity_barrier
        fx_barrier = self.fx_barrier
        if equity_barrier is None or fx_barrier is None:  # defensive validation guard
            raise ValueError("equity_barrier and fx_barrier must be positive")
        return float(self.payout) * ((equity <= float(equity_barrier)) & (fx >= float(fx_barrier)))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""

        return json_safe(asdict(self))


@dataclass(frozen=True)
class FEMGridSpec:
    """Structured triangular P1 finite-element grid and theta-step controls."""

    x_domain: tuple[float, float]
    y_domain: tuple[float, float]
    nx: int
    ny: int
    time_steps: int
    rannacher: bool = True
    rannacher_steps: int = 4

    def __post_init__(self) -> None:
        """Validate finite domains and supported structured-grid controls."""

        bounds = np.asarray([*self.x_domain, *self.y_domain], dtype=float)
        if not np.isfinite(bounds).all():
            raise ValueError("log-domain endpoints must be finite")
        if self.x_domain[0] >= self.x_domain[1] or self.y_domain[0] >= self.y_domain[1]:
            raise ValueError("log domains must be increasing")
        if not (
            self.x_domain[0] < 0.0 < self.x_domain[1] and self.y_domain[0] < 0.0 < self.y_domain[1]
        ):
            raise ValueError("both log domains must contain zero in their interior")
        if self.nx < 3 or self.ny < 3:
            raise ValueError("nx and ny must be at least 3")
        if self.time_steps < 2:
            raise ValueError("time_steps must be at least 2")
        if self.rannacher_steps not in (0, 4):
            raise ValueError("only zero or four Rannacher half-steps are supported")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""

        return json_safe(asdict(self))


@dataclass(frozen=True)
class FEMPriceResult:
    """Research FEM price and sparse-solve diagnostics."""

    per_regime_prices: list[float]
    mixture_price: float
    degrees_of_freedom: int
    nnz: int
    time_steps: int
    boundary_description: str
    residual: float
    factorizations: int = 0
    factorization_reuses: int = 0
    nodal_mixture_surface: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe diagnostics, omitting optional NumPy surface payloads."""

        return json_safe(
            {
                "per_regime_prices": self.per_regime_prices,
                "mixture_price": self.mixture_price,
                "degrees_of_freedom": self.degrees_of_freedom,
                "nnz": self.nnz,
                "time_steps": self.time_steps,
                "boundary_description": self.boundary_description,
                "residual": self.residual,
                "factorizations": self.factorizations,
                "factorization_reuses": self.factorization_reuses,
            }
        )


@dataclass(frozen=True)
class MonteCarloPriceResult:
    """Seeded Monte Carlo oracle price and sampling error."""

    price: float
    standard_error: float
    paths: int
    steps: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""

        return json_safe(asdict(self))


def _require_positive(value: float | None, name: str) -> None:
    if value is None or not np.isfinite(value) or float(value) <= 0.0:
        raise ValueError(f"{name} must be positive")
