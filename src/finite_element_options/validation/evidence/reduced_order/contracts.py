"""Typed contracts for the public-synthetic pyMOR adoption benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite
from typing import Any

import numpy as np


SCHEMA_VERSION = "pymor-black-scholes-rom/v1"


@dataclass(frozen=True, slots=True)
class PymorBlackScholesConfig:
    """Immutable training, holdout, accuracy, and promotion controls."""

    volatility_min: float = 0.10
    volatility_max: float = 0.35
    training_volatilities: tuple[float, ...] = (
        0.10,
        0.13125,
        0.1625,
        0.19375,
        0.225,
        0.25625,
        0.2875,
        0.31875,
        0.35,
    )
    holdout_volatilities: tuple[float, ...] = (
        0.1125,
        0.1375,
        0.1875,
        0.2375,
        0.3125,
        0.3375,
    )
    refinement_level: int = 11
    time_steps: int = 160
    theta: float = 1.0
    snapshot_stride: int = 4
    max_basis_size: int = 40
    pod_rtol: float = 0.0
    rate: float = 0.05
    maturity: float = 1.0
    strike: float = 1.0
    spot: float = 1.0
    domain_max: float = 4.0
    greek_bump: float = 0.02
    affine_relative_tolerance: float = 1.0e-11
    price_abs_tolerance: float = 1.0e-7
    delta_abs_tolerance: float = 1.0e-6
    gamma_abs_tolerance: float = 1.0e-5
    fom_oracle_price_tolerance: float = 2.0e-4
    fom_oracle_delta_tolerance: float = 2.0e-3
    fom_oracle_gamma_tolerance: float = 5.0e-3
    minimum_online_speedup: float = 10.0
    maximum_ten_x_amortization_solves: int = 1000
    benchmark_repeats: int = 11
    benchmark_warmups: int = 3

    def __post_init__(self) -> None:
        """Reject invalid or contaminated train/holdout contracts."""

        positive = (
            self.volatility_min,
            self.volatility_max,
            self.maturity,
            self.strike,
            self.spot,
            self.domain_max,
            self.greek_bump,
            self.affine_relative_tolerance,
            self.price_abs_tolerance,
            self.delta_abs_tolerance,
            self.gamma_abs_tolerance,
            self.minimum_online_speedup,
        )
        if any(not isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("positive benchmark controls must be finite and positive")
        if self.volatility_min >= self.volatility_max:
            raise ValueError("volatility_min must be below volatility_max")
        if self.refinement_level < 1 or self.time_steps < 2:
            raise ValueError("refinement_level and time_steps must define a non-trivial FOM")
        if not 0.5 <= self.theta <= 1.0:
            raise ValueError("theta must lie in [0.5, 1.0]")
        if self.snapshot_stride < 1 or self.max_basis_size < 1:
            raise ValueError("snapshot_stride and max_basis_size must be positive")
        if not isfinite(self.pod_rtol) or self.pod_rtol < 0.0:
            raise ValueError("pod_rtol must be finite and non-negative")
        if self.benchmark_repeats < 3 or self.benchmark_warmups < 1:
            raise ValueError("benchmark timing requires at least three repeats and one warmup")
        if self.maximum_ten_x_amortization_solves < 1:
            raise ValueError("maximum_ten_x_amortization_solves must be positive")
        training = tuple(float(value) for value in self.training_volatilities)
        holdout = tuple(float(value) for value in self.holdout_volatilities)
        if len(training) < 3 or len(holdout) < 3:
            raise ValueError("training and holdout domains each require at least three values")
        if len(set(training)) != len(training) or len(set(holdout)) != len(holdout):
            raise ValueError("training and holdout volatility values must be unique")
        if set(training) & set(holdout):
            raise ValueError("training and holdout volatility values must be disjoint")
        if any(
            not isfinite(value) or not self.volatility_min <= value <= self.volatility_max
            for value in training + holdout
        ):
            raise ValueError("all volatility values must lie inside the declared envelope")
        if not self.greek_bump < min(self.spot, self.domain_max - self.spot):
            raise ValueError("greek_bump must remain inside the spatial domain")
        object.__setattr__(self, "training_volatilities", training)
        object.__setattr__(self, "holdout_volatilities", holdout)

    @property
    def input_hash(self) -> str:
        """Return the canonical SHA-256 of the numerical study inputs."""

        encoded = json.dumps(self.to_input_dict(), sort_keys=True, separators=(",", ":"))
        return sha256(encoded.encode("utf-8")).hexdigest()

    def to_input_dict(self) -> dict[str, Any]:
        """Return JSON-safe benchmark inputs without environment or timings."""

        return {"schema_version": SCHEMA_VERSION, **asdict(self)}


@dataclass(frozen=True, slots=True)
class OptionOutputs:
    """Price and finite-bump Delta/Gamma at the declared evaluation spot."""

    price: float
    delta: float
    gamma: float

    def __post_init__(self) -> None:
        """Require finite outputs."""

        if any(not isfinite(value) for value in (self.price, self.delta, self.gamma)):
            raise FloatingPointError("price and Greeks must be finite")

    def to_dict(self) -> dict[str, float]:
        """Return JSON-safe outputs."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class FullOrderSolution:
    """One full-order solve plus optional interior trajectory snapshots."""

    outputs: OptionOutputs
    final_interior: np.ndarray
    snapshots: np.ndarray | None
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class PODProjection:
    """pyMOR-produced POD basis and projected affine operators."""

    library: str
    library_version: str
    basis: np.ndarray
    singular_values: np.ndarray
    reduced_mass: np.ndarray
    reduced_operator_constant: np.ndarray
    reduced_operator_variance: np.ndarray
    reduced_mass_boundary: np.ndarray
    reduced_constant_boundary: np.ndarray
    reduced_variance_boundary: np.ndarray
    reduced_initial: np.ndarray
    reduced_outputs: np.ndarray
    captured_energy_fraction: float
    setup_seconds: float
    pod_seconds: float
    projection_seconds: float

    @property
    def basis_size(self) -> int:
        """Return reduced dimension."""

        return int(self.basis.shape[1])

    @property
    def memory_bytes(self) -> int:
        """Return in-memory bytes for basis and all reduced numeric arrays."""

        arrays = (
            self.basis,
            self.singular_values,
            self.reduced_mass,
            self.reduced_operator_constant,
            self.reduced_operator_variance,
            self.reduced_mass_boundary,
            self.reduced_constant_boundary,
            self.reduced_variance_boundary,
            self.reduced_initial,
            self.reduced_outputs,
        )
        return int(sum(array.nbytes for array in arrays))


class ROMEnvelopeError(ValueError):
    """Raised when a reduced solve must fail closed to full-order FEM."""

    reason = "parameter_out_of_envelope"
    fallback = "full_order_fem"


__all__ = [
    "FullOrderSolution",
    "OptionOutputs",
    "PODProjection",
    "PymorBlackScholesConfig",
    "ROMEnvelopeError",
    "SCHEMA_VERSION",
]
