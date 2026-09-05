"""Contracts for the isolated Python 3.12 Bayesian/JAX research profile."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
import sys
from typing import Any, Final

from finite_element_options.contracts.evidence_serialization import canonical_json_sha256


SCHEMA_VERSION: Final = "bayesian-jax-profile/v1"
SUPPORTED_PYTHON: Final = (3, 12)


def require_supported_python() -> None:
    """Fail closed outside the evidence-backed Python 3.12 profile."""

    if sys.version_info[:2] != SUPPORTED_PYTHON:
        raise RuntimeError(
            "the Bayesian research profiles require Python 3.12; "
            f"received {sys.version_info.major}.{sys.version_info.minor}"
        )


OBSERVATIONS = (
    1.62,
    1.34,
    1.51,
    1.77,
    1.28,
    1.44,
    1.69,
    1.55,
    1.36,
    1.48,
    1.72,
    1.58,
    1.41,
    1.63,
    1.33,
    1.49,
    1.67,
    1.52,
    1.39,
    1.61,
)


@dataclass(frozen=True, slots=True)
class BayesianSmokeConfig:
    """Deterministic synthetic posterior and diagnostic controls."""

    observations: tuple[float, ...] = OBSERVATIONS
    known_sigma: float = 0.4
    prior_mean: float = 0.0
    prior_sigma: float = 2.0
    chains: int = 2
    warmup: int = 300
    draws: int = 300
    target_accept: float = 0.9
    pymc_seed: int = 13_701
    pymc_predictive_seed: int = 13_703
    numpyro_seed: int = 13_711
    numpyro_predictive_seed: int = 13_712
    posterior_mean_tolerance: float = 0.04
    posterior_sd_tolerance: float = 0.03
    predictive_mean_tolerance: float = 0.12
    predictive_sd_tolerance: float = 0.04
    maximum_rhat: float = 1.05
    minimum_bulk_ess: float = 100.0
    maximum_cross_engine_mean_difference: float = 0.04
    maximum_cross_engine_sd_difference: float = 0.02
    maximum_cross_engine_predictive_mean_difference: float = 0.04
    maximum_cross_engine_predictive_sd_difference: float = 0.04

    def __post_init__(self) -> None:
        """Reject underidentified or weakened smoke controls."""

        if len(self.observations) < 10 or any(not isfinite(item) for item in self.observations):
            raise ValueError("Bayesian smoke requires at least ten finite observations")
        positive = (
            self.known_sigma,
            self.prior_sigma,
            self.target_accept,
            self.posterior_mean_tolerance,
            self.posterior_sd_tolerance,
            self.predictive_mean_tolerance,
            self.predictive_sd_tolerance,
            self.maximum_rhat,
            self.minimum_bulk_ess,
            self.maximum_cross_engine_mean_difference,
            self.maximum_cross_engine_sd_difference,
            self.maximum_cross_engine_predictive_mean_difference,
            self.maximum_cross_engine_predictive_sd_difference,
        )
        if any(not isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("Bayesian smoke controls must be finite and positive")
        if self.chains < 2 or self.warmup < 300 or self.draws < 300:
            raise ValueError("Bayesian diagnostics require >=2 chains and >=300 tune/draws")
        if not 0.9 <= self.target_accept < 1.0:
            raise ValueError("target_accept cannot be below the evidenced 0.9")
        if self.maximum_rhat > 1.05 or self.minimum_bulk_ess < 100.0:
            raise ValueError("Bayesian convergence controls cannot be weakened")
        accuracy_limits = (
            ("posterior_mean_tolerance", self.posterior_mean_tolerance, 0.04),
            ("posterior_sd_tolerance", self.posterior_sd_tolerance, 0.03),
            ("predictive_mean_tolerance", self.predictive_mean_tolerance, 0.12),
            ("predictive_sd_tolerance", self.predictive_sd_tolerance, 0.04),
            (
                "maximum_cross_engine_mean_difference",
                self.maximum_cross_engine_mean_difference,
                0.04,
            ),
            (
                "maximum_cross_engine_sd_difference",
                self.maximum_cross_engine_sd_difference,
                0.02,
            ),
            (
                "maximum_cross_engine_predictive_mean_difference",
                self.maximum_cross_engine_predictive_mean_difference,
                0.04,
            ),
            (
                "maximum_cross_engine_predictive_sd_difference",
                self.maximum_cross_engine_predictive_sd_difference,
                0.04,
            ),
        )
        weakened = [name for name, value, limit in accuracy_limits if value > limit]
        if weakened:
            raise ValueError(f"Bayesian accuracy controls cannot be weakened: {weakened}")
        if any(
            not isinstance(seed, int) or seed < 0
            for seed in (
                self.pymc_seed,
                self.pymc_predictive_seed,
                self.numpyro_seed,
                self.numpyro_predictive_seed,
            )
        ):
            raise ValueError("Bayesian seeds must be non-negative integers")
        object.__setattr__(self, "observations", tuple(float(item) for item in self.observations))

    @property
    def input_hash(self) -> str:
        """Return the canonical hash of data, seeds, and diagnostic thresholds."""

        return canonical_json_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe immutable study inputs."""

        return {"schema_version": SCHEMA_VERSION, **asdict(self)}


__all__ = [
    "BayesianSmokeConfig",
    "OBSERVATIONS",
    "SCHEMA_VERSION",
    "SUPPORTED_PYTHON",
    "require_supported_python",
]
