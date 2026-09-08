"""Dependency-light contracts for the JAX-native regime research profile."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Literal

SCHEMA_VERSION = "feo-jax-regime-study-v2"
EXPECTED_LEVELS_SHA256 = "aa7ab317266bf37463e27aba9a4e990fa349bb0a6e0aefb5741e93480e0f79f4"
EXPECTED_ARCHIVE_SHA256 = "ca2debc4fcbf9bd6fb958a5cfcb986a3e8080c7923418c9d2367fcd2d9a99721"
PUBLICATION_MIN_CHAINS = 2
PUBLICATION_MIN_DRAWS_PER_CHAIN = 200
PUBLICATION_MAX_RHAT = 1.05
PUBLICATION_MIN_ESS = 100.0
DAILY_STEPS_PER_YEAR = 252
MIN_DIAGNOSTIC_DRAWS_PER_CHAIN = 4
MAX_PRICING_STEPS = 3_660
MAX_PRICING_PATH_STEPS = 131_072 * 126
MAX_POSTERIOR_PRICING_PATH_STEPS = MAX_PRICING_PATH_STEPS * 128
MAX_SEED_OFFSET = 1_701
MAX_BASE_SEED = 0xFFFF_FFFF - MAX_SEED_OFFSET
MAX_ABS_RATE = 1.0


@dataclass(frozen=True)
class NumPyroPriorConfig:
    """Immutable reference-identification prior strengths for marginalized HMM NUTS."""

    name: str = "reference"
    initial_weight: float = 20.0
    transition_weight: float = 40.0
    mean_scale_multiplier: float = 0.15
    log_scale_sd: float = 0.20
    correlation_raw_sd: float = 0.25

    def __post_init__(self) -> None:
        """Reject non-finite or non-positive prior strengths."""

        for name, value in asdict(self).items():
            if name != "name" and (not math.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be finite and positive")

    @classmethod
    def weak(cls) -> "NumPyroPriorConfig":
        """Return a deliberately weak EM-reference prior for sensitivity analysis."""

        return cls(
            name="weak",
            initial_weight=5.0,
            transition_weight=10.0,
            mean_scale_multiplier=0.50,
            log_scale_sd=0.50,
            correlation_raw_sd=0.75,
        )

    @classmethod
    def strong(cls) -> "NumPyroPriorConfig":
        """Return a deliberately strong EM-reference prior for sensitivity analysis."""

        return cls(
            name="strong",
            initial_weight=80.0,
            transition_weight=160.0,
            mean_scale_multiplier=0.075,
            log_scale_sd=0.10,
            correlation_raw_sd=0.125,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""

        return asdict(self)


@dataclass(frozen=True)
class HMMParameterSummary:
    """Immutable JSON-level HMM parameter contract."""

    initial_probs: tuple[float, ...]
    transition_matrix: tuple[tuple[float, ...], ...]
    means_percent_daily: tuple[tuple[float, ...], ...]
    covariances_percent_squared_daily: tuple[tuple[tuple[float, ...], ...], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the stable external field names."""

        return asdict(self)


@dataclass(frozen=True)
class PosteriorDiagnosticSummary:
    """Immutable promotion-relevant NumPyro diagnostic contract."""

    chains: int
    draws_per_chain: int
    divergences: int
    maximum_rhat: float | None
    minimum_ess: float | None
    finite: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""

        return asdict(self)


@dataclass(frozen=True)
class PriceEstimate:
    """Immutable Monte Carlo estimate with explicitly separated sampling error."""

    price_clp: float
    standard_error_clp: float

    def __post_init__(self) -> None:
        """Reject non-finite or negative estimates and errors."""

        if not math.isfinite(self.price_clp) or self.price_clp < 0.0:
            raise ValueError("price_clp must be finite and non-negative")
        if not math.isfinite(self.standard_error_clp) or self.standard_error_clp < 0.0:
            raise ValueError("standard_error_clp must be finite and non-negative")

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-safe representation."""

        return asdict(self)


@dataclass(frozen=True)
class EvidenceHash:
    """Immutable content identity for a generated public artifact."""

    digest: str
    filename: str
    algorithm: Literal["sha256"] = "sha256"

    def __post_init__(self) -> None:
        """Validate lowercase SHA-256 content identity."""

        if len(self.digest) != 64 or any(
            character not in "0123456789abcdef" for character in self.digest
        ):
            raise ValueError("digest must be a lowercase SHA-256 value")
        if not self.filename or Path(self.filename).name != self.filename:
            raise ValueError("filename must be a basename")


@dataclass(frozen=True)
class PromotionDecision:
    """Immutable promotion state derived from named executable gates."""

    gates: tuple[tuple[str, bool], ...]

    def __post_init__(self) -> None:
        """Reject duplicate or empty gate names."""

        names = [name for name, _passed in self.gates]
        if any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError("promotion gates must have unique non-empty names")

    @property
    def passed(self) -> bool:
        """Return whether every named gate passed."""

        return bool(self.gates) and all(passed for _name, passed in self.gates)

    def to_dict(self) -> dict[str, Any]:
        """Return canonical promotion evidence."""

        return {"diagnostics_passed": self.passed, "gates": dict(self.gates)}


@dataclass(frozen=True)
class JaxRegimeStudyConfig:
    """Bounded, reproducible settings for a research-only study run."""

    seed: int = 20260907
    num_states: int = 4
    em_iters: int = 250
    holdout_fraction: float = 0.15
    warmup: int = 300
    posterior_samples: int = 300
    chains: int = 2
    pricing_paths: int = 4096
    maturity_years: float = 0.5
    steps_per_year: int = 252
    domestic_rate: float = 0.045
    foreign_rate: float = 0.0439
    dividend_yield: float = 0.0
    research_only: Literal[True] = True
    market_calibrated: Literal[False] = False
    production_ready: Literal[False] = False

    def __post_init__(self) -> None:
        """Reject weakened evidence or non-positive computational settings."""

        positive = {
            "num_states": self.num_states,
            "em_iters": self.em_iters,
            "warmup": self.warmup,
            "posterior_samples": self.posterior_samples,
            "chains": self.chains,
            "pricing_paths": self.pricing_paths,
            "steps_per_year": self.steps_per_year,
        }
        integers = {"seed": self.seed, **positive}
        for integer_name, integer_value in integers.items():
            if isinstance(integer_value, bool) or not isinstance(integer_value, Integral):
                raise ValueError(f"{integer_name} must be an integer")
            object.__setattr__(self, integer_name, int(integer_value))
        if not 0 <= self.seed <= MAX_BASE_SEED:
            raise ValueError(
                f"seed must lie in [0, {MAX_BASE_SEED}] to reserve deterministic offsets"
            )
        real_settings = {
            "holdout_fraction": self.holdout_fraction,
            "maturity_years": self.maturity_years,
            "domestic_rate": self.domestic_rate,
            "foreign_rate": self.foreign_rate,
            "dividend_yield": self.dividend_yield,
        }
        for real_name, real_value in real_settings.items():
            if (
                isinstance(real_value, bool)
                or not isinstance(real_value, Real)
                or not math.isfinite(real_value)
            ):
                raise ValueError(f"{real_name} must be a finite real number")
            object.__setattr__(self, real_name, float(real_value))
        rate_settings = {
            "domestic_rate": self.domestic_rate,
            "foreign_rate": self.foreign_rate,
            "dividend_yield": self.dividend_yield,
        }
        for rate_name, rate_value in rate_settings.items():
            if abs(rate_value) > MAX_ABS_RATE:
                raise ValueError(f"{rate_name} must lie in [-{MAX_ABS_RATE}, {MAX_ABS_RATE}]")
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.pricing_paths < 2:
            raise ValueError("pricing_paths must be at least 2 for finite sample diagnostics")
        if self.posterior_samples < MIN_DIAGNOSTIC_DRAWS_PER_CHAIN:
            raise ValueError(
                "posterior_samples must be at least "
                f"{MIN_DIAGNOSTIC_DRAWS_PER_CHAIN} for finite split-chain diagnostics"
            )
        if self.chains < PUBLICATION_MIN_CHAINS:
            raise ValueError(
                f"chains must be at least {PUBLICATION_MIN_CHAINS} for between-chain diagnostics"
            )
        maxima = {
            "em_iters": 1_000,
            "warmup": 5_000,
            "posterior_samples": 10_000,
            "chains": 8,
            "pricing_paths": 1_000_000,
        }
        for name, maximum in maxima.items():
            if positive[name] > maximum:
                raise ValueError(f"{name} must not exceed {maximum:,}")
        if self.num_states not in {2, 3, 4}:
            raise ValueError("num_states must be one of the evaluated candidates: 2, 3, or 4")
        if not 0.05 <= self.holdout_fraction <= 0.4:
            raise ValueError("holdout_fraction must lie in [0.05, 0.4]")
        if self.maturity_years <= 0.0:
            raise ValueError("maturity_years must be finite and positive")
        if self.steps_per_year != DAILY_STEPS_PER_YEAR:
            raise ValueError(
                f"steps_per_year must be {DAILY_STEPS_PER_YEAR}; the fitted transition matrix is daily"
            )
        raw_pricing_steps = self.maturity_years * self.steps_per_year
        pricing_steps = self.pricing_steps
        if pricing_steps < 1:
            raise ValueError("maturity_years must produce at least one daily pricing step")
        if not math.isclose(raw_pricing_steps, pricing_steps, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("maturity_years must represent a whole number of daily pricing steps")
        if pricing_steps > MAX_PRICING_STEPS:
            raise ValueError(
                f"maturity_years must not produce more than {MAX_PRICING_STEPS:,} pricing steps"
            )
        path_steps = self.pricing_paths * pricing_steps
        if path_steps > MAX_PRICING_PATH_STEPS:
            raise ValueError(
                "pricing_paths * pricing_steps must not exceed "
                f"{MAX_PRICING_PATH_STEPS:,} path-steps"
            )
        if (
            self.research_only is not True
            or self.market_calibrated is not False
            or self.production_ready is not False
        ):
            raise ValueError("JAX regime profile is research-only and not market calibrated")

    @property
    def pricing_steps(self) -> int:
        """Return the validated number of daily regime/diffusion intervals."""

        return int(round(self.maturity_years * self.steps_per_year))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""

        return asdict(self)


@dataclass(frozen=True)
class PDPPreprocessingAudit:
    """Immutable audit trail for exclusions before return construction."""

    source_level_rows: int
    valid_level_rows: int
    missing_or_unparseable_equity_rows: int
    missing_or_unparseable_fx_rows: int
    invalid_equity_rows: int
    invalid_fx_rows: int
    quarantined_rows: tuple[tuple[str, str], ...]
    return_construction_rule: Literal[
        "adjacent_valid_joint_levels_after_exclusions_no_calendar_gap_rescaling"
    ] = "adjacent_valid_joint_levels_after_exclusions_no_calendar_gap_rescaling"

    def __post_init__(self) -> None:
        """Require a complete, nonnegative partition of source rows."""

        counts = (
            self.valid_level_rows,
            self.missing_or_unparseable_equity_rows,
            self.missing_or_unparseable_fx_rows,
            self.invalid_equity_rows,
            self.invalid_fx_rows,
        )
        if self.source_level_rows <= 0 or any(count < 0 for count in counts):
            raise ValueError("preprocessing row counts must be nonnegative with a positive source")
        if sum(counts) != self.source_level_rows:
            raise ValueError("preprocessing row counts must partition source_level_rows")
        if len(self.quarantined_rows) != self.invalid_equity_rows + self.invalid_fx_rows:
            raise ValueError("quarantined rows must enumerate all invalid finite-level exclusions")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe counts, reasons, and return-construction policy."""

        return {
            **asdict(self),
            "excluded_level_rows": self.source_level_rows - self.valid_level_rows,
            "quarantined_rows": [
                {"date": date, "reason": reason} for date, reason in self.quarantined_rows
            ],
        }


@dataclass(frozen=True)
class PDPObservationBatch:
    """Validated bivariate return observations from the immutable PDP export."""

    dates: tuple[str, ...]
    returns: tuple[tuple[float, float], ...]
    levels_sha256: str
    archive_sha256: str
    requested_start: str
    requested_end_exclusive: str
    equity_spot: float
    fx_spot: float
    preprocessing: PDPPreprocessingAudit
    series: tuple[str, str] = ("^GSPC", "CLP=X")
    units: Literal["decimal_daily_log_return"] = "decimal_daily_log_return"
    classification: Literal["public_research"] = "public_research"

    def __post_init__(self) -> None:
        """Validate alignment, provenance hashes, and finite bivariate rows."""

        if len(self.dates) != len(self.returns) or not self.returns:
            raise ValueError("dates and returns must be non-empty and aligned")
        for name, digest in (
            ("levels_sha256", self.levels_sha256),
            ("archive_sha256", self.archive_sha256),
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if any(len(row) != 2 for row in self.returns):
            raise ValueError("each return row must contain equity and FX values")
        if any(not math.isfinite(value) for row in self.returns for value in row):
            raise ValueError("all return values must be finite")
        if not math.isfinite(self.equity_spot) or self.equity_spot <= 0.0:
            raise ValueError("equity_spot must be finite and positive")
        if not math.isfinite(self.fx_spot) or self.fx_spot <= 0.0:
            raise ValueError("fx_spot must be finite and positive")

    @property
    def row_count(self) -> int:
        """Return the number of aligned joint returns."""

        return len(self.returns)

    def to_dict(self, *, include_observations: bool = False) -> dict[str, Any]:
        """Return JSON-safe provenance, optionally including all observations."""

        payload: dict[str, Any] = {
            "archive_sha256": self.archive_sha256,
            "classification": self.classification,
            "date_start": self.dates[0],
            "date_end": self.dates[-1],
            "levels_sha256": self.levels_sha256,
            "equity_spot": self.equity_spot,
            "fx_spot": self.fx_spot,
            "requested_end_exclusive": self.requested_end_exclusive,
            "requested_start": self.requested_start,
            "row_count": self.row_count,
            "series": list(self.series),
            "units": self.units,
            "preprocessing": self.preprocessing.to_dict(),
        }
        if include_observations:
            payload["dates"] = list(self.dates)
            payload["returns"] = [list(row) for row in self.returns]
        return payload
