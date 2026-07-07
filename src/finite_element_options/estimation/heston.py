"""Calibration adapters and synthetic surface fixtures.

The historical module name is retained for compatibility, but the Heston-named
calibrator is fail-closed until a real Heston pricing engine is wired in. The
toy formula used by tests and examples lives behind explicitly synthetic class
names so it cannot be mistaken for production Heston calibration.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .calibrator import CalibrationResult, Calibrator, ParameterVector

_SYNTHETIC_PARAMETER_NAMES = (
    "level",
    "strike_slope",
    "maturity_slope",
    "sqrt_strike_slope",
    "maturity_quadratic",
)

HESTON_PARAMETER_NAMES = ("v0", "kappa", "theta", "sigma", "rho")
_ALLOWED_FELLER_POLICIES = frozenset({"report", "enforce"})
_ALLOWED_LIKELIHOOD_UNITS = frozenset({"price", "implied_volatility"})
_FORBIDDEN_ENGINE_TOKENS = ("toy", "synthetic", "polynomial", "fixture")


@dataclass(frozen=True)
class HestonConstraintReport:
    """Constraint and Feller diagnostics for Heston posterior draws."""

    parameter_names: tuple[str, ...]
    draw_count: int
    feller_policy: str
    feller_ratio_min: float
    feller_ratio_max: float
    feller_condition_satisfied: bool

    def as_dict(self) -> dict[str, object]:
        """Return JSON-friendly diagnostics for provenance records."""

        return {
            "parameter_names": self.parameter_names,
            "draw_count": self.draw_count,
            "feller_policy": self.feller_policy,
            "feller_ratio_min": self.feller_ratio_min,
            "feller_ratio_max": self.feller_ratio_max,
            "feller_condition_satisfied": self.feller_condition_satisfied,
        }


@dataclass(frozen=True)
class HestonMCMCDiagnosticThresholds:
    """Acceptance thresholds for constrained Heston Bayesian calibration."""

    max_r_hat: float = 1.01
    min_bulk_ess: float = 400.0
    min_tail_ess: float = 400.0
    max_divergences: int = 0
    max_tree_depth_hits: int = 0
    max_heldout_rmse: float | None = None

    def __post_init__(self) -> None:
        """Validate monotone sampler/predictive gate thresholds."""

        if self.max_r_hat < 1.0:
            raise ValueError("max_r_hat must be at least 1.0")
        if self.min_bulk_ess < 0.0 or self.min_tail_ess < 0.0:
            raise ValueError("ESS thresholds must be non-negative")
        if self.max_divergences < 0 or self.max_tree_depth_hits < 0:
            raise ValueError("sampler failure thresholds must be non-negative")
        if self.max_heldout_rmse is not None and self.max_heldout_rmse < 0.0:
            raise ValueError("max_heldout_rmse must be non-negative")

    def as_dict(self) -> dict[str, object]:
        """Return JSON-friendly threshold metadata."""

        return {
            "max_r_hat": self.max_r_hat,
            "min_bulk_ess": self.min_bulk_ess,
            "min_tail_ess": self.min_tail_ess,
            "max_divergences": self.max_divergences,
            "max_tree_depth_hits": self.max_tree_depth_hits,
            "max_heldout_rmse": self.max_heldout_rmse,
        }


@dataclass(frozen=True)
class HestonMCMCDiagnosticReport:
    """Result of applying MCMC and predictive acceptance gates."""

    accepted: bool
    failures: tuple[str, ...]
    metrics: Mapping[str, object]
    thresholds: HestonMCMCDiagnosticThresholds

    def as_dict(self) -> dict[str, object]:
        """Return JSON-friendly MCMC diagnostics."""

        return {
            "accepted": self.accepted,
            "failures": self.failures,
            "metrics": dict(self.metrics),
            "thresholds": self.thresholds.as_dict(),
        }


def _coerce_heston_draws(
    posterior_draws: Mapping[str, Sequence[float] | np.ndarray],
) -> dict[str, np.ndarray]:
    missing = [name for name in HESTON_PARAMETER_NAMES if name not in posterior_draws]
    if missing:
        raise ValueError(f"missing Heston posterior draw(s): {missing}")
    arrays: dict[str, np.ndarray] = {}
    shape: tuple[int, ...] | None = None
    for name in HESTON_PARAMETER_NAMES:
        arr = np.asarray(posterior_draws[name], dtype=float)
        if arr.size == 0:
            raise ValueError(f"{name} posterior draws must not be empty")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} posterior draws must be finite")
        if shape is None:
            shape = arr.shape
        elif arr.shape != shape:
            raise ValueError("Heston posterior draws must have matching shapes")
        arrays[name] = arr.reshape(-1)
    return arrays


def validate_heston_posterior_draws(
    posterior_draws: Mapping[str, Sequence[float] | np.ndarray],
    *,
    feller_policy: str = "report",
) -> HestonConstraintReport:
    """Validate Heston posterior draws and report Feller diagnostics.

    The Heston variance process requires positive ``v0``, ``kappa``, ``theta``
    and volatility-of-variance ``sigma`` plus ``rho`` strictly between -1 and 1.
    The Feller condition can be reported for model-risk review or enforced as a
    hard acceptance gate when the solver route requires strict positivity away
    from the boundary.
    """

    if feller_policy not in _ALLOWED_FELLER_POLICIES:
        raise ValueError(f"feller_policy must be one of {sorted(_ALLOWED_FELLER_POLICIES)}")
    draws = _coerce_heston_draws(posterior_draws)
    for name in ("v0", "kappa", "theta", "sigma"):
        if np.any(draws[name] <= 0.0):
            raise ValueError(f"{name} posterior draws must be strictly positive")
    if np.any((draws["rho"] <= -1.0) | (draws["rho"] >= 1.0)):
        raise ValueError("rho posterior draws must remain strictly between -1 and 1")

    feller_ratio = 2.0 * draws["kappa"] * draws["theta"] / draws["sigma"] ** 2
    feller_condition_satisfied = bool(np.all(feller_ratio >= 1.0))
    if feller_policy == "enforce" and not feller_condition_satisfied:
        raise ValueError("Feller condition violated by at least one posterior draw")
    return HestonConstraintReport(
        parameter_names=HESTON_PARAMETER_NAMES,
        draw_count=int(draws["v0"].size),
        feller_policy=feller_policy,
        feller_ratio_min=float(np.min(feller_ratio)),
        feller_ratio_max=float(np.max(feller_ratio)),
        feller_condition_satisfied=feller_condition_satisfied,
    )


def _diagnostic_frame(
    diagnostic_summary: pd.DataFrame | Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    if isinstance(diagnostic_summary, pd.DataFrame):
        frame = diagnostic_summary.copy()
    else:
        frame = pd.DataFrame.from_dict(dict(diagnostic_summary), orient="index")
    required_columns = {"r_hat", "ess_bulk", "ess_tail"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            "diagnostic_summary must contain r_hat, ess_bulk and ess_tail; "
            f"missing {sorted(missing_columns)}"
        )
    missing_parameters = [name for name in HESTON_PARAMETER_NAMES if name not in frame.index]
    if missing_parameters:
        raise ValueError(f"diagnostic_summary is missing {missing_parameters}")
    return frame.loc[list(HESTON_PARAMETER_NAMES), ["r_hat", "ess_bulk", "ess_tail"]].astype(float)


def evaluate_heston_mcmc_diagnostics(
    diagnostic_summary: pd.DataFrame | Mapping[str, Mapping[str, float]],
    *,
    divergences: int = 0,
    tree_depth_hits: int = 0,
    heldout_rmse: float | None = None,
    thresholds: HestonMCMCDiagnosticThresholds | None = None,
) -> HestonMCMCDiagnosticReport:
    """Apply Heston Bayesian MCMC and predictive acceptance thresholds."""

    if thresholds is None:
        thresholds = HestonMCMCDiagnosticThresholds()
    if divergences < 0 or tree_depth_hits < 0:
        raise ValueError("sampler failure counts must be non-negative")
    if heldout_rmse is not None:
        if not np.isfinite(heldout_rmse):
            raise ValueError("heldout_rmse must be finite when supplied")
        if heldout_rmse < 0.0:
            raise ValueError("heldout_rmse must be non-negative when supplied")
    frame = _diagnostic_frame(diagnostic_summary)
    if not np.all(np.isfinite(frame.to_numpy())):
        raise ValueError("MCMC diagnostic summary must be finite")

    failures: list[str] = []
    max_r_hat = float(frame["r_hat"].max())
    min_bulk_ess = float(frame["ess_bulk"].min())
    min_tail_ess = float(frame["ess_tail"].min())
    if max_r_hat > thresholds.max_r_hat:
        failures.append(f"r_hat {max_r_hat:.4g} exceeds {thresholds.max_r_hat:.4g}")
    if min_bulk_ess < thresholds.min_bulk_ess:
        failures.append(f"bulk ESS {min_bulk_ess:.4g} below {thresholds.min_bulk_ess:.4g}")
    if min_tail_ess < thresholds.min_tail_ess:
        failures.append(f"tail ESS {min_tail_ess:.4g} below {thresholds.min_tail_ess:.4g}")
    if divergences > thresholds.max_divergences:
        failures.append(f"divergence count {divergences} exceeds {thresholds.max_divergences}")
    if tree_depth_hits > thresholds.max_tree_depth_hits:
        failures.append(
            f"tree depth hit count {tree_depth_hits} exceeds {thresholds.max_tree_depth_hits}"
        )
    if thresholds.max_heldout_rmse is not None:
        if heldout_rmse is None:
            failures.append("heldout RMSE is required by thresholds")
        elif heldout_rmse > thresholds.max_heldout_rmse:
            failures.append(
                f"heldout RMSE {heldout_rmse:.4g} exceeds {thresholds.max_heldout_rmse:.4g}"
            )
    metrics = {
        "max_r_hat": max_r_hat,
        "min_bulk_ess": min_bulk_ess,
        "min_tail_ess": min_tail_ess,
        "divergences": int(divergences),
        "tree_depth_hits": int(tree_depth_hits),
        "heldout_rmse": None if heldout_rmse is None else float(heldout_rmse),
    }
    return HestonMCMCDiagnosticReport(
        accepted=not failures,
        failures=tuple(failures),
        metrics=metrics,
        thresholds=thresholds,
    )


def _posterior_means(draws: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.asarray([np.mean(draws[name]) for name in HESTON_PARAMETER_NAMES], dtype=float)


def _validate_heston_engine_name(pricing_engine: str) -> str:
    normalized = pricing_engine.strip()
    if not normalized:
        raise ValueError("pricing_engine must name a validated Heston pricing engine")
    lowered = normalized.lower()
    if "heston" not in lowered or any(token in lowered for token in _FORBIDDEN_ENGINE_TOKENS):
        raise ValueError("pricing_engine must name a validated Heston pricing engine")
    return normalized


def _validate_heston_engine_metadata(
    pricing_engine: str,
    pricing_engine_validation: Mapping[str, object],
) -> dict[str, object]:
    """Validate non-lexical evidence for a Heston pricing engine."""

    engine = _validate_heston_engine_name(pricing_engine)
    metadata = dict(pricing_engine_validation)
    if metadata.get("validated") is not True:
        raise ValueError("pricing_engine_validation must mark the engine as validated")
    if str(metadata.get("engine_family", "")).lower() != "heston":
        raise ValueError("pricing_engine_validation must declare engine_family='heston'")
    artifact = str(metadata.get("validation_artifact", "")).strip()
    if not artifact:
        raise ValueError("pricing_engine_validation must include a validation_artifact")
    artifact_sha256 = str(metadata.get("validation_artifact_sha256", "")).strip().lower()
    if len(artifact_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in artifact_sha256):
        raise ValueError("pricing_engine_validation must include validation_artifact_sha256")
    version = str(metadata.get("version", "")).strip()
    if not version:
        raise ValueError("pricing_engine_validation must include a pricing engine version")
    metadata["pricing_engine"] = engine
    metadata["validation_artifact"] = artifact
    metadata["validation_artifact_sha256"] = artifact_sha256
    metadata["version"] = version
    return metadata


def _active_parameter_mask(
    parameters: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> np.ndarray:
    """Return SciPy-style bound activity mask for posterior summary parameters."""

    rtol = 1e-10
    atol = 1e-12
    lower_active = np.isfinite(lower_bounds) & np.isclose(
        parameters,
        lower_bounds,
        rtol=rtol,
        atol=atol,
    )
    upper_active = np.isfinite(upper_bounds) & np.isclose(
        parameters,
        upper_bounds,
        rtol=rtol,
        atol=atol,
    )
    mask = np.zeros(parameters.shape, dtype=int)
    mask[lower_active] = -1
    mask[upper_active & ~lower_active] = 1
    return mask


def build_heston_bayesian_calibration_result(
    *,
    posterior_draws: Mapping[str, Sequence[float] | np.ndarray],
    diagnostic_summary: pd.DataFrame | Mapping[str, Mapping[str, float]],
    observed_values: Sequence[float] | np.ndarray,
    fitted_values: Sequence[float] | np.ndarray,
    inference_data_artifact: str,
    pricing_engine: str,
    pricing_engine_validation: Mapping[str, object],
    likelihood_units: str,
    observation_noise: float,
    random_seed: int | None,
    thresholds: HestonMCMCDiagnosticThresholds | None = None,
    feller_policy: str = "report",
    divergences: int = 0,
    tree_depth_hits: int = 0,
    heldout_rmse: float | None = None,
) -> CalibrationResult:
    """Build an auditable PyMC Heston calibration result from validated artifacts.

    This helper does not price options itself. It accepts posterior draws and
    fitted values generated by a separately validated Heston pricing engine, then
    applies constraint, likelihood and sampler diagnostics before marking the run
    successful. It refuses toy/synthetic/polynomial engines so a fixture cannot be
    mislabeled as Heston calibration evidence.
    """

    engine_metadata = _validate_heston_engine_metadata(pricing_engine, pricing_engine_validation)
    if likelihood_units not in _ALLOWED_LIKELIHOOD_UNITS:
        raise ValueError(f"likelihood_units must be one of {sorted(_ALLOWED_LIKELIHOOD_UNITS)}")
    if observation_noise <= 0.0 or not np.isfinite(observation_noise):
        raise ValueError("observation_noise must be finite and strictly positive")
    if not inference_data_artifact:
        raise ValueError("inference_data_artifact must name a retained artifact")

    draws = _coerce_heston_draws(posterior_draws)
    constraint_report = validate_heston_posterior_draws(
        posterior_draws,
        feller_policy=feller_policy,
    )
    observed = np.asarray(observed_values, dtype=float)
    fitted = np.asarray(fitted_values, dtype=float)
    if observed.shape != fitted.shape:
        raise ValueError("observed_values and fitted_values must have matching shapes")
    if observed.size == 0 or not np.all(np.isfinite(observed)) or not np.all(np.isfinite(fitted)):
        raise ValueError("observed_values and fitted_values must be non-empty and finite")
    residuals = fitted - observed
    residuals_flat = np.ravel(residuals)
    fit_rmse = float(np.sqrt(np.mean(residuals_flat**2)))
    mcmc_report = evaluate_heston_mcmc_diagnostics(
        diagnostic_summary,
        divergences=divergences,
        tree_depth_hits=tree_depth_hits,
        heldout_rmse=heldout_rmse,
        thresholds=thresholds,
    )
    parameters = _posterior_means(draws)
    lower_bounds = np.array([0.0, 0.0, 0.0, 0.0, -1.0], dtype=float)
    upper_bounds = np.array([np.inf, np.inf, np.inf, np.inf, 1.0], dtype=float)
    diagnostics = {
        "constraints": constraint_report.as_dict(),
        "mcmc": mcmc_report.as_dict(),
        "likelihood": {
            "units": likelihood_units,
            "observation_noise": float(observation_noise),
            "fit_rmse": fit_rmse,
            "heldout_rmse": None if heldout_rmse is None else float(heldout_rmse),
        },
    }
    provenance = {
        "pricing_engine": engine_metadata["pricing_engine"],
        "pricing_engine_validation": engine_metadata,
        "likelihood_units": likelihood_units,
        "observation_noise": float(observation_noise),
        "random_seed": random_seed,
        "feller_policy": feller_policy,
        "draw_count": constraint_report.draw_count,
    }
    return CalibrationResult(
        parameters=parameters,
        success=mcmc_report.accepted,
        status=1 if mcmc_report.accepted else 0,
        message=(
            "PyMC Heston diagnostics accepted"
            if mcmc_report.accepted
            else "PyMC Heston diagnostics rejected: " + "; ".join(mcmc_report.failures)
        ),
        residuals=np.asarray(residuals, dtype=float),
        cost=float(0.5 * np.sum(residuals_flat**2)),
        optimality=float("nan"),
        jacobian_rank=0,
        jacobian_condition=np.inf,
        bounds=(lower_bounds, upper_bounds),
        active_mask=_active_parameter_mask(parameters, lower_bounds, upper_bounds),
        nfev=constraint_report.draw_count,
        njev=None,
        method="pymc.heston.diagnostics",
        parameter_names=HESTON_PARAMETER_NAMES,
        diagnostics=diagnostics,
        artifacts=(inference_data_artifact,),
        provenance=provenance,
    )


class SyntheticSurfaceCalibrator(Calibrator):
    """Calibrate a documented synthetic option-surface fixture.

    This class is intentionally not a Heston model. It exists for deterministic
    examples and tests while real Heston calibration remains blocked behind a
    supported pricing engine and model-risk diagnostics.
    """

    parameter_names = _SYNTHETIC_PARAMETER_NAMES

    @staticmethod
    def price_formula(
        strikes: np.ndarray, maturities: np.ndarray, params: ParameterVector
    ) -> np.ndarray:
        """Synthetic pricing formula used for fixtures only.

        Parameters
        ----------
        strikes, maturities:
            Arrays defining the option surface.
        params:
            Sequence ``[level, strike_slope, maturity_slope,
            sqrt_strike_slope, maturity_quadratic]``.
        """

        level, strike_slope, maturity_slope, sqrt_strike_slope, maturity_quadratic = params
        return (
            level
            + 1e-2 * strike_slope * strikes
            + maturity_slope * maturities
            + 1e-1 * sqrt_strike_slope * np.sqrt(strikes)
            + maturity_quadratic * maturities**2
        )

    def model_prices(self, params: ParameterVector) -> np.ndarray:
        """Return prices implied by ``params`` across the market grid."""
        return self.price_formula(self.strikes, self.maturities, params)


class HestonCalibrator(Calibrator):
    """Fail-closed placeholder for real Heston calibration.

    The previous implementation used a toy polynomial under a Heston name. That
    is model-risk unsafe, so this compatibility class refuses to calibrate until
    a real Heston pricing route and diagnostics are implemented.
    """

    parameter_names = ("v0", "kappa", "theta", "sigma", "rho")

    @staticmethod
    def _unsupported_message() -> str:
        return (
            "HestonCalibrator requires a real Heston pricing engine and model-risk "
            "diagnostics; use SyntheticSurfaceCalibrator only for synthetic fixtures."
        )

    def model_prices(self, params: ParameterVector) -> np.ndarray:
        """Refuse to price through a toy Heston route."""
        raise NotImplementedError(self._unsupported_message())

    def calibrate(
        self,
        initial_guess: ParameterVector,
        *,
        bounds: tuple[Sequence[float] | float, Sequence[float] | float] | None = None,
        weights: Sequence[float] | None = None,
        loss: str = "linear",
    ) -> CalibrationResult:
        """Refuse Heston calibration until a real model implementation exists."""
        del initial_guess, bounds, weights, loss
        raise NotImplementedError(self._unsupported_message())


class StatsmodelsCalibrator(SyntheticSurfaceCalibrator):
    """Deprecated compatibility shim for the removed Statsmodels NLS adapter."""

    def calibrate(
        self,
        initial_guess: ParameterVector,
        *,
        bounds: tuple[Sequence[float] | float, Sequence[float] | float] | None = None,
        weights: Sequence[float] | None = None,
        loss: str = "linear",
    ) -> CalibrationResult:
        """Delegate to the supported SciPy least-squares adapter."""
        warnings.warn(
            "StatsmodelsCalibrator no longer uses private statsmodels NLS APIs; "
            "it delegates to SciPy least_squares and will be removed after the "
            "compatibility window.",
            DeprecationWarning,
            stacklevel=2,
        )
        return super().calibrate(
            initial_guess,
            bounds=bounds,
            weights=weights,
            loss=loss,
        )


class PyMCCalibrator(SyntheticSurfaceCalibrator):
    """Bayesian calibration of the synthetic fixture with :mod:`pymc`.

    This is not a Heston calibration route. It returns posterior means plus a
    structured diagnostic shell for fixture experiments only.
    """

    def calibrate(  # type: ignore[override]
        self,
        draws: int = 1000,
        chains: int = 2,
        tune: int | None = None,
        random_seed: int | None = 123,
        target_accept: float = 0.9,
    ) -> CalibrationResult:
        """Return posterior means for the synthetic fixture parameters."""

        import pymc as pm

        if tune is None:
            tune = draws

        strikes, maturities = self.strikes, self.maturities

        with pm.Model():
            level = pm.Normal("level", mu=0.04, sigma=0.1)
            strike_slope = pm.Normal("strike_slope", mu=1.0, sigma=0.5)
            maturity_slope = pm.Normal("maturity_slope", mu=0.04, sigma=0.1)
            sqrt_strike_slope = pm.HalfNormal("sqrt_strike_slope", sigma=0.3)
            maturity_quadratic = pm.Uniform("maturity_quadratic", lower=-1.0, upper=1.0)
            params = pm.math.stack(
                [
                    level,
                    strike_slope,
                    maturity_slope,
                    sqrt_strike_slope,
                    maturity_quadratic,
                ]
            )
            mu = SyntheticSurfaceCalibrator.price_formula(strikes, maturities, params)
            pm.Normal("obs", mu=mu, sigma=0.01, observed=self.prices)
            trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                random_seed=random_seed,
                target_accept=target_accept,
                progressbar=False,
            )

        posterior = trace.posterior
        means = np.asarray(
            [posterior[var].mean().item() for var in _SYNTHETIC_PARAMETER_NAMES],
            dtype=float,
        )
        residuals = self.residuals(means)
        return CalibrationResult(
            parameters=means,
            success=True,
            status=1,
            message="PyMC posterior mean computed for synthetic fixture",
            residuals=np.asarray(residuals, dtype=float),
            cost=float(0.5 * np.dot(residuals, residuals)),
            optimality=float("nan"),
            jacobian_rank=0,
            jacobian_condition=np.inf,
            bounds=(
                np.full(means.shape, -np.inf, dtype=float),
                np.full(means.shape, np.inf, dtype=float),
            ),
            active_mask=np.zeros(means.shape, dtype=int),
            nfev=int(draws * chains),
            njev=None,
            method="pymc.sample",
            parameter_names=_SYNTHETIC_PARAMETER_NAMES,
        )


def _synthetic_market_data() -> tuple[pd.DataFrame, np.ndarray]:
    s = np.linspace(80, 120, 5)
    t = np.linspace(0.1, 1.0, 5)
    strikes_grid, maturities_grid = np.meshgrid(s, t)
    strikes = strikes_grid.ravel()
    maturities = maturities_grid.ravel()
    true_params = np.array([0.04, 1.0, 0.04, 0.3, -0.7])
    prices = SyntheticSurfaceCalibrator.price_formula(strikes, maturities, true_params)
    data = pd.DataFrame({"strike": strikes, "maturity": maturities, "price": prices})
    return data, true_params


def sample_calibration() -> CalibrationResult:
    """Run a toy calibration against explicitly synthetic market data."""

    data, true_params = _synthetic_market_data()
    calibrator = SyntheticSurfaceCalibrator(data)
    initial_guess = true_params + np.array([0.01, -0.1, 0.02, -0.05, 0.1])
    return calibrator.calibrate(initial_guess)


def sample_statsmodels_calibration() -> CalibrationResult:
    """Example calibration through the deprecated Statsmodels shim."""

    data, true_params = _synthetic_market_data()
    calibrator = StatsmodelsCalibrator(data)
    initial_guess = true_params + np.array([0.01, -0.1, 0.02, -0.05, 0.1])
    return calibrator.calibrate(initial_guess)


def sample_pymc_calibration() -> CalibrationResult:
    """Example Bayesian calibration returning synthetic posterior means."""

    data, _ = _synthetic_market_data()
    calibrator = PyMCCalibrator(data)
    return calibrator.calibrate(draws=500, chains=2, random_seed=123)
