"""Canonical public-synthetic FEM response for the OpenTURNS UQ pilot."""

from __future__ import annotations

from typing import Any

import numpy as np

from finite_element_options.core.market import Market
from finite_element_options.core.vanilla_bs import EuropeanOptionBs
from finite_element_options.examples.regime_switching_quanto import fem as regime_fem
from finite_element_options.examples.regime_switching_quanto.contracts import (
    ContractSpec,
    FEMGridSpec,
    TwoFactorRegimeModel,
)
from finite_element_options.examples.regime_switching_quanto.monte_carlo import (
    price_contract_monte_carlo,
)

from ..evidence_io import canonical_json_sha256
from .contracts import (
    COMPONENT_NAMES,
    QUANTILE_LEVELS,
    UQCalibration,
    UQPilotConfig,
    UncertaintyComponent,
)

SCHEMA_VERSION = "regime-switching-quanto-openturns-uq/v1"
SCOPE_STATEMENT = (
    "Public-synthetic one-regime fixed-FX quanto-call estimator/validation diagnostic. "
    "The FEM response is the existing regime-switching quanto solver with one regime, not an "
    "analytical surrogate. The combined distribution is not a risk-neutral payoff distribution."
)
QUANTLIB_ORACLE_ARTIFACT = "docs/evidence/regime_switching_quanto_quantlib_oracle_2026-09-04.json"
QUANTLIB_ORACLE_SHA256 = "ca2789e8f686a2f25b9abebc076f18ce7596673b038e52b681478cad22c4a056"
IMINUIT_ARTIFACT = "docs/evidence/regime_switching_quanto_iminuit_identifiability_2026-09-04.json"
IMINUIT_SHA256 = "6294b52e9d6aa26aeda39a1809486272223d41ecc7a00e42e670f5dcbba39a3b"

BASE_MATURITY = 458.0 / 365.0
BASELINE_SPOT = 100.0
BASELINE_SIGMA = 0.20
BASELINE_FULL_CORRELATION = 0.35
BASELINE_FX_VOL = 0.12
BASELINE_DOMESTIC_RATE = 0.035
BASELINE_FOREIGN_RATE = 0.015
BASELINE_DIVIDEND = 0.010
BASELINE_STRIKE = 105.0
BASELINE_FIXED_FX = 850.0
BASELINE_FX_SPOT = 1.0
BASELINE_GENERATOR = [[0.0]]
BASELINE_PROBABILITIES = [1.0]

FINE_GRID = FEMGridSpec((-1.6, 1.6), (-0.7, 0.7), nx=31, ny=7, time_steps=16)
COARSE_GRID = FEMGridSpec((-1.6, 1.6), (-0.7, 0.7), nx=21, ny=5, time_steps=10)
MC_CALIBRATION_SEED = 134_011
MC_CALIBRATION_PATHS = 4096
MC_CALIBRATION_STEPS_PER_YEAR = 32
ANALYTICAL_ORACLE_IDENTITY = "core.EuropeanOptionBs fixed-FX one-regime quanto reduction"
NUMERICAL_HALF_WIDTH_FORMULA = (
    "max(abs(fine_fem_price - analytical_oracle_price), "
    "abs(coarse_fem_price - analytical_oracle_price), "
    "1.5 * abs(fine_fem_price - coarse_fem_price), 1e-12)"
)
NUMERICAL_RESPONSE_ERROR_FORMULA = (
    "abs(fine_fem_price(input) - analytical_oracle_price(input)) * z_numerical"
)


def baseline_model(
    *, sigma: float = BASELINE_SIGMA, correlation_weight: float = 0.5
) -> TwoFactorRegimeModel:
    """Return the one-regime model at a supplied volatility and correlation weight."""

    weight = float(correlation_weight)
    sigma_value = float(sigma)
    if not np.isfinite(sigma_value) or not 0.01 <= sigma_value <= 1.50:
        raise ValueError("sigma must be finite and within [0.01, 1.50]")
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("correlation_weight must be finite and within [0, 1]")
    return TwoFactorRegimeModel(
        equity_vol=[sigma_value],
        fx_vol=[BASELINE_FX_VOL],
        correlation=[weight * BASELINE_FULL_CORRELATION],
        generator=BASELINE_GENERATOR,
        current_probabilities=BASELINE_PROBABILITIES,
        domestic_rate=BASELINE_DOMESTIC_RATE,
        foreign_rate=BASELINE_FOREIGN_RATE,
        dividend_yield=BASELINE_DIVIDEND,
        measure_note=(
            "Public-synthetic one-regime CLP-domestic Q fixed-FX quanto call; "
            "correlation is interpolated from no equity/FX coupling to the full Quanto endpoint."
        ),
    )


def baseline_contract() -> ContractSpec:
    """Return the fixed-FX quanto call payoff used by the pilot."""

    return ContractSpec(kind="quanto_call", strike=BASELINE_STRIKE, fixed_fx=BASELINE_FIXED_FX)


def grid_identity(grid: FEMGridSpec) -> dict[str, Any]:
    """Return a compact grid identity with nodes, steps, domain, and hash."""

    payload = grid.to_dict()
    payload["nodes"] = int(grid.nx * grid.ny)
    payload["triangular_cells"] = int(2 * (grid.nx - 1) * (grid.ny - 1))
    payload["element"] = "Lagrange-P1 triangular tensor grid"
    payload["theta_schedule"] = "four backward-Euler Rannacher half-steps then Crank-Nicolson"
    payload["hash"] = canonical_json_sha256(payload)
    return payload


def canonical_study_input(config: UQPilotConfig | None = None) -> dict[str, Any]:
    """Return the deterministic canonical study input whose hash binds the artifact."""

    controls = UQPilotConfig() if config is None else config
    model = baseline_model()
    contract = baseline_contract()
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": SCOPE_STATEMENT,
        "controls": controls.to_dict(),
        "baseline": {
            "maturity": BASE_MATURITY,
            "equity_spot": BASELINE_SPOT,
            "fx_spot": BASELINE_FX_SPOT,
            "spot_data_relative_range": 0.05,
            "equity_volatility_relative_range": 0.15,
            "model_form_endpoints": [
                "zero_correlation_independent_equity_fx_generator",
                "full_quanto_correlation_generator",
            ],
            "normalized_zero_input_correlation_weight": 0.5,
            "baseline_correlation": 0.5 * BASELINE_FULL_CORRELATION,
            "model": model.to_dict(),
            "payoff": contract.to_dict(),
        },
        "fine_grid": grid_identity(FINE_GRID),
        "coarse_grid": grid_identity(COARSE_GRID),
        "mc_calibration": {
            "seed": MC_CALIBRATION_SEED,
            "paths": MC_CALIBRATION_PATHS,
            "steps_per_year": MC_CALIBRATION_STEPS_PER_YEAR,
        },
        "numerical_calibration": {
            "oracle_identity": ANALYTICAL_ORACLE_IDENTITY,
            "baseline_half_width_formula": NUMERICAL_HALF_WIDTH_FORMULA,
            "response_error_formula": NUMERICAL_RESPONSE_ERROR_FORMULA,
        },
        "component_names": COMPONENT_NAMES,
        "predecessor_source": {
            "artifact": QUANTLIB_ORACLE_ARTIFACT,
            "sha256": QUANTLIB_ORACLE_SHA256,
            "case_id": "quanto_positive_correlation",
            "use": "baseline public-synthetic quanto parameters and payoff conventions",
        },
    }


def canonical_uq_input_hash(config: UQPilotConfig | None = None) -> str:
    """Return the SHA-256 hash of the canonical study input."""

    return canonical_json_sha256(canonical_study_input(config))


def analytical_price(*, spot: float, sigma: float, correlation_weight: float) -> float:
    """Return the exact one-regime fixed-FX quanto reduction at supplied inputs."""

    model = baseline_model(sigma=sigma, correlation_weight=correlation_weight)
    spot_value = float(spot)
    if not np.isfinite(spot_value) or spot_value <= 0.0:
        raise ValueError("spot must be finite and positive")
    correlation = float(model.correlation[0])
    q_eff = (
        BASELINE_DIVIDEND
        + BASELINE_DOMESTIC_RATE
        - BASELINE_FOREIGN_RATE
        + correlation * float(sigma) * BASELINE_FX_VOL
    )
    option = EuropeanOptionBs(
        k=BASELINE_STRIKE,
        q=q_eff,
        mkt=Market(r=BASELINE_DOMESTIC_RATE),
    )
    call = option.call_from_volatility(BASE_MATURITY, spot_value, float(sigma))
    return float(call) * BASELINE_FIXED_FX


def calibrate_scales() -> UQCalibration:
    """Derive numerical and seeded-MC additive scales before propagation."""

    model = baseline_model()
    contract = baseline_contract()
    fine = regime_fem.price_contract_fem(
        model,
        contract,
        maturity=BASE_MATURITY,
        equity_spot=BASELINE_SPOT,
        fx_spot=BASELINE_FX_SPOT,
        grid=FINE_GRID,
    )
    coarse = regime_fem.price_contract_fem(
        model,
        contract,
        maturity=BASE_MATURITY,
        equity_spot=BASELINE_SPOT,
        fx_spot=BASELINE_FX_SPOT,
        grid=COARSE_GRID,
    )
    mc = price_contract_monte_carlo(
        model,
        contract,
        maturity=BASE_MATURITY,
        equity_spot=BASELINE_SPOT,
        fx_spot=BASELINE_FX_SPOT,
        paths=MC_CALIBRATION_PATHS,
        seed=MC_CALIBRATION_SEED,
        steps_per_year=MC_CALIBRATION_STEPS_PER_YEAR,
    )
    oracle_price = analytical_price(
        spot=BASELINE_SPOT,
        sigma=BASELINE_SIGMA,
        correlation_weight=0.5,
    )
    discrepancy = abs(fine.mixture_price - coarse.mixture_price)
    fine_oracle_error = abs(fine.mixture_price - oracle_price)
    coarse_oracle_error = abs(coarse.mixture_price - oracle_price)
    half_width = max(fine_oracle_error, coarse_oracle_error, 1.5 * discrepancy, 1.0e-12)
    fine_identity = grid_identity(FINE_GRID)
    coarse_identity = grid_identity(COARSE_GRID)
    model_hash = canonical_json_sha256(model.to_dict())
    payoff_hash = canonical_json_sha256(contract.to_dict())
    oracle_payload = {
        "identity": ANALYTICAL_ORACLE_IDENTITY,
        "maturity": BASE_MATURITY,
        "equity_spot": BASELINE_SPOT,
        "equity_volatility": BASELINE_SIGMA,
        "correlation": 0.5 * BASELINE_FULL_CORRELATION,
        "fx_volatility": BASELINE_FX_VOL,
        "model_hash": model_hash,
        "payoff_hash": payoff_hash,
        "price": oracle_price,
    }
    return UQCalibration(
        baseline_price_fine=float(fine.mixture_price),
        baseline_price_coarse=float(coarse.mixture_price),
        baseline_price_oracle=oracle_price,
        fine_oracle_abs_error=float(fine_oracle_error),
        coarse_oracle_abs_error=float(coarse_oracle_error),
        oracle_identity=ANALYTICAL_ORACLE_IDENTITY,
        numerical_half_width=float(half_width),
        numerical_formula=NUMERICAL_HALF_WIDTH_FORMULA,
        mc_price=float(mc.price),
        mc_standard_error=float(mc.standard_error),
        mc_seed=MC_CALIBRATION_SEED,
        mc_paths=MC_CALIBRATION_PATHS,
        mc_steps=int(mc.steps),
        mc_steps_per_year=MC_CALIBRATION_STEPS_PER_YEAR,
        fine_grid=fine_identity,
        coarse_grid=coarse_identity,
        fine_grid_hash=str(fine_identity["hash"]),
        coarse_grid_hash=str(coarse_identity["hash"]),
        baseline_model_hash=model_hash,
        payoff_hash=payoff_hash,
        oracle_hash=canonical_json_sha256(oracle_payload),
    )


def build_components(
    calibration: UQCalibration, config: UQPilotConfig | None = None
) -> tuple[UncertaintyComponent, ...]:
    """Build the five named OpenTURNS-independent uncertainty contracts."""

    input_hash = canonical_uq_input_hash(config)
    return (
        UncertaintyComponent(
            name="data",
            distribution="Uniform(-1, 1) normalized; spot = 100 * (1 + 0.05*z_data)",
            scale_or_range={"spot_min": 95.0, "spot_baseline": 100.0, "spot_max": 105.0},
            units="equity spot currency units",
            role="fem_perturbation",
            source_identity="public-synthetic spot/input-state band",
            source_hash=input_hash,
            perturbs_fem_model=True,
            additive_validation_estimator_error=False,
            description="Spot/input-state uncertainty perturbs the FEM valuation point only.",
        ),
        UncertaintyComponent(
            name="parameter",
            distribution="Uniform(-1, 1) normalized; sigmaS = 0.20 * (1 + 0.15*z_parameter)",
            scale_or_range={"sigma_min": 0.17, "sigma_baseline": 0.20, "sigma_max": 0.23},
            units="annualized equity volatility",
            role="fem_perturbation",
            source_identity="public-synthetic equity-volatility-only parameter band",
            source_hash=input_hash,
            perturbs_fem_model=True,
            additive_validation_estimator_error=False,
            description="Parameter uncertainty is equity volatility only; numerical error is excluded.",
        ),
        UncertaintyComponent(
            name="model_form",
            distribution="Uniform(-1, 1) normalized; weight=(z_model_form+1)/2",
            scale_or_range={
                "rho_zero_coupling": 0.0,
                "rho_full_quanto": BASELINE_FULL_CORRELATION,
                "baseline_weight": 0.5,
            },
            units="dimensionless correlation inclusion weight",
            role="fem_perturbation",
            source_identity="zero-correlation to full-quanto-correlation interpolation",
            source_hash=input_hash,
            perturbs_fem_model=True,
            additive_validation_estimator_error=False,
            description=(
                "Model form interpolates the correlation/quanto coupling from the independent "
                "equity-FX generator endpoint to the full fixed-FX quanto endpoint."
            ),
        ),
        UncertaintyComponent(
            name="numerical",
            distribution=(
                "Uniform(-1, 1) normalized; additive error = "
                "abs(fine_fem(input)-analytical_oracle(input))*z_numerical"
            ),
            scale_or_range={
                "baseline_half_width": calibration.numerical_half_width,
                "mode": "input_dependent_exact_oracle_absolute_error",
            },
            units="price currency units",
            role="additive_validation_estimator_error",
            source_identity="input-local analytical oracle discrepancy plus baseline coarse/fine evidence",
            source_hash=canonical_json_sha256(
                {
                    "fine_grid_hash": calibration.fine_grid_hash,
                    "coarse_grid_hash": calibration.coarse_grid_hash,
                    "baseline_price_fine": calibration.baseline_price_fine,
                    "baseline_price_coarse": calibration.baseline_price_coarse,
                    "baseline_price_oracle": calibration.baseline_price_oracle,
                    "fine_oracle_abs_error": calibration.fine_oracle_abs_error,
                    "coarse_oracle_abs_error": calibration.coarse_oracle_abs_error,
                    "oracle_identity": calibration.oracle_identity,
                    "oracle_hash": calibration.oracle_hash,
                    "numerical_half_width": calibration.numerical_half_width,
                    "formula": calibration.numerical_formula,
                    "response_error_formula": NUMERICAL_RESPONSE_ERROR_FORMULA,
                }
            ),
            perturbs_fem_model=False,
            additive_validation_estimator_error=True,
            description=(
                "Input-dependent additive FEM discretization error; not included in parameter uncertainty."
            ),
        ),
        UncertaintyComponent(
            name="monte_carlo",
            distribution="Normal(0, 1) normalized; additive error = baseline_MC_standard_error*z_monte_carlo",
            scale_or_range={"standard_error": calibration.mc_standard_error},
            units="price currency units",
            role="additive_validation_estimator_error",
            source_identity="seeded direct MC validation estimator standard error",
            source_hash=canonical_json_sha256(
                {
                    "seed": calibration.mc_seed,
                    "paths": calibration.mc_paths,
                    "steps": calibration.mc_steps,
                    "steps_per_year": calibration.mc_steps_per_year,
                    "maturity": BASE_MATURITY,
                    "equity_spot": BASELINE_SPOT,
                    "fx_spot": BASELINE_FX_SPOT,
                    "model_hash": calibration.baseline_model_hash,
                    "payoff_hash": calibration.payoff_hash,
                    "mc_price": calibration.mc_price,
                    "mc_standard_error": calibration.mc_standard_error,
                }
            ),
            perturbs_fem_model=False,
            additive_validation_estimator_error=True,
            description=(
                "Monte Carlo component is validation-estimator uncertainty, not intrinsic fair-value uncertainty."
            ),
        ),
    )


def map_normalized_inputs(z: np.ndarray, calibration: UQCalibration) -> dict[str, float]:
    """Map five lawful normalized coordinates to FEM inputs and error coordinates."""

    values = np.asarray(z, dtype=float)
    if values.shape != (5,):
        raise ValueError("expected exactly five normalized inputs with shape (5,)")
    if not np.all(np.isfinite(values)):
        raise ValueError("normalized inputs must all be finite")
    uniform = values[:4]
    if not np.all((-1.0 <= uniform) & (uniform <= 1.0)):
        raise ValueError("first four normalized inputs must be within [-1, 1]")
    spot = BASELINE_SPOT * (1.0 + 0.05 * float(values[0]))
    sigma = BASELINE_SIGMA * (1.0 + 0.15 * float(values[1]))
    weight = 0.5 * (float(values[2]) + 1.0)
    numerical_coordinate = float(values[3])
    mc_error = calibration.mc_standard_error * float(values[4])
    return {
        "spot": spot,
        "sigma": sigma,
        "correlation_weight": weight,
        "correlation": BASELINE_FULL_CORRELATION * weight,
        "numerical_coordinate": numerical_coordinate,
        "monte_carlo_error": mc_error,
    }


def evaluate_response(z: np.ndarray, calibration: UQCalibration) -> float:
    """Evaluate the FEM solver with input-dependent numerical and additive MC errors."""

    mapped = map_normalized_inputs(z, calibration)
    model = baseline_model(sigma=mapped["sigma"], correlation_weight=mapped["correlation_weight"])
    result = regime_fem.price_contract_fem(
        model,
        baseline_contract(),
        maturity=BASE_MATURITY,
        equity_spot=mapped["spot"],
        fx_spot=BASELINE_FX_SPOT,
        grid=FINE_GRID,
    )
    oracle_price = analytical_price(
        spot=mapped["spot"],
        sigma=mapped["sigma"],
        correlation_weight=mapped["correlation_weight"],
    )
    numerical_half_width = abs(float(result.mixture_price) - oracle_price)
    numerical_error = numerical_half_width * mapped["numerical_coordinate"]
    return float(result.mixture_price + numerical_error + mapped["monte_carlo_error"])


def summarize_prices(values: np.ndarray) -> dict[str, Any]:
    """Summarize finite propagated price samples."""

    prices = np.asarray(values, dtype=float)
    finite = prices[np.isfinite(prices)]
    quantiles = np.quantile(finite, QUANTILE_LEVELS)
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=1)),
        "quantiles": {
            str(level): float(value)
            for level, value in zip(QUANTILE_LEVELS, quantiles, strict=True)
        },
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def numpy_direct_sample(seed: int, size: int) -> np.ndarray:
    """Draw the five independent marginals using NumPy for direct-reference parity."""

    rng = np.random.default_rng(seed)
    sample = np.empty((size, 5), dtype=float)
    sample[:, :4] = rng.uniform(-1.0, 1.0, size=(size, 4))
    sample[:, 4] = rng.standard_normal(size)
    return sample
