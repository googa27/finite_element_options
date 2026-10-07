"""Unchanged public-synthetic inputs for the OpenTURNS UQ pilot."""

from finite_element_options.examples.regime_switching_quanto.contracts import (
    FEMGridSpec,
)

from typing import Any
from finite_element_options.examples.regime_switching_quanto.contracts import (
    ContractSpec,
    TwoFactorRegimeModel,
)
from ..evidence_io import canonical_json_sha256
from .contracts import COMPONENT_NAMES, UQPilotConfig


SCHEMA_VERSION = "regime-switching-quanto-openturns-uq/v1"
SCOPE_STATEMENT = (
    "Public-synthetic one-regime fixed-FX quanto-call estimator/validation diagnostic. "
    "The FEM response is the existing regime-switching quanto solver with one regime, not an "
    "analytical surrogate. The combined distribution is not a risk-neutral payoff distribution."
)
QUANTLIB_ORACLE_ARTIFACT = (
    "docs/evidence/regime_switching_quanto_quantlib_oracle_2026-09-04.json"
)
QUANTLIB_ORACLE_SHA256 = (
    "ca2789e8f686a2f25b9abebc076f18ce7596673b038e52b681478cad22c4a056"
)
IMINUIT_ARTIFACT = (
    "docs/evidence/regime_switching_quanto_iminuit_identifiability_2026-09-04.json"
)
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
ANALYTICAL_ORACLE_IDENTITY = (
    "core.EuropeanOptionBs fixed-FX one-regime quanto reduction"
)
DOMAIN_ERROR_GRID = {
    "spot_levels": 11,
    "sigma_levels": 5,
    "correlation_weight_levels": 5,
}
DOMAIN_ERROR_SAFETY_FACTOR = 1.10
NUMERICAL_HALF_WIDTH_FORMULA = (
    "ceil_10sig(max(abs(fine_fem_price - analytical_oracle_price), "
    "abs(coarse_fem_price - analytical_oracle_price), "
    "1.5 * abs(fine_fem_price - coarse_fem_price), "
    "1.10 * max_domain_grid_fine_oracle_error, 1e-12))"
)


def grid_identity(grid: FEMGridSpec) -> dict[str, Any]:
    """Return a compact grid identity with nodes, steps, domain, and hash."""

    payload = grid.to_dict()
    payload["nodes"] = int(grid.nx * grid.ny)
    payload["triangular_cells"] = int(2 * (grid.nx - 1) * (grid.ny - 1))
    payload["element"] = "Lagrange-P1 triangular tensor grid"
    payload["theta_schedule"] = (
        "four backward-Euler Rannacher half-steps then Crank-Nicolson"
    )
    payload["hash"] = canonical_json_sha256(payload)
    return payload


def build_study_input(
    controls: UQPilotConfig, model: TwoFactorRegimeModel, contract: ContractSpec
) -> dict[str, Any]:
    """Serialize the unchanged model/payoff/controls into canonical study inputs."""

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
            "half_width_formula": NUMERICAL_HALF_WIDTH_FORMULA,
            "domain_error_grid": DOMAIN_ERROR_GRID,
            "domain_error_safety_factor": DOMAIN_ERROR_SAFETY_FACTOR,
        },
        "component_names": COMPONENT_NAMES,
        "predecessor_source": {
            "artifact": QUANTLIB_ORACLE_ARTIFACT,
            "sha256": QUANTLIB_ORACLE_SHA256,
            "case_id": "quanto_positive_correlation",
            "use": "baseline public-synthetic quanto parameters and payoff conventions",
        },
    }
