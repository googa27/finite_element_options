"""Public-synthetic Pinares fixed-price proxy FEM weak-form evidence.

Pinares owns the family real-estate deal semantics. This module validates only a
small public-synthetic fixed-price purchase-option proxy that can be represented
as a one-dimensional Black--Scholes-style European call under the project-wide
``Q*`` proxy measure. Full family-contract, ROFR, legal coordination,
mortality-table, liquidity/default, tax and market-rent routes remain
fail-closed at the capability-screening layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np
import scipy.stats as spst  # type: ignore[import-untyped]

from finite_element_options.contracts import DEFAULT_FEM_CAPABILITY_MANIFEST
from finite_element_options.core.dynamics_black_scholes import (
    DynamicsParametersBlackScholes,
)
from finite_element_options.core.market import Market
from finite_element_options.core.vanilla_bs import EuropeanOptionBs
from finite_element_options.space.boundary import DirichletBC
from finite_element_options.space.mesh import create_mesh
from finite_element_options.space.solver import SpaceSolver
from finite_element_options.time_integration.stepper import ThetaScheme

PINARES_FEM_FIXED_PRICE_PROXY_BENCHMARK_ID = "PINARES-FEM-FIXED-PRICE-PROXY-V0"
PINARES_QPS_CONTRACT_BENCHMARK_ID = "PINARES-QPS-FIXED-PRICE-PROXY-V0"
PINARES_FEM_FAIL_CLOSED_BENCHMARK_ID = "PINARES-FEM-FAIL-CLOSED-V0"
PINARES_FEM_FIXED_PRICE_PROXY_BENCHMARK_IDS = (
    PINARES_FEM_FIXED_PRICE_PROXY_BENCHMARK_ID,
    PINARES_QPS_CONTRACT_BENCHMARK_ID,
)
PINARES_FEM_FIXED_PRICE_PROXY_PROBLEM_ID = "pinares.fixed_price_option_proxy.v1"
PINARES_FEM_FIXED_PRICE_PROXY_PROBLEM_HASH = "publicsyntheticpinares001"
PINARES_FEM_FIXED_PRICE_PROXY_FIXTURE_ID = "public-synthetic.pinares-fem-fixed-price-proxy.v1"
PINARES_FEM_FIXED_PRICE_PROXY_ROUTE_ID = "fem.pinares_fixed_price_proxy.weak_form_p2_theta"
PINARES_FEM_FIXED_PRICE_PROXY_SCHEMA_VERSION = "finite-element-pinares-fixed-price-proxy/v0"
PINARES_FEM_PROXY_REFINEMENT_LEVELS = (5, 6, 7)
PINARES_FEM_PROXY_TIME_STEPS = 160
PINARES_FEM_PROXY_FIXTURE_ROOT = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "fem_pinares_fixed_price_proxy_v1"
)
PINARES_FEM_PROXY_PROBLEM_SPEC_PATH = PINARES_FEM_PROXY_FIXTURE_ROOT / "problem_spec.json"
PINARES_FEM_PROXY_RESULT_EXPORT_PATH = PINARES_FEM_PROXY_FIXTURE_ROOT / "result_export.json"
PINARES_FEM_PROXY_UNSUPPORTED_SPEC_PATH = (
    PINARES_FEM_PROXY_FIXTURE_ROOT / "unsupported_full_deal_problem_spec.json"
)
PINARES_QPS_FIXTURE_PATH = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "fixtures"
    / "quant_problem_specs"
    / "pinares_fixed_price_proxy.json"
)


@dataclass(frozen=True)
class PinaresFixedPriceProxyCase:
    """Public-synthetic Pinares fixed-price weak-form benchmark inputs.

    ``survival_probability`` scales a terminal fixed-price call payoff. It is not
    a full mortality table, full deal valuation, ROFR valuation, legal/tax
    conclusion, or production Pinares scenario.
    """

    fixture_id: str = PINARES_FEM_FIXED_PRICE_PROXY_FIXTURE_ID
    problem_id: str = PINARES_FEM_FIXED_PRICE_PROXY_PROBLEM_ID
    problem_hash: str = PINARES_FEM_FIXED_PRICE_PROXY_PROBLEM_HASH
    route_id: str = PINARES_FEM_FIXED_PRICE_PROXY_ROUTE_ID
    backend_id: str = DEFAULT_FEM_CAPABILITY_MANIFEST.backend_id
    code_version: str = "local-checkout"
    spot_uf: float = 6000.0
    strike_uf: float = 6200.0
    risk_free_rate: float = 0.015
    volatility: float = 0.12
    maturity_years: float = 1.0
    survival_probability: float = 0.97
    s_max_uf: float = 12000.0
    valuation_date: str = "2026-06-30"
    maturity_date: str = "2027-06-30"
    refinement_levels: tuple[int, ...] = PINARES_FEM_PROXY_REFINEMENT_LEVELS
    time_steps: int = PINARES_FEM_PROXY_TIME_STEPS
    price_abs_tolerance_uf: float = 1.0
    delta_abs_tolerance: float = 1.0e-3
    gamma_abs_tolerance: float = 5.0e-6
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate public-synthetic scalar inputs before solver construction."""

        positive_fields = {
            "spot_uf": self.spot_uf,
            "strike_uf": self.strike_uf,
            "volatility": self.volatility,
            "maturity_years": self.maturity_years,
            "s_max_uf": self.s_max_uf,
            "price_abs_tolerance_uf": self.price_abs_tolerance_uf,
            "delta_abs_tolerance": self.delta_abs_tolerance,
            "gamma_abs_tolerance": self.gamma_abs_tolerance,
        }
        for field_name, value in positive_fields.items():
            if value <= 0.0:
                msg = f"{field_name} must be positive"
                raise ValueError(msg)
        if self.s_max_uf <= self.spot_uf:
            msg = "s_max_uf must be greater than spot_uf"
            raise ValueError(msg)
        if not 0.0 <= self.survival_probability <= 1.0:
            msg = "survival_probability must be in [0, 1]"
            raise ValueError(msg)
        if not self.refinement_levels:
            msg = "at least one FEM refinement level is required"
            raise ValueError(msg)
        if any(level < 1 for level in self.refinement_levels):
            msg = "all FEM refinement levels must be positive"
            raise ValueError(msg)
        if self.time_steps <= 0:
            msg = "time_steps must be positive"
            raise ValueError(msg)

    @property
    def spot_ratio(self) -> float:
        """Return normalized spot in strike units."""

        return self.spot_uf / self.strike_uf

    @property
    def domain_max_ratio(self) -> float:
        """Return normalized right boundary in strike units."""

        return self.s_max_uf / self.strike_uf

    def normalized_units(self) -> dict[str, str]:
        """Return explicit Pinares proxy units."""

        return {
            "S": "UF",
            "underlying": "UF",
            "value": "UF",
            "rate": "1/year",
            "time": "year",
        }


@dataclass(frozen=True)
class PinaresFEMProxyConvergenceRow:
    """One refinement row in the Pinares public-synthetic FEM proxy report."""

    refinement_level: int
    time_steps: int
    degrees_of_freedom: int
    observed_price_uf: float
    expected_price_uf: float
    absolute_error_uf: float
    relative_error: float
    observed_delta: float
    expected_delta: float
    delta_absolute_error: float
    observed_gamma: float
    expected_gamma: float
    gamma_absolute_error: float

    def to_public_dict(self) -> dict[str, float | int]:
        """Return a JSON-safe convergence row."""

        return {
            "refinement_level": self.refinement_level,
            "time_steps": self.time_steps,
            "degrees_of_freedom": self.degrees_of_freedom,
            "observed_price_uf": self.observed_price_uf,
            "expected_price_uf": self.expected_price_uf,
            "absolute_error_uf": self.absolute_error_uf,
            "relative_error": self.relative_error,
            "observed_delta": self.observed_delta,
            "expected_delta": self.expected_delta,
            "delta_absolute_error": self.delta_absolute_error,
            "observed_gamma": self.observed_gamma,
            "expected_gamma": self.expected_gamma,
            "gamma_absolute_error": self.gamma_absolute_error,
        }


@dataclass(frozen=True)
class PinaresFEMProxyReport:
    """Scaled Pinares fixed-price weak-form proxy result with evidence payloads."""

    case: PinaresFixedPriceProxyCase
    rows: tuple[PinaresFEMProxyConvergenceRow, ...]
    expected_price_uf: float
    observed_price_uf: float
    price_absolute_error_uf: float
    price_relative_error: float
    expected_delta: float
    observed_delta: float
    delta_absolute_error: float
    expected_gamma: float
    observed_gamma: float
    gamma_absolute_error: float
    no_arbitrage: dict[str, Any]
    config_hash: str

    @property
    def converged(self) -> bool:
        """Whether all declared Pinares proxy tolerances pass."""

        return (
            self.price_absolute_error_uf <= self.case.price_abs_tolerance_uf
            and self.delta_absolute_error <= self.case.delta_abs_tolerance
            and self.gamma_absolute_error <= self.case.gamma_abs_tolerance
            and all(bool(value) for key, value in self.no_arbitrage.items() if key.endswith("_ok"))
        )

    def to_public_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable public-synthetic evidence payload."""

        return _stable_public_payload(
            {
                "schema_version": PINARES_FEM_FIXED_PRICE_PROXY_SCHEMA_VERSION,
                "benchmark_id": PINARES_FEM_FIXED_PRICE_PROXY_BENCHMARK_ID,
                "fixture_id": self.case.fixture_id,
                "problem_id": self.case.problem_id,
                "problem_hash": self.case.problem_hash,
                "privacy_class": "public_synthetic",
                "route_id": self.case.route_id,
                "backend_id": self.case.backend_id,
                "config_hash": self.config_hash,
                "converged": self.converged,
                "problem_spec": public_pinares_fixed_price_problem_spec(case=self.case),
                "weak_form": _weak_form_metadata(),
                "mesh_metadata": _mesh_metadata(self.case),
                "time_metadata": _time_metadata(self.case),
                "boundary_conditions": _boundary_metadata(self.case),
                "solution": {
                    "spot_uf": self.case.spot_uf,
                    "price_uf": self.observed_price_uf,
                    "delta": self.observed_delta,
                    "gamma": self.observed_gamma,
                },
                "reference": {
                    "oracle_price_uf": self.expected_price_uf,
                    "delta": self.expected_delta,
                    "gamma": self.expected_gamma,
                    "survival_probability": self.case.survival_probability,
                },
                "errors": {
                    "price_abs_uf": self.price_absolute_error_uf,
                    "price_rel": self.price_relative_error,
                    "delta_abs": self.delta_absolute_error,
                    "gamma_abs": self.gamma_absolute_error,
                },
                "tolerances": {
                    "price_abs_uf": self.case.price_abs_tolerance_uf,
                    "delta_abs": self.case.delta_abs_tolerance,
                    "gamma_abs": self.case.gamma_abs_tolerance,
                },
                "no_arbitrage": self.no_arbitrage,
                "convergence_rows": [row.to_public_dict() for row in self.rows],
                "unsupported_scope": {
                    "rofr": "unsupported; right of first refusal is not a vanilla call",
                    "full_family_contract": "unsupported; requires Pinares contract valuation semantics and PDP inputs",
                    "legal_tax_conclusion": "unsupported; FEM backend does not provide legal/tax advice",
                },
            }
        )

    def export_payload(self) -> dict[str, Any]:
        """Return a compact result-export shape suitable for public fixtures."""

        return _stable_public_payload(
            {
                "format_version": "fem-pinares-fixed-price-proxy-result-v1",
                "benchmark_id": PINARES_FEM_FIXED_PRICE_PROXY_BENCHMARK_ID,
                "problem_id": self.case.problem_id,
                "problem_hash": self.case.problem_hash,
                "privacy_class": "public_synthetic",
                "config_hash": self.config_hash,
                "route_id": self.case.route_id,
                "weak_form": _weak_form_metadata(),
                "mesh_metadata": _mesh_metadata(self.case),
                "time_metadata": _time_metadata(self.case),
                "boundary_conditions": _boundary_metadata(self.case),
                "rows": [row.to_public_dict() for row in self.rows],
                "summary": {
                    "observed_price_uf": self.observed_price_uf,
                    "expected_price_uf": self.expected_price_uf,
                    "price_absolute_error_uf": self.price_absolute_error_uf,
                    "price_relative_error": self.price_relative_error,
                    "price_tolerance_absolute_uf": self.case.price_abs_tolerance_uf,
                    "observed_delta": self.observed_delta,
                    "expected_delta": self.expected_delta,
                    "delta_absolute_error": self.delta_absolute_error,
                    "delta_tolerance_absolute": self.case.delta_abs_tolerance,
                    "observed_gamma": self.observed_gamma,
                    "expected_gamma": self.expected_gamma,
                    "gamma_absolute_error": self.gamma_absolute_error,
                    "gamma_tolerance_absolute": self.case.gamma_abs_tolerance,
                },
                "no_arbitrage": self.no_arbitrage,
            }
        )


def public_pinares_fixed_price_problem_spec(
    *, case: PinaresFixedPriceProxyCase | None = None
) -> dict[str, Any]:
    """Return the canonical public-synthetic Pinares fixed-price QuantProblemSpec."""

    case = case or PinaresFixedPriceProxyCase()
    coefficient_terms = [
        {
            "name": "drift",
            "operator": "S * dV/dS",
            "coefficient": case.risk_free_rate,
            "expression": "r S ∂V/∂S",
        },
        {
            "name": "diffusion",
            "operator": "S^2 * d²V/dS²",
            "coefficient": 0.5 * case.volatility**2,
            "variance": case.volatility**2,
            "expression": "0.5 σ² S² ∂²V/∂S²",
        },
        {
            "name": "reaction",
            "operator": "V",
            "coefficient": -case.risk_free_rate,
            "expression": "-r V",
        },
    ]
    resource_controls = {
        "refinement_levels": list(case.refinement_levels),
        "max_refinement_level": max(case.refinement_levels),
        "time_steps": case.time_steps,
        "deterministic": "true",
    }
    error_budgets = {
        "price_abs_uf": case.price_abs_tolerance_uf,
        "delta_abs": case.delta_abs_tolerance,
        "gamma_abs": case.gamma_abs_tolerance,
    }
    payload = {
        "schema_version": "quant-problem-spec/v0",
        "privacy_class": "public_synthetic",
        "artifact_manifest": {
            "schema_version": "artifact-manifest/v0",
            "manifest_id": "pinares-fem-fixed-price-proxy-public-synthetic-v1",
            "fixture_id": case.fixture_id,
            "benchmark_ids": list(PINARES_FEM_FIXED_PRICE_PROXY_BENCHMARK_IDS),
            "issue_refs": ["googa27/finite_element_options#78"],
        },
        "problem_id": case.problem_id,
        "problem_hash": case.problem_hash,
        "valuation_context": {
            "measure": "Q*",
            "numeraire": "UF_money_market_account_proxy",
            "valuation_date": case.valuation_date,
            "maturity_date": case.maturity_date,
            "time_domain": f"[0, {case.maturity_years:g}]",
            "units": case.normalized_units(),
            "privacy_tier": "public_synthetic",
        },
        "mathematical_problem": {
            "dimension": 1,
            "state_variables": [
                {
                    "name": "S",
                    "role": "underlying",
                    "unit": "UF",
                    "description": "public synthetic property-value proxy",
                }
            ],
            "measure_id": "Q*",
            "numeraire_id": "UF_money_market_account_proxy",
            "pde_terms": [term["name"] for term in coefficient_terms],
            "pde_operator_terms": coefficient_terms,
            "pde_coefficients": {
                "risk_free_rate": case.risk_free_rate,
                "volatility": case.volatility,
                "terms": coefficient_terms,
            },
            "boundary_conditions": {
                "S=0": "dirichlet",
                "S=S_max": "linear_growth",
            },
            "exercise_style": "european",
            "formulations": [
                {
                    "formulation_id": "expectation_fixed_price_proxy",
                    "kind": "conditional_expectation",
                },
                {"formulation_id": "pde_fixed_price_proxy", "kind": "generator_pde"},
                {"formulation_id": "weak_form_fixed_price_proxy", "kind": "weak_form"},
            ],
            "weak_form": _weak_form_metadata(),
            "terminal_payoff": {
                "payoff_id": "mortality_scaled_fixed_price_call_payoff",
                "expression": "p_survival * max(S - K, 0)",
                "timing": "terminal",
                "units": "UF",
                "parameters": {
                    "K_uf": case.strike_uf,
                    "p_survival": case.survival_probability,
                },
            },
            "has_obstacle": False,
            "has_jumps": False,
            "has_hjb_control": False,
            "unsupported_full_deal_terms": [
                "rofr",
                "legal_coordination",
                "tax_transfer_analysis",
                "liquidity_default",
                "market_rent_alternative",
            ],
            "requested_outputs": ["value", "delta", "gamma"],
            "units": case.normalized_units(),
        },
        "solver_plan": {
            "backend_id": case.backend_id,
            "mesh_family": "line_uniform",
            "element_family": "lagrange_p2",
            "method_id": "pinares-fem-fixed-price-proxy-weak-form-p2-theta",
            "method_kind": "finite_element",
            "formulation_kind": "weak_form",
            "stability_controls": ["theta"],
            "linear_solver": "scipy_direct",
            "requested_outputs": ["value", "delta", "gamma"],
            "time_controls": {"theta": 0.5},
            "resource_controls": resource_controls,
            "error_budgets": error_budgets,
        },
        "financial_graph": {
            "instrument": {
                "kind": "mortality_scaled_fixed_price_purchase_option_proxy",
                "unit": "UF",
                "strike_uf": case.strike_uf,
                "survival_probability": case.survival_probability,
            },
            "valuation_graph": {
                "solver_hints": {"benchmark_ids": list(PINARES_FEM_FIXED_PRICE_PROXY_BENCHMARK_IDS)}
            },
        },
        "result_bundle": {
            "benchmark_ids": list(PINARES_FEM_FIXED_PRICE_PROXY_BENCHMARK_IDS),
            "references": [
                "Pinares THEORY.md: fixed-price option is separate from ROFR/full contract"
            ],
        },
        "benchmark_ids": list(PINARES_FEM_FIXED_PRICE_PROXY_BENCHMARK_IDS),
    }
    payload["contract_id"] = build_pinares_fem_proxy_hash(payload)
    return payload


def public_pinares_full_deal_unsupported_problem_spec() -> dict[str, Any]:
    """Return a public-synthetic full-deal request that FEM must reject."""

    payload = public_pinares_fixed_price_problem_spec()
    payload["problem_id"] = "pinares.full_family_contract.unsupported.v1"
    payload["problem_hash"] = "publicsyntheticpinaresfullunsupported001"
    payload["benchmark_ids"] = [PINARES_FEM_FAIL_CLOSED_BENCHMARK_ID]
    payload["artifact_manifest"] = {
        **payload["artifact_manifest"],
        "benchmark_ids": [PINARES_FEM_FAIL_CLOSED_BENCHMARK_ID],
    }
    payload["result_bundle"] = {"benchmark_ids": [PINARES_FEM_FAIL_CLOSED_BENCHMARK_ID]}
    payload["solver_plan"] = {
        **payload["solver_plan"],
        "requested_outputs": ["value", "delta", "gamma", "legal_tax_conclusion"],
    }
    payload["financial_graph"] = {
        "instrument": {
            "kind": "family_real_estate_use_right_full_contract",
            "unit": "UF",
            "contains_rofr": True,
            "legal_coordination": "proposal_assumption",
        }
    }
    payload["mathematical_problem"] = {
        "dimension": 4,
        "state_variables": [
            "property_value",
            "father_alive",
            "liquidity_state",
            "legal_coordination_state",
        ],
        "measure_id": "P/Q*_mixed_unsupported",
        "numeraire_id": "UF_money_market_account_proxy",
        "pde_terms": [
            "drift",
            "diffusion",
            "reaction",
            "hazard_killing",
            "liquidity_jump",
            "hjb_control",
            "obstacle",
        ],
        "boundary_conditions": {
            "sale_or_stress": "absorbing",
            "legal_coordination": "legal_coordination_constraint",
            "rofr_exercise": "free_boundary_with_linear_continuation",
        },
        "exercise_style": "rofr_full_family_contract",
        "requested_outputs": ["value", "delta", "gamma", "legal_tax_conclusion"],
    }
    payload.pop("contract_id", None)
    payload["contract_id"] = build_pinares_fem_proxy_hash(payload)
    return payload


def build_pinares_fem_proxy_hash(payload: dict[str, Any]) -> str:
    """Compute a deterministic hash for Pinares FEM fixture contracts."""

    payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(payload_bytes).hexdigest()


def _stable_public_float(value: float) -> float:
    """Round public numeric evidence past platform-noise precision."""

    return float(f"{float(value):.11g}")


def _stable_public_payload(value: Any) -> Any:
    """Return a JSON payload with float noise normalized across Python/NumPy builds."""

    if isinstance(value, float):
        return _stable_public_float(value)
    if isinstance(value, list):
        return [_stable_public_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_stable_public_payload(item) for item in value]
    if isinstance(value, dict):
        return {key: _stable_public_payload(item) for key, item in value.items()}
    return value


def run_public_pinares_fixed_price_proxy_fixture(
    *, case: PinaresFixedPriceProxyCase | None = None, refresh_exports: bool = False
) -> PinaresFEMProxyReport:
    """Run the public-synthetic Pinares fixed-price weak-form proxy fixture."""

    case = case or PinaresFixedPriceProxyCase()
    rows = tuple(_run_row(case, refinement_level=level) for level in case.refinement_levels)
    final = rows[-1]
    intrinsic = case.survival_probability * max(case.spot_uf - case.strike_uf, 0.0)
    upper_bound = case.survival_probability * case.spot_uf
    no_arbitrage = {
        "value_minus_intrinsic_uf": final.observed_price_uf - intrinsic,
        "upper_gap_uf": upper_bound - final.observed_price_uf,
        "value_bound_ok": final.observed_price_uf >= intrinsic - 1e-12,
        "upper_bound_ok": final.observed_price_uf <= upper_bound + 1e-12,
        "delta_lower_bound_ok": final.observed_delta >= -1e-12,
        "delta_upper_bound_ok": final.observed_delta <= case.survival_probability + 1e-12,
        "gamma_non_negative_ok": final.observed_gamma >= -1e-12,
        "survival_scale_ok": 0.0 <= case.survival_probability <= 1.0,
    }
    report = PinaresFEMProxyReport(
        case=case,
        rows=rows,
        expected_price_uf=final.expected_price_uf,
        observed_price_uf=final.observed_price_uf,
        price_absolute_error_uf=final.absolute_error_uf,
        price_relative_error=final.relative_error,
        expected_delta=final.expected_delta,
        observed_delta=final.observed_delta,
        delta_absolute_error=final.delta_absolute_error,
        expected_gamma=final.expected_gamma,
        observed_gamma=final.observed_gamma,
        gamma_absolute_error=final.gamma_absolute_error,
        no_arbitrage=no_arbitrage,
        config_hash="",
    )
    report = PinaresFEMProxyReport(**{**report.__dict__, "config_hash": _config_hash(report)})
    if refresh_exports:
        write_public_pinares_fixed_price_problem_spec(report=report)
        write_public_pinares_fixed_price_result_export(report=report, refresh=True)
        write_public_pinares_unsupported_problem_spec(refresh=True)
        write_public_pinares_quant_problem_spec(report=report)
    return report


def write_public_pinares_fixed_price_problem_spec(
    path: Path | str = PINARES_FEM_PROXY_PROBLEM_SPEC_PATH,
    *,
    report: PinaresFEMProxyReport | None = None,
) -> Path:
    """Write the public-synthetic Pinares FEM problem spec."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    case = report.case if report is not None else PinaresFixedPriceProxyCase()
    payload = public_pinares_fixed_price_problem_spec(case=case)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def write_public_pinares_fixed_price_result_export(
    path: Path | str = PINARES_FEM_PROXY_RESULT_EXPORT_PATH,
    *,
    refresh: bool = False,
    report: PinaresFEMProxyReport | None = None,
) -> Path:
    """Run and write the Pinares FEM result export in a stable public artifact shape."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if (not target.exists()) or refresh:
        if report is None:
            report = run_public_pinares_fixed_price_proxy_fixture()
        target.write_text(
            json.dumps(report.export_payload(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return target


def write_public_pinares_unsupported_problem_spec(
    path: Path | str = PINARES_FEM_PROXY_UNSUPPORTED_SPEC_PATH, *, refresh: bool = False
) -> Path:
    """Write the unsupported full-deal public-synthetic request fixture."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if (not target.exists()) or refresh:
        payload = public_pinares_full_deal_unsupported_problem_spec()
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def write_public_pinares_quant_problem_spec(
    path: Path | str = PINARES_QPS_FIXTURE_PATH,
    *,
    report: PinaresFEMProxyReport | None = None,
) -> Path:
    """Write the shared QuantProblemSpec consumer fixture for adapter tests."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    case = report.case if report is not None else PinaresFixedPriceProxyCase()
    payload = public_pinares_fixed_price_problem_spec(case=case)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def _run_row(
    case: PinaresFixedPriceProxyCase, *, refinement_level: int
) -> PinaresFEMProxyConvergenceRow:
    dynamics = DynamicsParametersBlackScholes(r=case.risk_free_rate, q=0.0, sig=case.volatility)
    market = Market(r=dynamics.r)
    option = EuropeanOptionBs(k=1.0, q=dynamics.q, mkt=market)
    times = np.linspace(0.0, case.maturity_years, case.time_steps + 1)
    mesh, config = create_mesh([case.domain_max_ratio], refinement_level)
    mesh = mesh.with_boundaries(
        {
            "left": lambda x: np.isclose(x[0], 0.0),
            "right": lambda x: np.isclose(x[0], case.domain_max_ratio),
        }
    )
    space = SpaceSolver(mesh, dynamics, option, is_call=True, config=config)
    with np.errstate(divide="ignore", invalid="ignore"):
        solution = ThetaScheme(theta=0.5).solve(
            times, space, boundary_condition=DirichletBC(["left", "right"])
        )
    normalized_price, normalized_delta, normalized_gamma = _interpolate_solution_and_greeks(
        space.Vh.doflocs[0], solution[-1], case.spot_ratio
    )
    expected_price, expected_delta, expected_gamma = _analytical_reference(case)
    observed_price = case.survival_probability * case.strike_uf * normalized_price
    observed_delta = case.survival_probability * normalized_delta
    observed_gamma = case.survival_probability * normalized_gamma / case.strike_uf
    absolute_error = abs(observed_price - expected_price)
    relative_error = absolute_error / max(abs(expected_price), 1.0)
    delta_absolute_error = abs(observed_delta - expected_delta)
    gamma_absolute_error = abs(observed_gamma - expected_gamma)
    values = (
        observed_price,
        absolute_error,
        relative_error,
        observed_delta,
        expected_delta,
        observed_gamma,
        expected_gamma,
    )
    if any(not isfinite(float(value)) for value in values):
        msg = "non-finite Pinares FEM proxy fixture result"
        raise FloatingPointError(msg)
    return PinaresFEMProxyConvergenceRow(
        refinement_level=refinement_level,
        time_steps=case.time_steps,
        degrees_of_freedom=int(space.Vh.N),
        observed_price_uf=observed_price,
        expected_price_uf=expected_price,
        absolute_error_uf=absolute_error,
        relative_error=relative_error,
        observed_delta=observed_delta,
        expected_delta=expected_delta,
        delta_absolute_error=delta_absolute_error,
        observed_gamma=observed_gamma,
        expected_gamma=expected_gamma,
        gamma_absolute_error=gamma_absolute_error,
    )


def _interpolate_solution_and_greeks(
    dof_locations: np.ndarray, values: np.ndarray, spot_ratio: float
) -> tuple[float, float, float]:
    """Interpolate value and quadratic-fit Greeks at normalized spot."""

    order = np.argsort(dof_locations)
    coordinates = dof_locations[order]
    ordered_values = values[order]
    normalized_price = float(np.interp(spot_ratio, coordinates, ordered_values))
    nearest = np.argsort(np.abs(coordinates - spot_ratio))[:5]
    if len(nearest) < 3:
        msg = "at least three FEM nodes are required for quadratic Greek extraction"
        raise ValueError(msg)
    fit_coordinates = coordinates[np.sort(nearest)]
    fit_values = ordered_values[np.sort(nearest)]
    coeff = np.polyfit(fit_coordinates, fit_values, 2)
    first_derivative = 2.0 * coeff[0] * spot_ratio + coeff[1]
    second_derivative = 2.0 * coeff[0]
    return normalized_price, float(first_derivative), float(second_derivative)


def _analytical_reference(
    case: PinaresFixedPriceProxyCase,
) -> tuple[float, float, float]:
    sqrt_t = np.sqrt(case.maturity_years)
    d1 = (
        np.log(case.spot_uf / case.strike_uf)
        + (case.risk_free_rate + 0.5 * case.volatility**2) * case.maturity_years
    ) / (case.volatility * sqrt_t)
    d2 = d1 - case.volatility * sqrt_t
    unscaled_price = case.spot_uf * spst.norm.cdf(d1) - case.strike_uf * np.exp(
        -case.risk_free_rate * case.maturity_years
    ) * spst.norm.cdf(d2)
    delta = spst.norm.cdf(d1)
    gamma = spst.norm.pdf(d1) / (case.spot_uf * case.volatility * sqrt_t)
    return (
        case.survival_probability * float(unscaled_price),
        case.survival_probability * float(delta),
        case.survival_probability * float(gamma),
    )


def _config_hash(report: PinaresFEMProxyReport) -> str:
    payload = {
        "benchmark_id": PINARES_FEM_FIXED_PRICE_PROXY_BENCHMARK_ID,
        "fixture_id": report.case.fixture_id,
        "problem_id": report.case.problem_id,
        "problem_hash": report.case.problem_hash,
        "route_id": report.case.route_id,
        "backend_id": report.case.backend_id,
        "spot_uf": report.case.spot_uf,
        "strike_uf": report.case.strike_uf,
        "risk_free_rate": report.case.risk_free_rate,
        "volatility": report.case.volatility,
        "maturity_years": report.case.maturity_years,
        "survival_probability": report.case.survival_probability,
        "s_max_uf": report.case.s_max_uf,
        "valuation_date": report.case.valuation_date,
        "maturity_date": report.case.maturity_date,
        "refinement_levels": list(report.case.refinement_levels),
        "time_steps": report.case.time_steps,
        "weak_form": _weak_form_metadata(),
        "mesh_metadata": _mesh_metadata(report.case),
        "time_metadata": _time_metadata(report.case),
        "boundary_conditions": _boundary_metadata(report.case),
        "tolerances": {
            "price_abs_uf": report.case.price_abs_tolerance_uf,
            "delta_abs": report.case.delta_abs_tolerance,
            "gamma_abs": report.case.gamma_abs_tolerance,
        },
    }
    return build_pinares_fem_proxy_hash(payload)


def _weak_form_metadata() -> dict[str, str]:
    return {
        "equation_id": "pinares_fixed_price_proxy_black_scholes_weak_form",
        "sign_convention": "existing_forward_tau_identity_transform_black_scholes_forms",
        "time_transformation": "tau = T - t",
        "coordinate_transform": "normalized_spot_x_equals_S_over_K",
        "payoff_scaling": "UF value = survival_probability * K_uf * normalized_call_value",
    }


def _mesh_metadata(case: PinaresFixedPriceProxyCase) -> dict[str, Any]:
    return {
        "mesh_family": "line_uniform",
        "element_family": "lagrange_p2",
        "domain_min": 0.0,
        "domain_max": case.domain_max_ratio,
        "spatial_domain": f"[0, {case.domain_max_ratio:.12g}] normalized spot S/K",
        "refinement_levels": list(case.refinement_levels),
        "solver_backing": "scikit-fem+sparse-direct",
    }


def _time_metadata(case: PinaresFixedPriceProxyCase) -> dict[str, float | int | str]:
    return {
        "integrator": "theta_crank_nicolson",
        "theta": 0.5,
        "time_steps": case.time_steps,
        "start_time": 0.0,
        "end_time": case.maturity_years,
        "time_domain": f"[0, {case.maturity_years:g}]",
    }


def _boundary_metadata(
    case: PinaresFixedPriceProxyCase,
) -> list[dict[str, float | int | str]]:
    return [
        {
            "location": "S=0",
            "condition_type": "dirichlet",
            "expression": "0",
            "enforced_nodes": 1,
        },
        {
            "location": "S=S_max",
            "condition_type": "dirichlet",
            "expression": "linear_growth_call_far_field",
            "s_max_uf": case.s_max_uf,
            "enforced_nodes": 1,
        },
    ]


__all__ = [
    "PINARES_FEM_FAIL_CLOSED_BENCHMARK_ID",
    "PINARES_FEM_FIXED_PRICE_PROXY_BENCHMARK_ID",
    "PINARES_FEM_FIXED_PRICE_PROXY_BENCHMARK_IDS",
    "PINARES_FEM_FIXED_PRICE_PROXY_FIXTURE_ID",
    "PINARES_FEM_FIXED_PRICE_PROXY_PROBLEM_HASH",
    "PINARES_FEM_FIXED_PRICE_PROXY_PROBLEM_ID",
    "PINARES_FEM_FIXED_PRICE_PROXY_ROUTE_ID",
    "PINARES_FEM_PROXY_PROBLEM_SPEC_PATH",
    "PINARES_FEM_PROXY_RESULT_EXPORT_PATH",
    "PINARES_FEM_PROXY_UNSUPPORTED_SPEC_PATH",
    "PINARES_QPS_CONTRACT_BENCHMARK_ID",
    "PINARES_QPS_FIXTURE_PATH",
    "PinaresFEMProxyConvergenceRow",
    "PinaresFEMProxyReport",
    "PinaresFixedPriceProxyCase",
    "build_pinares_fem_proxy_hash",
    "public_pinares_fixed_price_problem_spec",
    "public_pinares_full_deal_unsupported_problem_spec",
    "run_public_pinares_fixed_price_proxy_fixture",
    "write_public_pinares_fixed_price_problem_spec",
    "write_public_pinares_fixed_price_result_export",
    "write_public_pinares_quant_problem_spec",
    "write_public_pinares_unsupported_problem_spec",
]
