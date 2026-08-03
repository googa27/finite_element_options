"""Public fixture provenance, canonicalization, and integrity helpers.

The helpers in this module own serialization-boundary behavior for public
synthetic Black--Scholes fixture evidence.  Numerical solver results are rounded
only for reproducible publication and hashing; acceptance tolerances remain the
separate FEM comparison-policy values emitted in each payload.
"""

from __future__ import annotations

from hashlib import sha256
import json
from math import isfinite
from typing import Any, Mapping

PUBLIC_NUMERIC_CANONICALIZATION: dict[str, str | int] = {
    "policy_id": "public-synthetic-fem-bs-significant-digits-v1",
    "significant_digits": 13,
    "scope": (
        "published Black-Scholes fixture numerical rows and summaries before "
        "JSON hashing/serialization"
    ),
    "rationale": (
        "SciPy/SuperLU-supported environments can differ by a few ULPs while "
        "satisfying the same FEM oracle tolerances; canonical rows keep hashes "
        "stable without changing comparison tolerances."
    ),
}
_SIGNIFICANT_DIGITS = int(PUBLIC_NUMERIC_CANONICALIZATION["significant_digits"])


def build_fixture_config_hash(payload: Mapping[str, Any]) -> str:
    """Compute a deterministic hash for fixture contracts and export control."""

    payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return sha256(payload_bytes).hexdigest()


def public_pde_convention_metadata() -> dict[str, str]:
    """Return public Black--Scholes PDE convention metadata."""

    return {
        "strong_form": "d_tau u = 0.5*sigma^2*S^2*d_SS u + r*S*d_S u - r*u",
        "operator_sign": "forward_tau_generator_minus_discount",
        "time_orientation": "backward_pricing_time_transformed_to_forward_tau",
        "state_variable": "S; solver uses normalized spot S/K and public values are scaled by strike",
        "initial_condition_tau_zero": "u(S,0)=max(S-K,0)",
        "terminal_condition_original_time": "V(S,T)=max(S-K,0)",
        "lower_dirichlet_boundary": "u(0,tau)=0",
        "upper_dirichlet_boundary": "u(S_max,tau)=Black-Scholes analytical value at finite S_max; documented as linear-growth far-field proxy",
        "source_term": "0",
        "volatility_convention": "volatility sigma is annualized_decimal; diffusion coefficient uses sigma**2",
    }


def public_fixture_provenance_metadata() -> dict[str, str]:
    """Return public-synthetic fixture provenance metadata."""

    return {
        "fixture_owner": "googa27/finite_element_options",
        "consumer": "arxiv-implementation-lab",
        "source_issue": "googa27/finite_element_options#74",
        "parity_issue": "googa27/finite_element_options#64",
        "verification_issue": "googa27/finite_element_options#117",
        "privacy_class": "public_synthetic",
        "generator": "finite_element_options.validation.black_scholes_parity.run_public_black_scholes_parity_fixture",
        "export_script": "scripts/export_arxiv_lab_black_scholes_fixture.py",
        "oracle": "analytical Black-Scholes price/Delta/Gamma",
    }


def canonicalize_public_number(value: float) -> float:
    """Return the deterministic public representation for a finite float."""

    if not isfinite(value):
        raise FloatingPointError("public fixture numerical value is not finite")
    return float(f"{value:.{_SIGNIFICANT_DIGITS}g}")


def canonicalize_public_payload(value: Any) -> Any:
    """Recursively canonicalize finite floats while preserving JSON structure."""

    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return canonicalize_public_number(value)
    if isinstance(value, dict):
        return {key: canonicalize_public_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [canonicalize_public_payload(item) for item in value]
    return value


def canonicalize_black_scholes_row(row: Mapping[str, Any]) -> dict[str, float | int]:
    """Canonicalize one Black--Scholes convergence row before publication.

    Primary observed/reference values are rounded to the published significant
    digits, then derived error fields are recomputed from those rounded primary
    values. This prevents exact-error fields from retaining environment-specific
    ULP noise while preserving a strict structural contract.
    """

    observed_price = canonicalize_public_number(float(row["observed_price"]))
    expected_price = canonicalize_public_number(float(row["expected_price"]))
    observed_delta = canonicalize_public_number(float(row["observed_delta"]))
    expected_delta = canonicalize_public_number(float(row["expected_delta"]))
    observed_gamma = canonicalize_public_number(float(row["observed_gamma"]))
    expected_gamma = canonicalize_public_number(float(row["expected_gamma"]))
    absolute_error = canonicalize_public_number(abs(observed_price - expected_price))
    delta_absolute_error = canonicalize_public_number(
        abs(observed_delta - expected_delta)
    )
    gamma_absolute_error = canonicalize_public_number(
        abs(observed_gamma - expected_gamma)
    )
    return {
        "refinement_level": int(row["refinement_level"]),
        "time_steps": int(row["time_steps"]),
        "degrees_of_freedom": int(row["degrees_of_freedom"]),
        "observed_price": observed_price,
        "expected_price": expected_price,
        "absolute_error": absolute_error,
        "relative_error": canonicalize_public_number(
            absolute_error / max(abs(expected_price), 1.0)
        ),
        "observed_delta": observed_delta,
        "expected_delta": expected_delta,
        "delta_absolute_error": delta_absolute_error,
        "observed_gamma": observed_gamma,
        "expected_gamma": expected_gamma,
        "gamma_absolute_error": gamma_absolute_error,
    }


def black_scholes_summary_from_row(
    row: Mapping[str, Any],
    *,
    price_tolerance_absolute: float,
    price_tolerance_relative: float,
    delta_tolerance_absolute: float,
    gamma_tolerance_absolute: float,
) -> dict[str, float]:
    """Build a deterministic public result summary from the canonical final row."""

    return canonicalize_public_payload(
        {
            "expected_price": float(row["expected_price"]),
            "observed_price": float(row["observed_price"]),
            "price_absolute_error": float(row["absolute_error"]),
            "price_relative_error": float(row["relative_error"]),
            "price_tolerance_absolute": price_tolerance_absolute,
            "price_tolerance_relative": price_tolerance_relative,
            "expected_delta": float(row["expected_delta"]),
            "observed_delta": float(row["observed_delta"]),
            "delta_absolute_error": float(row["delta_absolute_error"]),
            "delta_tolerance_absolute": delta_tolerance_absolute,
            "expected_gamma": float(row["expected_gamma"]),
            "observed_gamma": float(row["observed_gamma"]),
            "gamma_absolute_error": float(row["gamma_absolute_error"]),
            "gamma_tolerance_absolute": gamma_tolerance_absolute,
        }
    )


def finalize_public_result_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Canonicalize a final result payload and attach its recomputable hash."""

    canonical = canonicalize_public_payload(
        {key: value for key, value in payload.items() if key != "result_hash"}
    )
    canonical["result_hash"] = build_fixture_config_hash(canonical)
    return canonical


__all__ = [
    "PUBLIC_NUMERIC_CANONICALIZATION",
    "black_scholes_summary_from_row",
    "build_fixture_config_hash",
    "canonicalize_black_scholes_row",
    "canonicalize_public_number",
    "canonicalize_public_payload",
    "finalize_public_result_payload",
    "public_fixture_provenance_metadata",
    "public_pde_convention_metadata",
]
