from __future__ import annotations

import json


from finite_element_options.contracts.formula_bundle import (
    finite_element_formula_bundle,
    formula_bundle_json,
    validate_formula_bundle,
)


def _contains_forbidden_style_key(value: object) -> bool:
    if isinstance(value, dict):
        if {"color", "style", "style_token", "css"} & set(value):
            return True
        return any(_contains_forbidden_style_key(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_forbidden_style_key(child) for child in value)
    return False


def test_fem_formula_bundle_exports_weak_form_crosswalk() -> None:
    payload = finite_element_formula_bundle()
    assert payload["producer"] == "finite_element_options"
    assert not validate_formula_bundle(payload)
    roles = {
        component["role"] for formula in payload["formulas"] for component in formula["components"]
    }
    assert {"weak_form", "bilinear_form", "linear_form", "basis", "quadrature"} <= roles
    assert not _contains_forbidden_style_key(payload)


def test_fem_formula_bundle_json_round_trips() -> None:
    payload = json.loads(formula_bundle_json())
    assert payload["formulas"][0]["formula_id"] == "fem_black_scholes_weak_form"
    assert payload["bundle_hash"].startswith("sha256:")
