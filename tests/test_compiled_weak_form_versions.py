"""Characterize genuine producer output independently of packaged authority."""

import json
from hashlib import sha256
from pathlib import Path

from finite_element_options.validation.compiled_weak_form_adapter import (
    screen_compiled_weak_form,
)
from finite_element_options.validation.compiled_weak_form_golden import (
    packaged_golden_fixture,
)


def test_genuine_compiler_v1_fixture_is_admitted() -> None:
    path = (
        Path(__file__).parent / "fixtures/compiled_weak_form/black_scholes_call_v1.json"
    )
    payload = json.loads(path.read_bytes())
    screen = screen_compiled_weak_form(payload)
    assert screen.accepted, screen.diagnostics


def test_v1_fixture_is_exact_producer_data_with_the_historical_route() -> None:
    root = Path(__file__).parent / "fixtures/compiled_weak_form"
    previous = json.loads((root / "black_scholes_call_v0.json").read_bytes())
    current = json.loads((root / "black_scholes_call_v1.json").read_bytes())
    assert previous == packaged_golden_fixture()
    assert current == packaged_golden_fixture(
        compiler_version="pde_ir_symbolic_compiler.v1"
    )
    assert {k: v for k, v in previous.items() if k != "compiled_operator"} == {
        k: v for k, v in current.items() if k != "compiled_operator"
    }
    emitted = {"accepted": True, "compiled_operator": current["compiled_operator"]}
    encoded = (
        json.dumps(
            emitted,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        + b"\n"
    )
    assert (
        sha256(encoded).hexdigest()
        == "88de7622123f40e44649d3d27a86550751d1b4c3bd4b0a996ebc4700c8456f10"
    )
