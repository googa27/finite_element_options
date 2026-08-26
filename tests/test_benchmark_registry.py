"""Benchmark registry extraction and compatibility contract."""

from __future__ import annotations

import pytest

import finite_element_options.validation as validation
import finite_element_options.validation.evidence.benchmark_registry as benchmark_registry
from finite_element_options.validation import verification_gates


def test_benchmark_registry_module_exports_default_contract() -> None:
    registry = benchmark_registry.default_benchmark_registry()

    assert benchmark_registry.DEFAULT_VALIDATION_BENCHMARK_ID in registry
    assert set(benchmark_registry.REQUIRED_TOLERANCE_COMPONENTS) == {
        "discretization",
        "oracle",
        "floating_point",
    }
    for benchmark_id, spec in registry.items():
        assert isinstance(spec, benchmark_registry.BenchmarkSpec)
        assert spec.to_public_dict()["benchmark_id"] == benchmark_id


def test_verification_gates_reexports_benchmark_registry_symbols() -> None:
    assert validation.ValidationGateError is benchmark_registry.ValidationGateError
    assert validation.BenchmarkSpec is benchmark_registry.BenchmarkSpec
    assert validation.default_benchmark_registry is benchmark_registry.default_benchmark_registry
    assert verification_gates.ValidationGateError is benchmark_registry.ValidationGateError
    assert verification_gates.BenchmarkSpec is benchmark_registry.BenchmarkSpec
    assert (
        verification_gates.REQUIRED_TOLERANCE_COMPONENTS
        is benchmark_registry.REQUIRED_TOLERANCE_COMPONENTS
    )
    assert (
        verification_gates.DEFAULT_VALIDATION_BENCHMARK_ID
        == benchmark_registry.DEFAULT_VALIDATION_BENCHMARK_ID
    )
    assert (
        verification_gates.default_benchmark_registry
        is benchmark_registry.default_benchmark_registry
    )


def test_benchmark_spec_validation_keeps_validation_gate_error_type() -> None:
    incomplete = benchmark_registry.BenchmarkSpec(
        benchmark_id="MISSING-TOLERANCES",
        model="Black-Scholes",
        instrument="European call",
        state_convention="forward tau",
        domain="[0, 4K]",
        grid="line_uniform",
        time_schedule="theta",
        oracle="analytical",
        norm="linf",
        expected_order=2.0,
        tolerance_components={"discretization": 1.0e-3},
    )

    with pytest.raises(benchmark_registry.ValidationGateError, match="tolerance components"):
        incomplete.validate()
