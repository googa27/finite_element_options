"""Haircut backend adapter contract tests for issue #115."""

from __future__ import annotations

import ast
import json
import pathlib
import sys
import types
from dataclasses import dataclass, field, replace
from enum import Enum

import pytest

from finite_element_options.contracts import UnsupportedRouteError
from finite_element_options.integrations import haircut_protocol
from finite_element_options.integrations.haircut_backend import (
    HAIRCUT_BACKEND_ENTRY_POINT,
    HAIRCUT_BACKEND_IMPLEMENTATION_ID,
    ContractMajorMismatchError,
    FiniteElementHaircutBackend,
    HaircutProtocolUnavailableError,
    create_backend,
)
from finite_element_options.validation.pinares_fixed_price_proxy import (
    public_pinares_fixed_price_problem_spec,
)

FIXTURE_DIR = pathlib.Path(__file__).parent / "fixtures" / "quant_problem_specs"


@dataclass(frozen=True, order=True)
class _FakeContractVersion:
    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, value: object) -> "_FakeContractVersion":
        if isinstance(value, cls):
            return value
        major, minor, patch = (int(part) for part in str(value).split("."))
        return cls(major, minor, patch)

    def is_compatible_with(self, other: object) -> bool:
        return self.major == self.parse(other).major

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


class _FakeBackendMaturity(Enum):
    VALIDATED = "validated"


class _FakeMethodMaturity(Enum):
    PRODUCTION_GATED = "production_gated"


@dataclass(frozen=True)
class _FakeBackendIdentity:
    distribution_name: str
    distribution_version: str
    implementation_id: str
    implementation_version: str
    contract_version: _FakeContractVersion
    maturity: _FakeBackendMaturity
    license_identifier: str = "UNKNOWN"
    build_metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class _FakeMethodCapability:
    method_id: str
    backend_id: str
    family: str
    exactness: str
    maturity: _FakeMethodMaturity
    equation_families: tuple[str, ...]
    dimensions: tuple[int, ...]
    state_variables: tuple[str, ...]
    boundary_conditions: tuple[str, ...]
    smoothness_assumptions: tuple[str, ...]
    output_types: tuple[str, ...]
    runtime_controls: tuple[str, ...]
    validation_gates: tuple[str, ...]
    failure_modes: tuple[str, ...]
    fallback_route_id: str | None
    fallback_triggers: tuple[str, ...]
    fallback_policy: str
    references: tuple[str, ...]


@dataclass(frozen=True)
class _FakeBackendCapabilityManifest:
    backend_id: str
    contract_version: str
    methods: tuple[_FakeMethodCapability, ...]

    def validate(self) -> tuple[str, ...]:
        return ()


@pytest.fixture(autouse=True)
def haircut_public_solver_seam(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide Haircut's public seam shape without a source-tree dependency."""

    haircut = types.ModuleType("haircut")
    solvers = types.ModuleType("haircut.solvers")
    backend_protocol = types.ModuleType("haircut.solvers.backend_protocol")
    contracts = types.ModuleType("haircut.solvers.contracts")
    backend_protocol.ContractVersion = _FakeContractVersion  # type: ignore[attr-defined]
    backend_protocol.BackendMaturity = _FakeBackendMaturity  # type: ignore[attr-defined]
    backend_protocol.BackendIdentity = _FakeBackendIdentity  # type: ignore[attr-defined]
    contracts.MethodMaturity = _FakeMethodMaturity  # type: ignore[attr-defined]
    contracts.MethodCapability = _FakeMethodCapability  # type: ignore[attr-defined]
    contracts.BackendCapabilityManifest = _FakeBackendCapabilityManifest  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "haircut", haircut)
    monkeypatch.setitem(sys.modules, "haircut.solvers", solvers)
    monkeypatch.setitem(
        sys.modules, "haircut.solvers.backend_protocol", backend_protocol
    )
    monkeypatch.setitem(sys.modules, "haircut.solvers.contracts", contracts)


def _vanilla_payload() -> dict[str, object]:
    payload = json.loads(
        (FIXTURE_DIR / "vanilla_call.json").read_text(encoding="utf-8")
    )
    assert isinstance(payload, dict)
    return payload


def _executable_payload() -> dict[str, object]:
    return public_pinares_fixed_price_problem_spec()


def _local_backend() -> FiniteElementHaircutBackend:
    return create_backend()


def _section(payload: dict[str, object], name: str) -> dict[str, object]:
    section = payload[name]
    assert isinstance(section, dict)
    return section


def test_haircut_backend_identity_manifest_and_entry_point_are_canonical() -> None:
    from haircut.solvers.backend_protocol import BackendIdentity
    from haircut.solvers.contracts import BackendCapabilityManifest

    backend = _local_backend()

    assert isinstance(backend.identity, BackendIdentity)
    assert isinstance(backend.capability_manifest, BackendCapabilityManifest)
    assert backend.identity.implementation_id == HAIRCUT_BACKEND_IMPLEMENTATION_ID
    assert backend.identity.distribution_name == "finite-element-options"
    assert backend.identity.build_metadata["entry_point"] == HAIRCUT_BACKEND_ENTRY_POINT
    assert backend.capability_manifest.backend_id == backend.identity.implementation_id
    manifest = backend.fem_capability_manifest()
    assert manifest["backend_id"] == "finite_element_options.fem_backend.v0"
    assert manifest["status"] == "validated"
    assert manifest["supported_dimensions"] == [1]


def test_haircut_backend_screen_preserves_quant_problem_spec_without_assembly() -> None:
    backend = _local_backend()
    result = backend.screen(_executable_payload())

    assert result.supported
    assert result.diagnostics == ()
    assert result.request["source_schema_version"] == "quant-problem-spec/v0"
    assert result.request["backend_id"] == "finite_element_options.fem_backend.v0"
    assert result.request["measure"] == "Q*"
    assert result.request["numeraire"] == "UF_money_market_account_proxy"
    assert result.request["units"]["underlying"] == "UF"
    assert result.request["requested_outputs"] == ("value", "delta", "gamma")


def test_haircut_backend_screen_fails_closed_before_assembly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _local_backend()
    payload = _vanilla_payload()
    math_section = _section(payload, "mathematical_problem")
    payload["mathematical_problem"] = {
        **math_section,
        "pde_terms": ["drift", "diffusion", "jump_integral"],
        "exercise_style": "swing",
    }

    def forbidden_runner() -> None:
        raise AssertionError(
            "validation runner must not execute for unsupported screen"
        )

    from finite_element_options.validation import pinares_fixed_price_proxy as proxy

    monkeypatch.setattr(
        proxy,
        "run_public_pinares_fixed_price_proxy_fixture",
        forbidden_runner,
    )
    result = backend.screen(payload)

    assert not result.supported
    assert {diagnostic["field"] for diagnostic in result.diagnostics} >= {
        "pde_terms",
        "exercise_style",
    }
    assert {diagnostic["reason"] for diagnostic in result.diagnostics} >= {
        "unsupported_pde_term",
        "unsupported_exercise_style",
    }


def test_haircut_backend_screen_fails_closed_for_malformed_dimension() -> None:
    backend = _local_backend()
    payload = _vanilla_payload()
    math_section = _section(payload, "mathematical_problem")
    payload["mathematical_problem"] = {**math_section, "dimension": "auto"}

    result = backend.screen(payload)

    assert not result.supported
    assert any(
        diagnostic["field"] == "dimension"
        and diagnostic["reason"] == "unsupported_dimension"
        for diagnostic in result.diagnostics
    )


@pytest.mark.parametrize("dimension", [True, 1.0])
def test_haircut_backend_rejects_non_integer_fixture_dimensions(
    dimension: object,
) -> None:
    backend = _local_backend()
    payload = _executable_payload()
    math_section = _section(payload, "mathematical_problem")
    payload["mathematical_problem"] = {**math_section, "dimension": dimension}

    result = backend.screen(payload)

    assert not result.supported
    with pytest.raises(
        UnsupportedRouteError, match="validated public-synthetic executable benchmark"
    ):
        backend.solve(payload)


def test_haircut_backend_rejects_unknown_top_level_private_fields() -> None:
    backend = _local_backend()
    payload = _executable_payload()
    payload["private_terms"] = {"cashflows": [1, 2, 3]}

    result = backend.screen(payload)

    assert not result.supported
    with pytest.raises(
        UnsupportedRouteError, match="validated public-synthetic executable benchmark"
    ):
        backend.solve(payload)


def test_haircut_backend_rejects_supported_route_without_executable_fixture() -> None:
    backend = _local_backend()
    payload = _vanilla_payload()

    result = backend.screen(payload)

    assert not result.supported
    assert result.diagnostics[0]["reason"] == "unsupported_benchmark"
    assert result.diagnostics[0]["field"] == "benchmark_ids"
    with pytest.raises(
        UnsupportedRouteError, match="validated public-synthetic executable benchmark"
    ):
        backend.solve(payload)


def test_haircut_backend_rejects_private_payload_even_with_public_fixture_ids() -> None:
    backend = _local_backend()
    payload = _executable_payload()
    payload["privacy_class"] = "private"

    result = backend.screen(payload)

    assert not result.supported
    assert result.diagnostics[0]["reason"] == "unsupported_benchmark"
    with pytest.raises(
        UnsupportedRouteError, match="validated public-synthetic executable benchmark"
    ):
        backend.solve(payload)


def test_haircut_backend_rejects_mutated_public_fixture_fields() -> None:
    backend = _local_backend()
    payload = _executable_payload()
    math_section = _section(payload, "mathematical_problem")
    payload["mathematical_problem"] = {
        **math_section,
        "terminal_payoff": {"expression": "mutated private payoff"},
    }

    result = backend.screen(payload)

    assert not result.supported
    assert result.diagnostics[0]["reason"] == "unsupported_benchmark"
    with pytest.raises(
        UnsupportedRouteError, match="validated public-synthetic executable benchmark"
    ):
        backend.solve(payload)


def test_haircut_backend_results_are_json_serializable() -> None:
    backend = _local_backend()
    screen = backend.screen(_executable_payload())

    screen_payload = json.loads(json.dumps(screen.as_dict()))

    assert screen_payload["status"] == "supported"
    assert screen_payload["request"]["boundary_conditions"] == ["dirichlet"]


def test_haircut_backend_solve_executes_only_validated_public_synthetic_fixture() -> (
    None
):
    backend = _local_backend()

    result = backend.solve(_executable_payload())

    assert result.passed
    assert result.problem_id == "pinares.fixed_price_option_proxy.v1"
    assert result.benchmark_ids == (
        "PINARES-FEM-FIXED-PRICE-PROXY-V0",
        "PINARES-QPS-FIXED-PRICE-PROXY-V0",
    )
    assert result.values["price"] == pytest.approx(
        result.values["oracle_price"], abs=1.0
    )
    assert result.diagnostics["fallbacks"] == ()
    assert result.evidence["privacy_class"] == "public_synthetic"
    assert result.evidence["numeraire"] == result.request["numeraire"]
    assert result.request["boundary_conditions"] == ("dirichlet",)


def test_haircut_backend_imports_only_public_solver_protocol_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []
    original_import_module = haircut_protocol.importlib.import_module

    def spy_import_module(name: str):
        if name.startswith("haircut"):
            seen.append(name)
        return original_import_module(name)

    monkeypatch.setattr(haircut_protocol.importlib, "import_module", spy_import_module)
    create_backend()
    assert set(seen) == {
        "haircut.solvers.backend_protocol",
        "haircut.solvers.contracts",
    }


def test_haircut_backend_fails_closed_on_contract_major_mismatch() -> None:
    from finite_element_options.contracts import DEFAULT_FEM_CAPABILITY_MANIFEST

    manifest = replace(DEFAULT_FEM_CAPABILITY_MANIFEST, contract_version="1.0.0")
    with pytest.raises(ContractMajorMismatchError, match="contract major mismatch"):
        create_backend(manifest=manifest, expected_contract_version="0.1.0")


def test_haircut_backend_fails_closed_when_public_protocol_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(name: str):
        raise ImportError(name)

    monkeypatch.setattr(haircut_protocol.importlib, "import_module", unavailable)
    with pytest.raises(
        HaircutProtocolUnavailableError, match="released haircut-engine wheel"
    ):
        create_backend()


def test_haircut_backend_module_keeps_haircut_and_validation_imports_lazy() -> None:
    source = pathlib.Path(
        "src/finite_element_options/integrations/haircut_backend.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level_imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)]

    assert not [
        node.module
        for node in top_level_imports
        if node.module and node.module.startswith("haircut")
    ]
    assert not [
        node.module
        for node in top_level_imports
        if node.module and node.module.startswith("finite_element_options.validation")
    ]
