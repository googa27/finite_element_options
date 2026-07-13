"""Haircut Engine solver-backend entry point for the FEM backend.

The adapter is intentionally an optional boundary module.  Importing it does not
import Haircut Engine, validation runners, meshes, bases, forms, or solvers.  The
entry-point factory returns native Haircut protocol objects when Haircut's public
protocol package is installed; otherwise it returns the same lightweight plugin
object used by repository-local tests and metadata inspection.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
from typing import Any, Literal

from finite_element_options.contracts import (
    DEFAULT_FEM_CAPABILITY_MANIFEST,
    FEMCapabilityManifest,
    FEMRouteRequest,
    UnsupportedReason,
    UnsupportedRouteDiagnostic,
    UnsupportedRouteError,
    diagnose_unsupported_route,
    ensure_route_supported,
)
from finite_element_options.integrations.haircut_protocol import (
    HAIRCUT_BACKEND_ENTRY_POINT,
    HAIRCUT_PUBLIC_CONTRACT_VERSION,
    ContractMajorMismatchError,
    HaircutProtocolUnavailableError,
    build_haircut_contracts,
)

AdapterStatus = Literal["supported", "unsupported"]
SolveStatus = Literal["passed", "failed"]

HAIRCUT_BACKEND_IMPLEMENTATION_ID = DEFAULT_FEM_CAPABILITY_MANIFEST.backend_id

_EXECUTABLE_PUBLIC_BENCHMARKS = frozenset(
    {
        "PINARES-FEM-FIXED-PRICE-PROXY-V0",
        "PINARES-QPS-FIXED-PRICE-PROXY-V0",
    }
)
_PUBLIC_SYNTHETIC_PROBLEM_IDS = frozenset({"pinares.fixed_price_option_proxy.v1"})


@dataclass(frozen=True)
class FEMBackendScreeningResult:
    """Capability-screening result emitted before numerical work."""

    backend_id: str
    status: AdapterStatus
    request: dict[str, Any]
    diagnostics: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]

    @property
    def supported(self) -> bool:
        """Whether the payload is eligible for execution."""

        return self.status == "supported"

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe screening payload."""

        return asdict(self)


@dataclass(frozen=True)
class FEMBackendSolveResult:
    """Normalized public-synthetic solve/evidence bundle for Haircut consumers."""

    backend_id: str
    status: SolveStatus
    problem_id: str
    benchmark_ids: tuple[str, ...]
    values: dict[str, float]
    diagnostics: dict[str, Any]
    evidence: dict[str, Any]
    request: dict[str, Any]

    @property
    def passed(self) -> bool:
        """Whether the executed repository-local evidence passed."""

        return self.status == "passed"

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe solve payload."""

        return asdict(self)


class FiniteElementHaircutBackend:
    """Thin fail-closed adapter for Haircut's public solver backend seam."""

    def __init__(
        self,
        manifest: FEMCapabilityManifest = DEFAULT_FEM_CAPABILITY_MANIFEST,
        *,
        expected_contract_version: str = HAIRCUT_PUBLIC_CONTRACT_VERSION,
    ) -> None:
        """Initialize native capabilities and validate Haircut's public contract major."""

        contracts = build_haircut_contracts(
            manifest,
            expected_contract_version=expected_contract_version,
        )
        self._manifest = manifest
        self._identity = contracts.identity
        self._capability_manifest = contracts.capability_manifest

    @property
    def identity(self) -> Any:
        """Return Haircut's public ``BackendIdentity`` shape."""

        return self._identity

    @property
    def manifest(self) -> FEMCapabilityManifest:
        """Return the native FEM capability manifest used for preflight."""

        return self._manifest

    @property
    def capability_manifest(self) -> Any:
        """Return Haircut's public ``BackendCapabilityManifest`` shape."""

        return self._capability_manifest

    def fem_capability_manifest(self) -> dict[str, Any]:
        """Return native FEM capability metadata for adapter diagnostics."""

        return self._manifest.to_public_dict()

    def screen(self, payload: Mapping[str, Any]) -> FEMBackendScreeningResult:
        """Map and validate a QuantProblemSpec-like payload before assembly."""

        request, route_diagnostics = _route_diagnostics(payload, self._manifest)
        execution_diagnostics = (
            () if route_diagnostics else _execution_diagnostics(payload)
        )
        diagnostics = (*route_diagnostics, *execution_diagnostics)
        return FEMBackendScreeningResult(
            backend_id=self._manifest.backend_id,
            status="unsupported" if diagnostics else "supported",
            request={} if request is None else asdict(request),
            diagnostics=tuple(
                _diagnostic_as_dict(diagnostic) for diagnostic in diagnostics
            ),
            manifest=self.fem_capability_manifest(),
        )

    def solve(self, payload: Mapping[str, Any]) -> FEMBackendSolveResult:
        """Execute only validated public-synthetic FEM fixture routes.

        Supported-looking generic PDE records are not enough to execute.  The
        payload must exactly match a public-synthetic fixture generated by this
        repository before validation runners are imported or mesh assembly starts.
        """

        request = FEMRouteRequest.from_quant_problem_spec(payload)
        ensure_route_supported(request, self._manifest)
        execution_diagnostics = _execution_diagnostics(payload)
        if execution_diagnostics:
            raise UnsupportedRouteError(execution_diagnostics)

        from finite_element_options.validation.pinares_fixed_price_proxy import (
            PINARES_FEM_FIXED_PRICE_PROXY_BENCHMARK_IDS,
            PINARES_FEM_FIXED_PRICE_PROXY_PROBLEM_ID,
            run_public_pinares_fixed_price_proxy_fixture,
        )

        report = run_public_pinares_fixed_price_proxy_fixture()
        problem_id = _optional_string(payload.get("problem_id"))
        if problem_id != PINARES_FEM_FIXED_PRICE_PROXY_PROBLEM_ID:
            raise UnsupportedRouteError(_unsupported_benchmark_diagnostics(payload))

        values = {
            "price": report.observed_price_uf,
            "oracle_price": report.expected_price_uf,
            "delta": report.observed_delta,
            "reference_delta": report.expected_delta,
            "gamma": report.observed_gamma,
            "reference_gamma": report.expected_gamma,
        }
        diagnostics = {
            "requested_benchmark_ids": _benchmark_ids(payload),
            "errors": {
                "price_abs_uf": report.price_absolute_error_uf,
                "price_rel": report.price_relative_error,
                "delta_abs": report.delta_absolute_error,
                "gamma_abs": report.gamma_absolute_error,
            },
            "no_arbitrage": report.no_arbitrage,
            "convergence_rows": [row.to_public_dict() for row in report.rows],
            "unsupported_route_diagnostics": (),
            "fallbacks": (),
        }
        evidence = {
            "adapter_schema_version": "haircut-fem-backend-adapter/v0",
            "source_schema_version": request.source_schema_version,
            "problem_id": problem_id,
            "privacy_class": _optional_string(payload.get("privacy_class")),
            "config_hash": report.config_hash,
            "route_id": report.case.route_id,
            "valuation_date": request.valuation_date or "",
            "numeraire": request.numeraire or "",
            "units": dict(request.units),
        }
        return FEMBackendSolveResult(
            backend_id=self._manifest.backend_id,
            status="passed" if report.converged else "failed",
            problem_id=problem_id,
            benchmark_ids=PINARES_FEM_FIXED_PRICE_PROXY_BENCHMARK_IDS,
            values=values,
            diagnostics=diagnostics,
            evidence=evidence,
            request=asdict(request),
        )


def create_backend(
    manifest: FEMCapabilityManifest = DEFAULT_FEM_CAPABILITY_MANIFEST,
    *,
    expected_contract_version: str = HAIRCUT_PUBLIC_CONTRACT_VERSION,
) -> FiniteElementHaircutBackend:
    """Entry-point factory for canonical ``haircut.solver_backends`` discovery."""

    return FiniteElementHaircutBackend(
        manifest=manifest,
        expected_contract_version=expected_contract_version,
    )


def _route_diagnostics(
    payload: Mapping[str, Any], manifest: FEMCapabilityManifest
) -> tuple[FEMRouteRequest | None, tuple[UnsupportedRouteDiagnostic, ...]]:
    try:
        request = FEMRouteRequest.from_quant_problem_spec(payload)
    except (TypeError, ValueError) as exc:
        return None, (
            UnsupportedRouteDiagnostic(
                reason=UnsupportedReason.UNSUPPORTED_DIMENSION,
                field="dimension",
                value=str(exc),
                supported=tuple(str(item) for item in manifest.supported_dimensions),
                message="FEM route request dimension must be an explicit supported integer before assembly.",
            ),
        )
    return request, diagnose_unsupported_route(request, manifest)


def _execution_diagnostics(
    payload: Mapping[str, Any],
) -> tuple[UnsupportedRouteDiagnostic, ...]:
    if _is_executable_public_synthetic_payload(payload):
        return ()
    return _unsupported_benchmark_diagnostics(payload)


def _unsupported_benchmark_diagnostics(
    payload: Mapping[str, Any],
) -> tuple[UnsupportedRouteDiagnostic, ...]:
    benchmark_ids = _benchmark_ids(payload)
    return (
        UnsupportedRouteDiagnostic(
            reason=UnsupportedReason.UNSUPPORTED_BENCHMARK,
            field="benchmark_ids",
            value=",".join(benchmark_ids) or "<missing>",
            supported=tuple(sorted(_EXECUTABLE_PUBLIC_BENCHMARKS)),
            message=(
                "FEM backend solve requires a validated public-synthetic executable benchmark; "
                "supported mathematical routes are execution-blocked until a fixture is registered."
            ),
        ),
    )


def _is_executable_public_synthetic_payload(payload: Mapping[str, Any]) -> bool:
    problem_id = _optional_string(payload.get("problem_id"))
    benchmark_ids = _benchmark_ids(payload)
    return bool(
        _optional_string(payload.get("privacy_class")) == "public_synthetic"
        and problem_id in _PUBLIC_SYNTHETIC_PROBLEM_IDS
        and (_EXECUTABLE_PUBLIC_BENCHMARKS & set(benchmark_ids))
        and _matches_public_fixture(problem_id, payload)
    )


def _matches_public_fixture(problem_id: str | None, payload: Mapping[str, Any]) -> bool:
    if problem_id != "pinares.fixed_price_option_proxy.v1":
        return False
    from finite_element_options.validation.pinares_fixed_price_proxy import (
        public_pinares_fixed_price_problem_spec,
    )

    try:
        supplied = json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        canonical = json.dumps(
            public_pinares_fixed_price_problem_spec(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError):
        return False
    return supplied == canonical


def _diagnostic_as_dict(diagnostic: UnsupportedRouteDiagnostic) -> dict[str, Any]:
    payload = asdict(diagnostic)
    payload["reason"] = diagnostic.reason.value
    return payload


def _benchmark_ids(payload: Mapping[str, Any]) -> tuple[str, ...]:
    ids: list[str] = []
    for section_name in ("artifact_manifest", "solver_plan", "result_bundle"):
        section = payload.get(section_name)
        if isinstance(section, Mapping):
            ids.extend(str(item) for item in _tuple(section.get("benchmark_ids")))
    ids.extend(str(item) for item in _tuple(payload.get("benchmark_ids")))
    financial_graph = payload.get("financial_graph")
    if isinstance(financial_graph, Mapping):
        valuation_graph = financial_graph.get("valuation_graph")
        if isinstance(valuation_graph, Mapping):
            solver_hints = valuation_graph.get("solver_hints")
            if isinstance(solver_hints, Mapping):
                ids.extend(
                    str(item) for item in _tuple(solver_hints.get("benchmark_ids"))
                )
    return tuple(dict.fromkeys(item for item in ids if item))


def _tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        return tuple(value.values())
    try:
        return tuple(value)
    except TypeError:
        return (value,)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


__all__ = [
    "ContractMajorMismatchError",
    "FEMBackendScreeningResult",
    "FEMBackendSolveResult",
    "FiniteElementHaircutBackend",
    "HAIRCUT_BACKEND_ENTRY_POINT",
    "HaircutProtocolUnavailableError",
    "create_backend",
]
