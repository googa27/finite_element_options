"""QuantLib vanilla/quanto oracle tests for adoption issue #132."""

from __future__ import annotations

from datetime import date
import hashlib
import json
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ORACLE = "finite_element_options.examples.regime_switching_quanto.adoption.quantlib_oracle"


def _spec(**overrides: object):
    from finite_element_options.examples.regime_switching_quanto.adoption.quantlib_oracle import (
        QuantLibOracleSpec,
    )

    params = {
        "evaluation_date": date(2026, 9, 4),
        "maturity_date": date(2027, 9, 4),
        "spot": 100.0,
        "strike": 105.0,
        "equity_vol": 0.20,
        "domestic_rate": 0.035,
        "foreign_rate": 0.015,
        "dividend_yield": 0.010,
        "fx_vol": 0.12,
        "correlation": 0.25,
        "fixed_fx": 850.0,
        "kind": "fixed_fx_quanto",
    }
    params.update(overrides)
    return QuantLibOracleSpec(**params)


def test_quantlib_oracle_contracts_are_json_safe_and_lazy_without_quantlib() -> None:
    """Contracts/facades import with QuantLib blocked and expose only plain data."""

    code = f"""
    import builtins
    import importlib
    import json
    import sys
    from datetime import date

    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name.split('.')[0] == 'QuantLib':
            raise ModuleNotFoundError('blocked optional dependency: QuantLib', name='QuantLib')
        return original_import(name, *args, **kwargs)
    builtins.__import__ = guarded_import

    facade = importlib.import_module({ORACLE!r})
    contracts = importlib.import_module({ORACLE + ".contracts"!r})
    spec = contracts.QuantLibOracleSpec(
        evaluation_date=date(2026, 9, 4), maturity_date=date(2027, 9, 4),
        spot=100.0, strike=100.0, equity_vol=0.2, domestic_rate=0.03,
        foreign_rate=0.03, dividend_yield=0.01, fx_vol=0.0,
        correlation=0.0, fixed_fx=1.0, kind='vanilla')
    json.dumps(spec.to_dict(), sort_keys=True, allow_nan=False)
    assert {ORACLE + ".adapter"!r} not in sys.modules
    assert {ORACLE + ".matrix"!r} not in sys.modules
    assert 'price_quantlib_oracle' in dir(facade)
    """
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src")},
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_quantlib_adapter_missing_extra_error_is_exact() -> None:
    """Executing the adapter without QuantLib raises the advertised extra hint."""

    code = f"""
    import importlib.abc
    import sys
    from datetime import date
    from {ORACLE} import QuantLibOracleSpec, price_quantlib_oracle

    class BlockQuantLib(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname == 'QuantLib' or fullname.startswith('QuantLib.'):
                raise ModuleNotFoundError('blocked optional dependency', name='QuantLib')
            return None

    sys.modules.pop('QuantLib', None)
    sys.meta_path.insert(0, BlockQuantLib())
    spec = QuantLibOracleSpec(
        evaluation_date=date(2026, 9, 4), maturity_date=date(2027, 9, 4),
        spot=100.0, strike=100.0, equity_vol=0.2, domestic_rate=0.03,
        foreign_rate=0.03, dividend_yield=0.01, fx_vol=0.0,
        correlation=0.0, fixed_fx=1.0, kind='vanilla')
    try:
        price_quantlib_oracle(spec)
    except ImportError as exc:
        message = str(exc)
        assert 'QuantLib' in message
        assert 'finite-element-options[quantlib]' in message
        assert 'QuantLib>=1.43,<2' in message
    else:
        raise AssertionError('expected missing-extra ImportError')
    """
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src")},
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize(
    "field,value,supported",
    [
        ("calendar", "UnitedStates", ("TARGET",)),
        ("day_count", "Actual360", ("Actual365Fixed",)),
        ("business_day_convention", "Following", ("Unadjusted",)),
        ("rate_compounding", "Compounded", ("Continuous",)),
        ("exercise", "American", ("European",)),
        ("option_type", "put", ("call",)),
        ("kind", "quanto_by_drift", ("vanilla", "fixed_fx_quanto")),
    ],
)
def test_unsupported_conventions_raise_typed_actionable_error(
    field: str, value: str, supported: tuple[str, ...]
) -> None:
    """Unsupported conventions fail with field/received/supported attributes."""

    from finite_element_options.examples.regime_switching_quanto.adoption.quantlib_oracle import (
        QuantLibConventionError,
    )

    with pytest.raises(QuantLibConventionError) as exc_info:
        _spec(**{field: value})
    error = exc_info.value
    assert error.field == field
    assert error.received == value
    assert error.supported == supported
    assert error.to_dict() == {"field": field, "received": value, "supported": list(supported)}


@pytest.mark.parametrize(
    "field,value,expected",
    [
        ("foreign_rate", 0.031, 0.030),
        ("fx_vol", 0.12, 0.0),
        ("correlation", -0.25, 0.0),
        ("fixed_fx", 850.0, 1.0),
    ],
)
def test_vanilla_reduction_invariants_raise_typed_actionable_error(
    field: str, value: float, expected: float
) -> None:
    """Vanilla reductions reject inconsistent quanto/Fx fields instead of overriding them."""

    from finite_element_options.examples.regime_switching_quanto.adoption import (
        QuantLibReductionError as AdoptionReductionError,
    )
    from finite_element_options.examples.regime_switching_quanto.adoption.quantlib_oracle import (
        QuantLibReductionError,
    )

    assert AdoptionReductionError is QuantLibReductionError
    params = {
        "kind": "vanilla",
        "domestic_rate": 0.030,
        "foreign_rate": 0.030,
        "fx_vol": 0.0,
        "correlation": 0.0,
        "fixed_fx": 1.0,
        field: value,
    }
    with pytest.raises(QuantLibReductionError) as exc_info:
        _spec(**params)
    error = exc_info.value
    assert error.field == field
    assert error.received == value
    assert error.expected == expected
    assert error.kind == "vanilla"
    assert error.to_dict() == {
        "field": field,
        "received": value,
        "expected": expected,
        "kind": "vanilla",
    }
    assert field in str(error)
    assert "vanilla" in str(error)


def test_reduction_validation_runs_before_quantlib_state_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even mutated specs fail reduction checks before entering QuantLib global state."""

    from finite_element_options.examples.regime_switching_quanto.adoption.quantlib_oracle import (
        QuantLibReductionError,
    )
    from finite_element_options.examples.regime_switching_quanto.adoption.quantlib_oracle import (
        adapter,
    )

    spec = _spec(
        kind="vanilla",
        domestic_rate=0.030,
        foreign_rate=0.030,
        fx_vol=0.0,
        correlation=0.0,
        fixed_fx=1.0,
    )
    object.__setattr__(spec, "fx_vol", 0.12)

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("QuantLib state was mutated before reduction validation")

    monkeypatch.setattr(adapter, "quantlib_evaluation_date", fail_if_called)
    with pytest.raises(QuantLibReductionError) as exc_info:
        adapter.price_quantlib_oracle(spec)
    assert exc_info.value.field == "fx_vol"


@pytest.mark.parametrize(
    "spec",
    [
        _spec(kind="vanilla", fixed_fx=1.0, foreign_rate=0.035, fx_vol=0.0, correlation=0.0),
        _spec(
            kind="vanilla",
            fixed_fx=1.0,
            spot=95.0,
            strike=105.0,
            equity_vol=0.25,
            domestic_rate=0.041,
            foreign_rate=0.041,
            dividend_yield=0.004,
            fx_vol=0.0,
            correlation=0.0,
            maturity_date=date(2027, 4, 15),
        ),
        _spec(correlation=0.35),
        _spec(
            spot=120.0,
            strike=110.0,
            equity_vol=0.28,
            domestic_rate=0.025,
            foreign_rate=0.045,
            dividend_yield=0.020,
            fx_vol=0.18,
            correlation=-0.45,
            fixed_fx=780.0,
            evaluation_date=date(2026, 11, 2),
            maturity_date=date(2027, 8, 2),
        ),
    ],
)
def test_real_quantlib_prices_match_repository_analytical_reduction(spec: object) -> None:
    """QuantLib engines price vanilla and QuantoVanillaOption reductions independently."""

    pytest.importorskip("QuantLib")
    from finite_element_options.examples.regime_switching_quanto.adoption.quantlib_oracle import (
        price_quantlib_oracle,
    )

    result = price_quantlib_oracle(spec)
    assert result.analytical_passed
    assert result.analytical_absolute_error <= 1.0e-9
    assert result.year_fraction > 0.0
    if result.spec["kind"] == "fixed_fx_quanto":
        assert result.price == pytest.approx(result.quantlib_npv * result.spec["fixed_fx"])
        assert result.effective_dividend_yield == pytest.approx(
            result.spec["dividend_yield"]
            + result.spec["domestic_rate"]
            - result.spec["foreign_rate"]
            + result.quanto_adjustment
        )
    else:
        assert result.quanto_adjustment == 0.0
        assert result.effective_dividend_yield == result.spec["dividend_yield"]


def test_unadjusted_convention_uses_explicit_maturity_date() -> None:
    """Unadjusted maturity dates remain exact even on TARGET non-business days."""

    QuantLib = pytest.importorskip("QuantLib")
    from finite_element_options.examples.regime_switching_quanto.adoption.quantlib_oracle import (
        price_quantlib_oracle,
    )

    spec = _spec(
        kind="vanilla",
        evaluation_date=date(2026, 9, 4),
        maturity_date=date(2027, 9, 4),  # Saturday; Unadjusted must not roll to Monday.
        domestic_rate=0.030,
        foreign_rate=0.030,
        fx_vol=0.0,
        correlation=0.0,
        fixed_fx=1.0,
    )
    result = price_quantlib_oracle(spec)
    adjusted = QuantLib.TARGET().adjust(
        QuantLib.Date(4, QuantLib.September, 2027), QuantLib.Unadjusted
    )
    assert adjusted == QuantLib.Date(4, QuantLib.September, 2027)
    assert result.spec["maturity_date"] == "2027-09-04"


def test_quantlib_evaluation_date_accepts_stdlib_date_and_restores_after_exception() -> None:
    """The global QuantLib evaluation date is restored for stdlib dates on failures."""

    QuantLib = pytest.importorskip("QuantLib")
    from finite_element_options.examples.regime_switching_quanto.adoption.quantlib_state import (
        quantlib_evaluation_date,
    )

    settings = QuantLib.Settings.instance()
    original = settings.evaluationDate
    expected = QuantLib.Date(4, QuantLib.September, 2026)
    with pytest.raises(RuntimeError, match="inside state context"):
        with quantlib_evaluation_date(date(2026, 9, 4)):
            assert settings.evaluationDate == expected
            raise RuntimeError("inside state context")
    assert settings.evaluationDate == original


def test_quantlib_evaluation_date_serializes_two_threads_without_overlap() -> None:
    """Two reference calls cannot overlap process-global QuantLib dates."""

    QuantLib = pytest.importorskip("QuantLib")
    from finite_element_options.examples.regime_switching_quanto.adoption.quantlib_state import (
        quantlib_evaluation_date,
    )

    settings = QuantLib.Settings.instance()
    original = settings.evaluationDate
    first = date(2026, 9, 4)
    second = date(2026, 9, 5)
    first_ql = QuantLib.Date(4, QuantLib.September, 2026)
    second_ql = QuantLib.Date(5, QuantLib.September, 2026)
    entered_first = threading.Event()
    release_first = threading.Event()
    events: list[tuple[str, float]] = []
    failures: list[BaseException] = []

    def worker_one() -> None:
        try:
            with quantlib_evaluation_date(first):
                events.append(("first_enter", time.monotonic()))
                entered_first.set()
                assert settings.evaluationDate == first_ql
                assert release_first.wait(5.0)
                assert settings.evaluationDate == first_ql
                events.append(("first_exit", time.monotonic()))
        except BaseException as exc:  # pragma: no cover - thread handoff
            failures.append(exc)

    def worker_two() -> None:
        try:
            assert entered_first.wait(5.0)
            with quantlib_evaluation_date(second):
                events.append(("second_enter", time.monotonic()))
                assert settings.evaluationDate == second_ql
                events.append(("second_exit", time.monotonic()))
        except BaseException as exc:  # pragma: no cover - thread handoff
            failures.append(exc)

    t1 = threading.Thread(target=worker_one)
    t2 = threading.Thread(target=worker_two)
    t1.start()
    t2.start()
    entered_first.wait(5.0)
    time.sleep(0.05)
    assert [name for name, _ in events] == ["first_enter"]
    release_first.set()
    t1.join(5.0)
    t2.join(5.0)
    assert not failures
    assert [name for name, _ in events] == [
        "first_enter",
        "first_exit",
        "second_enter",
        "second_exit",
    ]
    assert settings.evaluationDate == original


def test_canonical_artifact_scope_and_hash_regression() -> None:
    """The committed artifact is canonical and records the one-regime scope hash."""

    from finite_element_options.examples.regime_switching_quanto.adoption.quantlib_oracle import (
        SCOPE_STATEMENT,
        canonical_matrix_input_hash,
    )
    from finite_element_options.examples.regime_switching_quanto.adoption.evidence_io import (
        canonical_json,
    )

    path = ROOT / "docs" / "evidence" / "regime_switching_quanto_quantlib_oracle_2026-09-04.json"
    artifact_text = path.read_text(encoding="utf-8")
    assert hashlib.sha256(artifact_text.encode("utf-8")).hexdigest() == (
        "ca2789e8f686a2f25b9abebc076f18ce7596673b038e52b681478cad22c4a056"
    )
    artifact = json.loads(artifact_text)
    assert artifact_text == canonical_json(artifact) + "\n"
    assert artifact["matrix_spec_hash"] == canonical_matrix_input_hash()
    assert artifact["summary"]["case_count"] == 4
    assert artifact["summary"]["vanilla_case_count"] >= 2
    assert artifact["summary"]["quanto_case_count"] >= 2
    assert artifact["summary"]["all_passed"] is True
    assert artifact["scope"] == SCOPE_STATEMENT
    assert "not evidence for the full multi-regime" in artifact["scope"]


def test_artifact_fem_mc_gates_are_seeded_and_account_for_standard_error() -> None:
    """Artifact gates compare FEM and seeded MC with explicit tolerances."""

    path = ROOT / "docs" / "evidence" / "regime_switching_quanto_quantlib_oracle_2026-09-04.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))
    seeds = set()
    for row in artifact["cases"]:
        gates = row["gates"]
        assert gates["quantlib_analytical_passed"] is True
        assert gates["fem_passed"] is True
        assert gates["mc_passed"] is True
        assert gates["all_passed"] is True
        assert row["errors"]["fem_vs_analytical_abs"] <= row["tolerances"]["fem_vs_analytical_abs"]
        assert row["mc_result"]["standard_error"] > 0.0
        assert row["mc_result"]["paths"] >= 90_000
        seeds.add(row["mc_result"]["seed"])
        assert row["errors"]["mc_vs_analytical_abs"] <= row["tolerances"]["mc_vs_analytical_abs"]
        assert row["tolerances"]["mc_vs_analytical_abs"] >= (
            row["tolerances"]["mc_standard_error_multiplier"] * row["mc_result"]["standard_error"]
        )
    assert seeds == {132_001, 132_002, 132_003, 132_004}
    assert len(artifact["cases"]) == 4


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"case_id": ""}, "case_id"),
        ({"mc_paths": 1}, "mc_paths"),
        ({"mc_steps_per_year": 0}, "mc_steps_per_year"),
        ({"fem_abs_tolerance": -0.1}, "fem_abs_tolerance"),
        ({"mc_abs_floor": float("nan")}, "mc_abs_floor"),
        ({"mc_standard_error_multiplier": 0.0}, "mc_standard_error_multiplier"),
    ],
)
def test_matrix_case_validation_rejects_actionably(kwargs: dict[str, object], message: str) -> None:
    """Matrix cases validate IDs, paths, steps, and tolerances before execution."""

    from finite_element_options.examples.regime_switching_quanto import FEMGridSpec
    from finite_element_options.examples.regime_switching_quanto.adoption.quantlib_oracle import (
        MatrixCase,
    )

    params = {
        "case_id": "valid_case",
        "spec": _spec(
            kind="vanilla",
            domestic_rate=0.030,
            foreign_rate=0.030,
            fx_vol=0.0,
            correlation=0.0,
            fixed_fx=1.0,
        ),
        "grid": FEMGridSpec((-1.0, 1.0), (-0.5, 0.5), nx=5, ny=5, time_steps=2),
        "mc_paths": 2,
        "mc_seed": 1,
        "mc_steps_per_year": 1,
        "fem_abs_tolerance": 0.0,
        "mc_abs_floor": 0.0,
        "mc_standard_error_multiplier": 1.0,
    }
    params.update(kwargs)
    with pytest.raises(ValueError, match=message):
        MatrixCase(**params)


def test_empty_quantlib_matrix_rejected_before_summary_reductions() -> None:
    """Explicit empty matrices fail actionably instead of crashing in max()."""

    from finite_element_options.examples.regime_switching_quanto.adoption.quantlib_oracle import (
        run_quantlib_oracle_matrix,
    )

    with pytest.raises(ValueError, match="at least one"):
        run_quantlib_oracle_matrix(())
