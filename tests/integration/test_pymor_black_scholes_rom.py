"""RED-GREEN tests for the optional pyMOR Black--Scholes ROM pilot."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Lock
from time import sleep

import numpy as np
import pytest
import scipy.stats as spst

pytestmark = pytest.mark.reduction
pytest.importorskip("pymor", reason="install finite-element-options[reduction]")

from finite_element_options.validation.evidence.reduced_order import (  # noqa: E402
    PymorBlackScholesConfig,
    ROMEnvelopeError,
    build_affine_black_scholes_system,
    train_pymor_rom,
)
from finite_element_options.validation.evidence.reduced_order import (  # noqa: E402
    pymor_adapter,
)
from finite_element_options.validation.evidence.reduced_order.assembly import (  # noqa: E402
    space_solver,
)


def smoke_config() -> PymorBlackScholesConfig:
    """Return a small deterministic case for mathematical integration tests."""

    return PymorBlackScholesConfig(
        refinement_level=6,
        time_steps=48,
        training_volatilities=(0.10, 0.1625, 0.225, 0.2875, 0.35),
        holdout_volatilities=(0.1375, 0.2375, 0.3375),
        snapshot_stride=4,
        max_basis_size=24,
        pod_rtol=1.0e-9,
        benchmark_repeats=3,
        benchmark_warmups=1,
        price_abs_tolerance=1.0e-7,
        delta_abs_tolerance=5.0e-6,
        gamma_abs_tolerance=5.0e-5,
    )


def test_config_is_hash_bound_and_rejects_overlapping_or_invalid_domains() -> None:
    config = smoke_config()
    assert config.input_hash == config.input_hash
    assert config.volatility_min == 0.10
    assert config.volatility_max == 0.35
    with pytest.raises(ValueError, match="disjoint"):
        PymorBlackScholesConfig(
            training_volatilities=(0.10, 0.20, 0.30),
            holdout_volatilities=(0.15, 0.20, 0.25),
        )
    with pytest.raises(ValueError, match="inside the declared envelope"):
        PymorBlackScholesConfig(holdout_volatilities=(0.08, 0.20, 0.30))
    with pytest.raises(ValueError, match="both declared envelope bounds"):
        PymorBlackScholesConfig(training_volatilities=(0.12, 0.20, 0.30))
    with pytest.raises(ValueError, match="positive benchmark controls"):
        replace(config, fom_oracle_gamma_tolerance=float("inf"))
    with pytest.raises(ValueError, match="rate must be finite"):
        replace(config, rate=float("nan"))
    with pytest.raises(ValueError, match="asymptotic call boundary"):
        replace(config, domain_max=1.0, spot=0.5)
    with pytest.raises(ValueError, match="asymptotic call boundary"):
        replace(config, rate=-0.5, domain_max=1.2, spot=0.5)
    with pytest.raises(ValueError, match="fixed 10x policy"):
        replace(config, minimum_online_speedup=9.99)
    with pytest.raises(ValueError, match="fixed 1000-query policy"):
        replace(config, maximum_ten_x_amortization_solves=1001)


def test_affine_operator_reconstructs_direct_fem_assembly() -> None:
    config = smoke_config()
    system = build_affine_black_scholes_system(config)
    for volatility in (0.10, 0.173, 0.35):
        direct = system.assemble_direct_operator(volatility)
        affine = system.assemble_affine_operator(volatility)
        numerator = np.linalg.norm((direct - affine).toarray())
        denominator = np.linalg.norm(direct.toarray())
        assert numerator / denominator <= config.affine_relative_tolerance
    assert system.boundary_policy == "volatility-independent-asymptotic-call"
    full_price_weights = np.zeros(system.full_dofs)
    full_price_weights[system.interior] = system.output_weights[0]
    full_price_weights[[system.left_boundary, system.right_boundary]] = (
        system.output_boundary_weights[0]
    )
    expected_probe = np.asarray(
        space_solver(config, 0.2).Vh.probes(np.array([[config.spot]])).toarray()
    ).ravel()
    assert full_price_weights == pytest.approx(expected_probe)


def test_nonunit_maturity_gamma_oracle_includes_sqrt_time() -> None:
    """The analytical Gamma contract must remain correct away from T=1."""

    config = replace(smoke_config(), maturity=0.25)
    system = build_affine_black_scholes_system(config)
    sigma = 0.2
    observed = system.analytical_outputs(sigma).gamma
    d1 = (
        np.log(config.spot / config.strike) + (config.rate + 0.5 * sigma * sigma) * config.maturity
    ) / (sigma * np.sqrt(config.maturity))
    expected = spst.norm.pdf(d1) / (config.spot * sigma * np.sqrt(config.maturity))
    assert observed == pytest.approx(expected, rel=1.0e-13)


def test_boundary_outputs_and_cached_fom_match_cold_reference() -> None:
    """Eliminated boundary values must contribute to outputs near the right edge."""

    config = replace(
        smoke_config(),
        domain_max=1.05,
        spot=1.0,
        greek_bump=0.02,
        refinement_level=4,
    )
    system = build_affine_black_scholes_system(config)
    cold = system.solve_full_order(0.2)
    cached = system.prepare_full_order(0.2).solve()
    boundary = np.array(
        [0.0, config.domain_max - config.strike * np.exp(-config.rate * config.maturity)]
    )
    expected = (
        system.output_weights @ cold.final_interior + system.output_boundary_weights @ boundary
    )
    assert np.any(system.output_boundary_weights[:, 1] != 0.0)
    observed = (cold.outputs.price, cold.outputs.delta, cold.outputs.gamma)
    assert observed == pytest.approx(expected)
    assert cached.outputs.to_dict() == pytest.approx(cold.outputs.to_dict())
    assert cached.final_interior == pytest.approx(cold.final_interior)


def test_pymor_pod_galerkin_matches_full_order_price_and_greeks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pymor.core.cache as pymor_cache

    cache_disabled = False
    cache_restored = False
    original_disable = pymor_cache.disable_caching
    original_enable = pymor_cache.enable_caching

    def record_disable() -> None:
        nonlocal cache_disabled
        cache_disabled = True
        original_disable()

    def record_enable() -> None:
        nonlocal cache_restored
        cache_restored = True
        original_enable()

    monkeypatch.setattr(pymor_cache, "disable_caching", record_disable)
    monkeypatch.setattr(pymor_cache, "enable_caching", record_enable)
    config = smoke_config()
    system = build_affine_black_scholes_system(config)
    trained = train_pymor_rom(system, config)
    assert cache_disabled
    assert cache_restored
    assert trained.library == "pymor"
    assert trained.basis_size <= config.max_basis_size
    assert trained.basis_size > 0
    assert trained.full_dofs > trained.basis_size
    assert trained.projection.basis.shape == (system.interior_dofs, trained.basis_size)
    assert trained.projection.singular_values.shape == (trained.basis_size,)
    assert not hasattr(trained, "system")
    for volatility in config.holdout_volatilities:
        fom = system.solve_full_order(volatility)
        rom = trained.solve_with_diagnostics(volatility)
        assert fom.residual_linf <= config.linear_residual_tolerance
        assert rom.residual_linf <= config.linear_residual_tolerance
        assert fom.linear_solves == rom.linear_solves == config.time_steps
        assert fom.operator_nnz > 0
        assert abs(rom.outputs.price - fom.outputs.price) <= config.price_abs_tolerance
        assert abs(rom.outputs.delta - fom.outputs.delta) <= config.delta_abs_tolerance
        assert abs(rom.outputs.gamma - fom.outputs.gamma) <= config.gamma_abs_tolerance


def test_pymor_adapter_preserves_preexisting_disabled_cache_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Security scoping must not re-enable a caller-disabled pyMOR cache."""

    import pymor.core.cache as pymor_cache

    monkeypatch.delenv("PYMOR_CACHE_DISABLE", raising=False)
    pymor_cache.disable_caching()
    try:
        config = smoke_config()
        train_pymor_rom(build_affine_black_scholes_system(config), config)
        assert getattr(pymor_cache, "_caching_disabled") is True
    finally:
        pymor_cache.enable_caching()


def test_pymor_cache_scope_serializes_concurrent_calls() -> None:
    """Concurrent adapter calls must not interleave process-global cache state."""

    state_lock = Lock()
    active = 0
    maximum_active = 0

    @pymor_adapter._without_persisted_pymor_cache  # type: ignore[attr-defined]
    def guarded_probe() -> None:
        nonlocal active, maximum_active
        with state_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        sleep(0.02)
        with state_lock:
            active -= 1

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(guarded_probe) for _ in range(2)]
        for future in futures:
            future.result()
    assert maximum_active == 1


def test_rom_refuses_outside_envelope_and_names_full_order_fallback() -> None:
    config = smoke_config()
    trained = train_pymor_rom(build_affine_black_scholes_system(config), config)
    with pytest.raises(ROMEnvelopeError) as exc_info:
        trained.solve(0.08)
    assert exc_info.value.reason == "parameter_out_of_envelope"
    assert exc_info.value.fallback == "full_order_fem"
    with pytest.raises(ROMEnvelopeError):
        trained.solve(0.40)
