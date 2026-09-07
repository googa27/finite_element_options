"""End-to-end DYNAMAX, NumPyro, and Diffrax regime-study orchestration."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
from typing import Any

from .contracts import (
    HMMParameterSummary,
    JaxRegimeStudyConfig,
    PosteriorDiagnosticSummary,
    PUBLICATION_MAX_RHAT,
    PUBLICATION_MIN_CHAINS,
    PUBLICATION_MIN_DRAWS_PER_CHAIN,
    PUBLICATION_MIN_ESS,
    PromotionDecision,
    SCHEMA_VERSION,
)
from .data import _load_pdp_snapshot
from .hmm.dynamax_adapter import dynamax_marginal_log_prob
from .hmm.experiment import _candidate_comparison, _full_fit, _synthetic_recovery
from .hmm.forward import gaussian_hmm_filter_probs, gaussian_hmm_log_prob
from .hmm.generator_check import check_ctmc_generator
from .hmm.numpyro_model import run_numpyro_hmm
from .hmm.statsmodels_baseline import fit_statsmodels_var_baseline
from .pricing.diffrax_sde import simulate_diffrax_terminal_states
from .pricing.exact import draw_paths_and_increments, simulate_exact_terminal_states
from .pricing.study import (
    _contracts,
    _historical_prices,
    _matched_historical_oracle,
    _one_state_oracles,
    _posterior_price_intervals,
    _price_contracts,
    _risk_neutral_coefficients,
)
from .pricing.validation import martingale_checks, refinement_invariance, strike_monotonicity
from .prior_sensitivity import run_prior_sensitivity
from .utils import stack as _stack, to_python as _python


def run_jax_regime_study(
    archive: str | Path,
    *,
    config: JaxRegimeStudyConfig | None = None,
) -> dict[str, Any]:
    """Execute the bounded JAX-native regime calibration and pricing study."""

    config = config or JaxRegimeStudyConfig()
    _jax_module, jnp, jr = _stack()
    batch, archive_snapshot = _load_pdp_snapshot(archive)
    observations = jnp.asarray(batch.returns) * 100.0
    train_count = int(len(observations) * (1.0 - config.holdout_fraction))
    candidates, _training_fit = _candidate_comparison(observations, train_count, config)
    statsmodels_baseline = fit_statsmodels_var_baseline(observations, train_count)
    full_fit = _full_fit(observations, config)
    parameters = full_fit["parameters"]
    ours = gaussian_hmm_log_prob(
        observations,
        parameters["initial_probs"],
        parameters["transition_matrix"],
        parameters["means"],
        parameters["covariances"],
    )
    theirs = dynamax_marginal_log_prob(
        observations,
        parameters["initial_probs"],
        parameters["transition_matrix"],
        parameters["means"],
        parameters["covariances"],
    )
    likelihood_parity_error = abs(float(ours - theirs))
    posterior = run_numpyro_hmm(
        observations,
        initial=parameters["initial_probs"],
        transition=parameters["transition_matrix"],
        means=parameters["means"],
        covariances=parameters["covariances"],
        seed=config.seed + 500,
        warmup=config.warmup,
        samples=config.posterior_samples,
        chains=config.chains,
    )
    posterior_mean = posterior["posterior_mean"]
    filtered = gaussian_hmm_filter_probs(
        observations,
        posterior_mean["initial_probs"],
        posterior_mean["transition_matrix"],
        posterior_mean["means"],
        posterior_mean["covariances"],
    )
    current_probs = filtered[-1]
    drift, diffusion, equity_vol, fx_vol, correlation = _risk_neutral_coefficients(
        posterior_mean["covariances"], config
    )
    steps = config.pricing_steps
    regimes, increments = draw_paths_and_increments(
        jr.key(config.seed + 600),
        current_probs,
        posterior_mean["transition_matrix"],
        paths=config.pricing_paths,
        steps=steps,
        maturity=config.maturity_years,
    )
    exact_states = simulate_exact_terminal_states(
        regimes, increments, drift, diffusion, config.maturity_years
    )
    diffrax_states = simulate_diffrax_terminal_states(
        regimes, increments, drift, diffusion, config.maturity_years
    )
    pathwise_error = float(jnp.max(jnp.abs(exact_states - diffrax_states)))
    contracts = _contracts(batch.equity_spot, batch.fx_spot)
    point_prices = _price_contracts(
        diffrax_states,
        contracts,
        equity_spot=batch.equity_spot,
        fx_spot=batch.fx_spot,
        config=config,
    )
    martingale = martingale_checks(
        diffrax_states,
        equity_spot=batch.equity_spot,
        fx_spot=batch.fx_spot,
        config=config,
    )
    monotonicity = strike_monotonicity(
        diffrax_states,
        equity_spot=batch.equity_spot,
        fx_spot=batch.fx_spot,
        config=config,
    )
    refinement = refinement_invariance(
        regimes,
        increments,
        drift,
        diffusion,
        config.maturity_years,
    )
    one_state = _one_state_oracles(
        increments,
        drift,
        diffusion,
        equity_vol,
        fx_vol,
        correlation,
        contracts,
        equity_spot=batch.equity_spot,
        fx_spot=batch.fx_spot,
        config=config,
    )
    intervals = _posterior_price_intervals(
        posterior,
        observations,
        contracts,
        equity_spot=batch.equity_spot,
        fx_spot=batch.fx_spot,
        config=config,
    )
    matched_historical = _matched_historical_oracle(
        archive_snapshot,
        contracts,
        equity_spot=batch.equity_spot,
        fx_spot=batch.fx_spot,
        config=config,
    )
    prior_sensitivity = run_prior_sensitivity(
        observations,
        parameters,
        posterior,
        contracts,
        equity_spot=batch.equity_spot,
        fx_spot=batch.fx_spot,
        config=config,
    )
    posterior_uncertainty_separated = all(
        row["posterior_draws"] >= 128
        and row["posterior_draws_per_chain"] >= 64
        and row["paths_per_draw"] >= 131_072
        and row["common_random_numbers"]
        and row["mc_se_to_posterior_half_width"] <= 0.5
        and row["split_subsample_max_quantile_delta_to_full_half_width"] <= 0.5
        for row in intervals.values()
    )
    synthetic = _synthetic_recovery(config)
    ctmc = check_ctmc_generator(posterior_mean["transition_matrix"])
    selected_candidate = max(candidates, key=lambda row: row["heldout_mean_log_score"])
    selected_states = int(selected_candidate["states"])
    beats_statsmodels = (
        selected_candidate["heldout_mean_log_score"]
        > statsmodels_baseline["heldout_mean_log_score"]
    )
    three_state = next(row for row in candidates if row["states"] == 3)
    four_state = next(row for row in candidates if row["states"] == 4)
    four_over_three_gain = (
        four_state["heldout_mean_log_score"] - three_state["heldout_mean_log_score"]
    )
    three_state_competitive = four_over_three_gain <= 0.01
    hmm_summary = HMMParameterSummary(
        initial_probs=tuple(float(value) for value in parameters["initial_probs"]),
        transition_matrix=tuple(
            tuple(float(value) for value in row) for row in parameters["transition_matrix"]
        ),
        means_percent_daily=tuple(
            tuple(float(value) for value in row) for row in parameters["means"]
        ),
        covariances_percent_squared_daily=tuple(
            tuple(tuple(float(value) for value in row) for row in matrix)
            for matrix in parameters["covariances"]
        ),
    )
    posterior_diagnostics = PosteriorDiagnosticSummary(
        chains=posterior["chains"],
        draws_per_chain=posterior["draws_per_chain"],
        divergences=posterior["divergences"],
        maximum_rhat=posterior["maximum_rhat"],
        minimum_ess=posterior["minimum_ess"],
        finite=posterior["finite"],
    )
    em_converged = all(bool(row["multistart_converged"]) for row in candidates)
    full_em_converged = bool(full_fit["multistart_converged"])
    gates = {
        "heldout_model_selection": selected_states == config.num_states,
        "selected_model_beats_statsmodels_var": beats_statsmodels,
        "candidate_multistart_em_convergence": em_converged,
        "full_fit_multistart_em_convergence": full_em_converged,
        "statsmodels_var_baseline_finite": bool(statsmodels_baseline["finite"]),
        "ctmc_generator_laws": bool(ctmc["passed"]),
        "dynamax_jax_likelihood_parity": likelihood_parity_error <= 1.0e-8,
        "diffrax_exact_pathwise_parity": pathwise_error <= 1.0e-10,
        "diffrax_brownian_bridge_refinement": bool(refinement["passed"]),
        "risk_neutral_martingales": bool(martingale["passed"]),
        "strike_monotonicity": bool(monotonicity["passed"]),
        "matched_historical_jax_numpy_mc": bool(matched_historical["passed"]),
        "posterior_vs_mc_uncertainty_separated": posterior_uncertainty_separated,
        "prior_sensitivity_diagnostics": bool(prior_sensitivity["passed"]),
        "numpyro_two_chains": posterior["chains"] >= PUBLICATION_MIN_CHAINS,
        "numpyro_draws_per_chain": (
            posterior["draws_per_chain"] >= PUBLICATION_MIN_DRAWS_PER_CHAIN
        ),
        "numpyro_finite": bool(posterior["finite"]),
        "numpyro_zero_divergence": posterior["divergences"] == 0,
        "numpyro_rhat": posterior["maximum_rhat"] is not None
        and posterior["maximum_rhat"] <= PUBLICATION_MAX_RHAT,
        "numpyro_ess": posterior["minimum_ess"] is not None
        and posterior["minimum_ess"] >= PUBLICATION_MIN_ESS,
        "one_state_analytic": all(bool(row["passed_5se"]) for row in one_state.values()),
        "synthetic_recovery": bool(synthetic["passed"]),
    }
    promotion = PromotionDecision(tuple(sorted(gates.items())))
    diagnostics_passed = promotion.passed
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if diagnostics_passed else "failed",
        "claims": {
            "research_only": True,
            "market_calibrated": False,
            "production_ready": False,
            "measure_policy": (
                "DYNAMAX/NumPyro estimate historical P dynamics; pricing reuses the P transition "
                "matrix under a constrained domestic Qd scenario with no option-implied regime-risk premium."
            ),
            "diffrax_scope": (
                "Diffrax Euler is a tested SDE abstraction, not an accuracy upgrade; the aligned "
                "piecewise-constant log diffusion is conditionally exact under the JAX scan oracle."
            ),
            "regime_timing_scope": (
                "Pricing uses JAX-native daily discrete-HMM regime paths. SciPy CTMC projection is "
                "an embeddability/law diagnostic only; no continuous-time regime simulation is claimed."
            ),
            "inference_scope": (
                "NumPyro inference is an empirical-Bayes analysis conditional on volatility-ordered "
                "DYNAMAX EM reference parameters; intervals do not include uncertainty from selecting "
                "or estimating that reference."
            ),
        },
        "config": config.to_dict(),
        "dependencies": {
            name: version(name)
            for name in (
                "jax",
                "jaxlib",
                "numpyro",
                "dynamax",
                "diffrax",
                "statsmodels",
                "fastprogress",
                "tfp-nightly",
            )
        },
        "data": batch.to_dict(),
        "hmm": {
            "candidate_comparison": candidates,
            "statsmodels_var_baseline": statsmodels_baseline,
            "selection": {
                "states": config.num_states,
                "selected_by_heldout_score": selected_states,
                "rule": (
                    "highest chronological held-out mean log score after selecting the best finite "
                    "converged fit from three deterministic 250-iteration starts; each HMM "
                    "candidate requires at least two converged starts"
                ),
                "four_over_three_mean_log_score_gain": four_over_three_gain,
                "three_state_practically_competitive_at_0_01": three_state_competitive,
                "statsmodels_var_heldout_mean_log_score": statsmodels_baseline[
                    "heldout_mean_log_score"
                ],
                "selected_model_beats_statsmodels_var": beats_statsmodels,
            },
            "dynamax_full_fit": {
                **hmm_summary.to_dict(),
                "current_filtered_probs": _python(full_fit["filtered_probs"][-1]),
                "occupancy": _python(jnp.mean(full_fit["smoothed_probs"], axis=0)),
                "annualized_composite_volatility_percent": _python(
                    full_fit["annualized_composite_volatility"]
                ),
                "marginal_log_likelihood": float(full_fit["marginal_log_likelihood"]),
                "minimum_em_increment": float(full_fit["minimum_em_increment"]),
                "final_em_increment": float(full_fit["final_em_increment"]),
                "converged_starts": int(full_fit["converged_starts"]),
                "minimum_converged_starts_required": int(
                    full_fit["minimum_converged_starts_required"]
                ),
                "multistart_converged": bool(full_fit["multistart_converged"]),
                "all_starts_converged": bool(full_fit["all_starts_converged"]),
                "start_diagnostics": full_fit["start_diagnostics"],
                "em_iterations": max(config.em_iters, 250),
                "starts": 3,
            },
            "marginal_likelihood_parity_absolute_error": likelihood_parity_error,
            "numpyro": {
                **posterior_diagnostics.to_dict(),
                "prior": posterior["prior"],
                "worst_rhat": posterior["worst_rhat"],
                "worst_ess": posterior["worst_ess"],
                "weakly_identified_parameters": posterior["weakly_identified_parameters"],
                "finite": posterior["finite"],
                "diagnostic_fields": posterior["diagnostic_fields"],
                "state_treatment": "analytically marginalized with a JAX log-space forward scan",
                "identification": (
                    "empirical-Bayes reference-identified priors conditional on volatility-ordered "
                    "DYNAMAX EM states"
                ),
                "posterior_mean_initial_probs": _python(posterior_mean["initial_probs"]),
                "posterior_mean_transition_matrix": _python(posterior_mean["transition_matrix"]),
                "posterior_mean_covariances_percent_squared_daily": _python(
                    posterior_mean["covariances"]
                ),
                "posterior_current_filtered_probs": _python(current_probs),
            },
            "synthetic_recovery": synthetic,
            "prior_sensitivity": prior_sensitivity,
            "ctmc_generator": ctmc,
        },
        "pricing": {
            "model": {
                "annualized_equity_volatility": _python(equity_vol),
                "annualized_fx_volatility": _python(fx_vol),
                "correlation": _python(correlation),
                "domestic_log_drifts": _python(drift),
            },
            "diffrax": {
                "solver": "Euler",
                "stepsize_controller": "StepTo at every regime boundary",
                "paths": config.pricing_paths,
                "steps": steps,
                "maximum_pathwise_error_vs_exact_jax": pathwise_error,
                "point_prices": point_prices,
            },
            "posterior_parameter_price_intervals": intervals,
            "martingale_checks": martingale,
            "strike_monotonicity": monotonicity,
            "brownian_bridge_refinement": refinement,
            "one_state_analytical_oracles": one_state,
            "matched_historical_three_state_jax_numpy_oracle": matched_historical,
            "historical_scikit_fem_and_numpy_mc": _historical_prices(archive_snapshot),
        },
        "verification": promotion.to_dict(),
    }
    return _python(evidence)
