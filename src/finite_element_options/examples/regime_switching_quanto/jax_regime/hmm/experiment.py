"""Candidate comparison, multi-start fitting, and synthetic HMM recovery."""

from __future__ import annotations

from itertools import permutations
import math
from typing import Any

from ..contracts import JaxRegimeStudyConfig
from ..utils import stack as _stack, to_python as _python
from .dynamax_adapter import fit_dynamax_hmm, sample_dynamax_hmm
from .forward import conditional_holdout_log_prob

_EM_CONVERGENCE_TOLERANCE = 1.0e-3


def _select_converged_multistart(
    fits: list[dict[str, Any]], seeds: list[int]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select the best converged finite fit while preserving every start outcome."""

    diagnostics: list[dict[str, Any]] = []
    eligible: list[tuple[int, dict[str, Any]]] = []
    for index, (fit, seed) in enumerate(zip(fits, seeds, strict=True)):
        likelihood = float(fit["marginal_log_likelihood"])
        final_increment = float(fit["final_em_increment"])
        finite = (
            bool(fit["finite"]) and math.isfinite(likelihood) and math.isfinite(final_increment)
        )
        converged = finite and abs(final_increment) <= _EM_CONVERGENCE_TOLERANCE
        diagnostics.append(
            {
                "start_index": index,
                "seed": seed,
                "finite": finite,
                "em_converged": converged,
                "selected": False,
                "final_em_increment": final_increment,
                "minimum_em_increment": float(fit["minimum_em_increment"]),
                "train_log_likelihood": likelihood,
            }
        )
        if converged:
            eligible.append((index, fit))
    if not eligible:
        raise RuntimeError("no finite converged DYNAMAX fit was found across deterministic starts")
    selected_index, selected = max(
        eligible, key=lambda item: float(item[1]["marginal_log_likelihood"])
    )
    diagnostics[selected_index]["selected"] = True
    return selected, diagnostics


def _candidate_comparison(
    observations: Any, train_count: int, config: JaxRegimeStudyConfig
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _jax, jnp, _jr = _stack()
    train = observations[:train_count]
    mean = jnp.mean(train, axis=0, keepdims=True)
    covariance = jnp.cov(train.T)[None, :, :]
    baseline = conditional_holdout_log_prob(
        observations, train_count, jnp.ones(1), jnp.ones((1, 1)), mean, covariance
    )
    candidates: list[dict[str, Any]] = [
        {
            "engine": "closed_form_gaussian",
            "states": 1,
            "starts": 1,
            "converged_starts": 1,
            "minimum_converged_starts_required": 1,
            "multistart_converged": True,
            "all_starts_converged": True,
            "start_diagnostics": [],
            "heldout_log_likelihood": float(baseline),
            "heldout_mean_log_score": float(baseline) / (len(observations) - train_count),
            "note": "DYNAMAX 1.0.2 cannot initialize a one-state Dirichlet; this is the exact Gaussian baseline.",
        }
    ]
    target_fit: dict[str, Any] | None = None
    candidate_iterations = max(config.em_iters, 250)
    for states in (2, 3, 4):
        seeds = [config.seed + 10 * states + start for start in range(3)]
        fits = [
            fit_dynamax_hmm(
                train,
                num_states=states,
                seed=seed,
                em_iters=candidate_iterations,
            )
            for seed in seeds
        ]
        fit, start_diagnostics = _select_converged_multistart(fits, seeds)
        parameters = fit["parameters"]
        holdout = conditional_holdout_log_prob(
            observations,
            train_count,
            parameters["initial_probs"],
            parameters["transition_matrix"],
            parameters["means"],
            parameters["covariances"],
        )
        final_increment = float(fit["final_em_increment"])
        converged_starts = sum(bool(row["em_converged"]) for row in start_diagnostics)
        candidates.append(
            {
                "engine": "dynamax",
                "states": states,
                "starts": len(fits),
                "converged_starts": converged_starts,
                "minimum_converged_starts_required": 2,
                "multistart_converged": converged_starts >= 2,
                "all_starts_converged": all(bool(row["em_converged"]) for row in start_diagnostics),
                "start_diagnostics": start_diagnostics,
                "em_iterations": candidate_iterations,
                "em_converged": abs(final_increment) <= _EM_CONVERGENCE_TOLERANCE,
                "final_em_increment": final_increment,
                "train_log_likelihood": float(fit["marginal_log_likelihood"]),
                "heldout_log_likelihood": float(holdout),
                "heldout_mean_log_score": float(holdout) / (len(observations) - train_count),
                "annualized_composite_volatility_percent": _python(
                    fit["annualized_composite_volatility"]
                ),
                "occupancy": _python(jnp.mean(fit["smoothed_probs"], axis=0)),
                "minimum_covariance_eigenvalue_percent_squared": float(
                    fit["minimum_covariance_eigenvalue"]
                ),
                "minimum_em_increment": float(fit["minimum_em_increment"]),
            }
        )
        if states == config.num_states:
            target_fit = fit
    if target_fit is None:
        raise ValueError("num_states must be one of the evaluated HMM candidates: 2, 3, or 4")
    return candidates, target_fit


def _full_fit(observations: Any, config: JaxRegimeStudyConfig) -> dict[str, Any]:
    seeds = [config.seed + 100 + index for index in range(3)]
    fits = [
        fit_dynamax_hmm(
            observations,
            num_states=config.num_states,
            seed=seed,
            em_iters=max(config.em_iters, 250),
        )
        for seed in seeds
    ]
    selected, start_diagnostics = _select_converged_multistart(fits, seeds)
    result = dict(selected)
    result["start_diagnostics"] = start_diagnostics
    result["converged_starts"] = sum(bool(row["em_converged"]) for row in start_diagnostics)
    result["minimum_converged_starts_required"] = 2
    result["multistart_converged"] = result["converged_starts"] >= 2
    result["all_starts_converged"] = all(bool(row["em_converged"]) for row in start_diagnostics)
    return result


def _synthetic_recovery(config: JaxRegimeStudyConfig) -> dict[str, Any]:
    runs = [_single_synthetic_recovery(config, offset) for offset in (0, 101, 211)]
    return {
        "deterministic_seeds": [int(row["seed"]) for row in runs],
        "run_count": len(runs),
        "timesteps_per_run": int(runs[0]["timesteps"]),
        "maximum_relative_volatility_error": max(
            float(row["maximum_relative_volatility_error"]) for row in runs
        ),
        "maximum_transition_rmse": max(float(row["transition_rmse"]) for row in runs),
        "minimum_smoothed_state_accuracy": min(
            float(row["smoothed_state_accuracy"]) for row in runs
        ),
        "runs": runs,
        "passed": all(bool(row["passed"]) for row in runs),
    }


def _best_state_permutation(true_covariances: Any, fitted_covariances: Any) -> tuple[int, ...]:
    """Return true-state to fitted-state assignment minimizing covariance error."""

    _jax, jnp, _jr = _stack()
    candidates = permutations(range(int(true_covariances.shape[0])))
    return min(
        candidates,
        key=lambda candidate: float(
            jnp.sum((true_covariances - fitted_covariances[jnp.asarray(candidate, dtype=int)]) ** 2)
        ),
    )


def _single_synthetic_recovery(config: JaxRegimeStudyConfig, seed_offset: int) -> dict[str, Any]:
    _jax, jnp, _jr = _stack()
    initial = jnp.array([0.5, 0.3, 0.2])
    transition = jnp.array([[0.97, 0.02, 0.01], [0.02, 0.96, 0.02], [0.02, 0.03, 0.95]])
    means = jnp.array([[0.02, -0.01], [-0.04, 0.03], [-0.12, 0.10]])
    covariances = jnp.array(
        [
            [[0.25, 0.02], [0.02, 0.16]],
            [[1.00, 0.10], [0.10, 0.64]],
            [[4.00, 0.50], [0.50, 2.25]],
        ]
    )
    data_seed = config.seed + 700 + seed_offset
    states, emissions = sample_dynamax_hmm(
        initial=initial,
        transition=transition,
        means=means,
        covariances=covariances,
        seed=data_seed,
        timesteps=1800,
    )
    fit = fit_dynamax_hmm(
        emissions,
        num_states=3,
        seed=data_seed + 1,
        em_iters=max(config.em_iters, 250),
    )
    true_vol = jnp.sqrt(252.0 * jnp.einsum("d,kde,e->k", jnp.ones(2), covariances, jnp.ones(2)))
    permutation = _best_state_permutation(covariances, fit["parameters"]["covariances"])
    fitted_indices = jnp.asarray(permutation, dtype=int)
    fitted_vol = fit["annualized_composite_volatility"][fitted_indices]
    relative_error = jnp.abs(fitted_vol - true_vol) / true_vol
    decoded = jnp.argmax(fit["smoothed_probs"], axis=1)
    fitted_to_true = jnp.zeros(3, dtype=int)
    fitted_to_true = fitted_to_true.at[fitted_indices].set(jnp.arange(3))
    aligned_decoded = fitted_to_true[decoded]
    state_accuracy = jnp.mean(aligned_decoded == states)
    fitted_transition = fit["parameters"]["transition_matrix"]
    aligned_transition = fitted_transition[fitted_indices][:, fitted_indices]
    transition_rmse = jnp.sqrt(jnp.mean((aligned_transition - transition) ** 2))
    passed = (
        (jnp.max(relative_error) <= 0.35)
        & (transition_rmse <= 0.12)
        & (state_accuracy >= 0.60)
        & (jnp.abs(fit["final_em_increment"]) <= 1.0e-3)
    )
    return {
        "seed": data_seed,
        "timesteps": len(emissions),
        "true_annualized_composite_volatility_percent": _python(true_vol),
        "fitted_annualized_composite_volatility_percent": _python(fitted_vol),
        "true_to_fitted_state_assignment": list(permutation),
        "assignment_method": "exact covariance Frobenius-cost minimization over all permutations",
        "maximum_relative_volatility_error": float(jnp.max(relative_error)),
        "transition_rmse": float(transition_rmse),
        "smoothed_state_accuracy": float(state_accuracy),
        "final_em_increment": float(fit["final_em_increment"]),
        "passed": bool(passed),
    }
