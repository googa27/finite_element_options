"""DYNAMAX-owned Gaussian-HMM EM, filter, and smoother adapter."""

from __future__ import annotations

from typing import Any


def _stack() -> tuple[Any, Any, Any]:
    try:
        import jax
        import jax.numpy as jnp
        import jax.random as jr
        from dynamax.hidden_markov_model import GaussianHMM
    except ModuleNotFoundError as exc:
        raise ImportError("DYNAMAX fitting requires finite-element-options[jax-regime].") from exc
    jax.config.update("jax_enable_x64", True)
    return jnp, jr, GaussianHMM


def _arrays(params: Any) -> tuple[Any, Any, Any, Any]:
    return (
        params.initial.probs,
        params.transitions.transition_matrix,
        params.emissions.means,
        params.emissions.covs,
    )


def _canonical_order(covariances: Any) -> Any:
    jnp, _jr, _hmm = _stack()
    direction = jnp.ones((covariances.shape[-1],))
    composite_variance = jnp.einsum("d,kde,e->k", direction, covariances, direction)
    return jnp.argsort(composite_variance)


def dynamax_marginal_log_prob(
    observations: Any,
    initial: Any,
    transition: Any,
    means: Any,
    covariances: Any,
) -> Any:
    """Evaluate DYNAMAX's Gaussian-HMM marginal likelihood on explicit parameters."""

    _jnp, jr, GaussianHMM = _stack()
    model = GaussianHMM(num_states=int(initial.shape[0]), emission_dim=int(means.shape[1]))
    params, _props = model.initialize(
        key=jr.key(0),
        initial_probs=initial,
        transition_matrix=transition,
        emission_means=means,
        emission_covariances=covariances,
    )
    return model.marginal_log_prob(params, observations)


def sample_dynamax_hmm(
    *,
    initial: Any,
    transition: Any,
    means: Any,
    covariances: Any,
    seed: int,
    timesteps: int,
) -> tuple[Any, Any]:
    """Sample latent states and emissions using DYNAMAX's model implementation."""

    _jnp, jr, GaussianHMM = _stack()
    model = GaussianHMM(num_states=int(initial.shape[0]), emission_dim=int(means.shape[1]))
    params, _props = model.initialize(
        key=jr.key(seed),
        initial_probs=initial,
        transition_matrix=transition,
        emission_means=means,
        emission_covariances=covariances,
    )
    return model.sample(params, jr.key(seed + 1), num_timesteps=timesteps)


def fit_dynamax_hmm(
    observations: Any,
    *,
    num_states: int,
    seed: int,
    em_iters: int,
) -> dict[str, Any]:
    """Fit, filter, and smooth a full-covariance Gaussian HMM with DYNAMAX."""

    jnp, jr, GaussianHMM = _stack()
    if num_states < 2:
        raise ValueError(
            "DYNAMAX 1.0.2 cannot initialize a one-state HMM; use the closed-form Gaussian baseline"
        )
    observations = jnp.asarray(observations)
    model = GaussianHMM(num_states=num_states, emission_dim=int(observations.shape[1]))
    params, props = model.initialize(key=jr.key(seed), method="kmeans", emissions=observations)
    params, log_likelihoods = model.fit_em(
        params, props, observations, num_iters=em_iters, verbose=False
    )
    filtered = model.filter(params, observations)
    smoothed = model.smoother(params, observations)
    initial, transition, means, covariances = _arrays(params)
    order = _canonical_order(covariances)
    initial = initial[order]
    transition = transition[order][:, order]
    means = means[order]
    covariances = covariances[order]
    annualized_composite = jnp.sqrt(
        252.0 * jnp.einsum("d,kde,e->k", jnp.ones(2), covariances, jnp.ones(2))
    )
    eigenvalues = jnp.linalg.eigvalsh(covariances)
    em_differences = jnp.diff(log_likelihoods)
    final_increment = em_differences[-1] if em_iters > 1 else jnp.asarray(0.0)
    return {
        "parameters": {
            "initial_probs": initial,
            "transition_matrix": transition,
            "means": means,
            "covariances": covariances,
        },
        "em_log_likelihoods": log_likelihoods,
        "marginal_log_likelihood": filtered.marginal_loglik,
        "filtered_probs": filtered.filtered_probs[:, order],
        "smoothed_probs": smoothed.smoothed_probs[:, order],
        "annualized_composite_volatility": annualized_composite,
        "minimum_covariance_eigenvalue": jnp.min(eigenvalues),
        "minimum_em_increment": jnp.min(em_differences) if em_iters > 1 else jnp.asarray(0.0),
        "final_em_increment": final_increment,
        "finite": jnp.all(jnp.isfinite(log_likelihoods)) & jnp.all(jnp.isfinite(covariances)),
    }
