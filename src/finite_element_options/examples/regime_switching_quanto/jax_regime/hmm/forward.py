"""HMM likelihood compatibility, research filtering, and forecast composition."""

from __future__ import annotations

from typing import Any

from .tfp_adapter import tfp_marginal_log_prob


def _jax() -> tuple[Any, Any, Any]:
    try:
        import jax
        import jax.numpy as jnp
        from jax.scipy.special import logsumexp
    except ModuleNotFoundError as exc:
        raise ImportError(
            "JAX HMM likelihood requires finite-element-options[jax-regime]."
        ) from exc
    return jax, jnp, logsumexp


def gaussian_emission_log_prob(observations: Any, means: Any, covariances: Any) -> Any:
    """Return multivariate-normal log densities with shape ``(time, states)``."""

    _jax_module, jnp, _logsumexp = _jax()
    observations = jnp.asarray(observations)
    means = jnp.asarray(means)
    covariances = jnp.asarray(covariances)
    difference = observations[:, None, :] - means[None, :, :]
    sign, logdet = jnp.linalg.slogdet(covariances)
    solved = jnp.linalg.solve(covariances[None, :, :, :], difference[:, :, :, None])
    quadratic = jnp.sum(difference * solved[..., 0], axis=-1)
    dimension = observations.shape[-1]
    return jnp.where(
        sign[None, :] > 0,
        -0.5 * (dimension * jnp.log(2.0 * jnp.pi) + logdet[None, :] + quadratic),
        -jnp.inf,
    )


def gaussian_hmm_log_prob(
    observations: Any,
    initial_probs: Any,
    transition_matrix: Any,
    means: Any,
    covariances: Any,
) -> Any:
    """Delegate scalar marginalization to the maintained TFP log-space owner."""

    return tfp_marginal_log_prob(
        observations, initial_probs, transition_matrix, means, covariances
    )


def gaussian_hmm_filter_probs(
    observations: Any,
    initial_probs: Any,
    transition_matrix: Any,
    means: Any,
    covariances: Any,
) -> Any:
    """Return filtered state probabilities with the same marginalized recursion."""

    jax, jnp, logsumexp = _jax()
    emissions = gaussian_emission_log_prob(observations, means, covariances)
    log_transition = jnp.log(jnp.clip(jnp.asarray(transition_matrix), min=1.0e-300))
    first = jnp.log(jnp.clip(jnp.asarray(initial_probs), min=1.0e-300)) + emissions[0]
    initial_alpha = first - logsumexp(first)

    def step(log_alpha: Any, emission: Any) -> tuple[Any, Any]:
        prediction = logsumexp(log_alpha[:, None] + log_transition, axis=0)
        current = prediction + emission
        normalized = current - logsumexp(current)
        return normalized, jnp.exp(normalized)

    _last, rest = jax.lax.scan(step, initial_alpha, emissions[1:])
    return jnp.concatenate([jnp.exp(initial_alpha)[None, :], rest], axis=0)


def forecast_next_state_probs(filtered_probs: Any, transition_matrix: Any) -> Any:
    """Advance a filtered end-of-sample law to the first future emission state."""

    _jax_module, jnp, _logsumexp = _jax()
    return jnp.asarray(filtered_probs) @ jnp.asarray(transition_matrix)


def conditional_holdout_log_prob(
    observations: Any,
    train_count: int,
    initial_probs: Any,
    transition_matrix: Any,
    means: Any,
    covariances: Any,
) -> Any:
    """Return chronological hold-out log score conditional on the training sample."""

    full = gaussian_hmm_log_prob(
        observations, initial_probs, transition_matrix, means, covariances
    )
    train = gaussian_hmm_log_prob(
        observations[:train_count], initial_probs, transition_matrix, means, covariances
    )
    return full - train
