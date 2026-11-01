"""Stable maintained log-space HMM likelihood for the locked research profile."""

from __future__ import annotations

from typing import Any


def tfp_marginal_log_prob(
    observations: Any,
    initial: Any,
    transition: Any,
    means: Any,
    covariances: Any,
) -> Any:
    """Marginalize Gaussian-HMM states through TFP's public JAX distribution.

    Preserve the existing input-probability floor, including zero-probability
    compatibility. DYNAMAX 1.0.2's separate intermediate weight floor changes
    tail likelihoods and gradients, so it is not this NumPyro likelihood owner.
    """

    try:
        import jax
        import jax.numpy as jnp
        from tensorflow_probability.substrates.jax import distributions as tfd
    except ModuleNotFoundError as exc:
        raise ImportError(
            "TFP HMM likelihood requires finite-element-options[jax-regime]."
        ) from exc
    jax.config.update("jax_enable_x64", True)
    # Match the original linalg covariance promotion, then give TFP one shared
    # inexact emission dtype. Homogeneous float32/float64 precision is retained.
    observations = jnp.asarray(observations)
    means = jnp.asarray(means)
    covariances = jnp.asarray(covariances)
    covariance_dtype = jnp.result_type(covariances, 0.0)
    emission_dtype = jnp.result_type(observations, means, covariance_dtype)
    observations = observations.astype(emission_dtype)
    model = tfd.HiddenMarkovModel(
        initial_distribution=tfd.Categorical(
            logits=jnp.log(jnp.clip(jnp.asarray(initial), min=1.0e-300))
        ),
        transition_distribution=tfd.Categorical(
            logits=jnp.log(jnp.clip(jnp.asarray(transition), min=1.0e-300))
        ),
        observation_distribution=tfd.MultivariateNormalFullCovariance(
            loc=means.astype(emission_dtype),
            covariance_matrix=covariances.astype(emission_dtype),
        ),
        num_steps=observations.shape[0],
    )
    return model.log_prob(observations)
