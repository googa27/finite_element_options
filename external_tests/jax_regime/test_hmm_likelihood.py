"""Independent scalar likelihood oracles for the locked JAX research profile."""

from __future__ import annotations
from typing import Any
from itertools import product
import jax
import jax.numpy as jnp
import numpy as np
from numpyro import handlers
from numpyro.distributions import biject_to, constraints
from numpyro.infer.util import log_density
from jax.scipy.special import logsumexp
import pytest
from tensorflow_probability.substrates.jax import distributions as tfd
from finite_element_options.examples.regime_switching_quanto.jax_regime.contracts import (
    NumPyroPriorConfig,
)
from finite_element_options.examples.regime_switching_quanto.jax_regime.hmm import (
    numpyro_model,
)
from finite_element_options.examples.regime_switching_quanto.jax_regime.hmm.forward import (
    gaussian_hmm_log_prob,
)

jax.config.update("jax_enable_x64", True)


# Independent pre-migration derivation, test oracle only. Production delegates
# marginalization and Gaussian emissions to the maintained TFP JAX distribution.
def _reference_emission(observations: Any, means: Any, covariances: Any) -> Any:
    """Return multivariate-normal log densities with shape ``(time, states)``."""

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


def _reference_log_prob(
    observations: Any,
    initial_probs: Any,
    transition_matrix: Any,
    means: Any,
    covariances: Any,
) -> Any:
    """Marginalize all discrete states with a scaled log-space forward scan."""

    emissions = _reference_emission(observations, means, covariances)
    log_initial = jnp.log(jnp.clip(jnp.asarray(initial_probs), min=1.0e-300))
    log_transition = jnp.log(jnp.clip(jnp.asarray(transition_matrix), min=1.0e-300))
    unnormalized = log_initial + emissions[0]
    first_scale = logsumexp(unnormalized)
    initial_alpha = unnormalized - first_scale

    def step(log_alpha: Any, emission: Any) -> tuple[Any, Any]:
        prediction = logsumexp(log_alpha[:, None] + log_transition, axis=0)
        current = prediction + emission
        scale = logsumexp(current)
        return current - scale, scale

    final_alpha, scales = jax.lax.scan(step, initial_alpha, emissions[1:])
    del final_alpha
    return first_scale + jnp.sum(scales)


def _fixture():
    return (
        jnp.array([0.65, 0.35]),
        jnp.array([[0.96, 0.04], [0.10, 0.90]]),
        jnp.array([[0.01, -0.01], [-0.04, 0.02]]),
        jnp.array([[[0.05, 0.012], [0.012, 0.03]], [[0.20, -0.04], [-0.04, 0.12]]]),
    )


def test_numpyro_propagates_maintained_likelihood_failure(monkeypatch):
    """Removing provider delegation must fail rather than invent a fallback."""
    initial, transition, means, covariances = _fixture()

    def unavailable(*args, **kwargs):
        raise RuntimeError("maintained likelihood unavailable")

    monkeypatch.setattr(tfd.HiddenMarkovModel, "log_prob", unavailable)
    with pytest.raises(RuntimeError, match="maintained likelihood unavailable"):
        handlers.seed(numpyro_model._model, 17)(
            jnp.zeros((4, 2)),
            initial,
            transition,
            means,
            covariances,
            NumPyroPriorConfig(),
        )


@pytest.mark.parametrize(
    "kind",
    [
        "interior",
        "zero-initial",
        "zero-transition",
        "one-state",
        "three-state",
        "four-state",
    ],
)
def test_likelihood_and_emission_gradients_preserve_independent_reference(kind):
    initial, transition, means, covariances = _fixture()
    observations = jax.random.multivariate_normal(
        jax.random.key(1), means[0], covariances[0], (16,)
    )
    if kind == "zero-initial":
        initial = jnp.array([1.0, 0.0])
    if kind == "zero-transition":
        transition = jnp.eye(2)
    if kind == "one-state":
        initial, transition, means, covariances = (
            jnp.ones(1),
            jnp.ones((1, 1)),
            means[:1],
            covariances[:1],
        )
    if kind in {"three-state", "four-state"}:
        states = 3 if kind == "three-state" else 4
        initial = jnp.ones(states) / states
        transition = 0.8 * jnp.eye(states) + 0.2 * jnp.ones((states, states)) / states
        means = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])[:states]
        covariances = jnp.tile(jnp.eye(2), (states, 1, 1))
    args = (observations, initial, transition, means, covariances)
    expected = _reference_log_prob(*args)
    np.testing.assert_allclose(
        gaussian_hmm_log_prob(*args), expected, rtol=1e-11, atol=1e-11
    )
    np.testing.assert_allclose(
        jax.jit(gaussian_hmm_log_prob)(*args), expected, rtol=1e-11, atol=1e-11
    )
    for argnum in [3, 4]:
        actual = jax.grad(gaussian_hmm_log_prob, argnums=argnum)(*args)
        assert bool(jnp.all(jnp.isfinite(actual)))
        np.testing.assert_allclose(
            actual,
            jax.grad(_reference_log_prob, argnums=argnum)(*args),
            rtol=1e-9,
            atol=1e-9,
        )


def test_strict_interior_tail_preserves_literal_likelihood_and_simplex_gradient():
    observations = jnp.zeros((1, 2))
    means = jnp.array([[10.0, 0.0], [0.0, 0.0]])
    covariance = jnp.stack([jnp.eye(2), jnp.eye(2)])
    transition = jax.nn.softmax(jnp.array([[0.0, -40.0], [-40.0, 0.0]]), axis=-1)

    def likelihood(z):
        return gaussian_hmm_log_prob(
            observations, jax.nn.softmax(z), transition, means, covariance
        )

    value, gradient = jax.jit(jax.value_and_grad(likelihood))(jnp.array([0.0, -50.0]))
    # log(2*exp(-50)/(2*pi)); pinned DYNAMAX's intermediate floor is wrong here.
    np.testing.assert_allclose(value, -51.1447298858494, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(gradient, [-0.5, 0.5], atol=1e-12, rtol=1e-12)


def test_existing_clipped_zero_policy_is_preserved():
    args = (
        jnp.zeros((1, 2)),
        jnp.array([1.0, 0.0]),
        jnp.eye(2),
        jnp.array([[1000.0, 0.0], [0.0, 0.0]]),
        jnp.stack([jnp.eye(2), jnp.eye(2)]),
    )
    np.testing.assert_allclose(
        gaussian_hmm_log_prob(*args), -692.613404964623, atol=1e-10, rtol=1e-12
    )


def test_numpyro_joint_and_unconstrained_gradients_match_reference(monkeypatch):
    initial, transition, means, covariances = _fixture()
    observations = jax.random.multivariate_normal(
        jax.random.key(4), means[0], covariances[0], (16,)
    )
    simplex = biject_to(constraints.simplex)
    scales = jnp.sqrt(jnp.diagonal(covariances, axis1=-2, axis2=-1))
    rho = covariances[:, 0, 1] / (scales[:, 0] * scales[:, 1])
    z = jnp.concatenate(
        [
            simplex.inv(initial),
            simplex.inv(transition).reshape(-1),
            means.reshape(-1),
            jnp.log(scales).reshape(-1),
            jnp.arctanh(rho / 0.98),
        ]
    )

    def joint(raw):
        params = {
            "initial_probs": simplex(raw[:1]),
            "transition_matrix": simplex(raw[1:3].reshape(2, 1)),
            "means": raw[3:7].reshape(2, 2),
            "log_scales": raw[7:11].reshape(2, 2),
            "rho_raw": raw[11:13],
        }
        return log_density(
            numpyro_model._model,
            (
                observations,
                initial,
                transition,
                means,
                covariances,
                NumPyroPriorConfig(),
            ),
            {},
            params,
        )[0]

    value, gradient = jax.jit(jax.value_and_grad(joint))(z)
    monkeypatch.setattr(numpyro_model, "gaussian_hmm_log_prob", _reference_log_prob)
    expected, expected_gradient = jax.value_and_grad(joint)(z)
    assert bool(jnp.all(jnp.isfinite(gradient)))
    np.testing.assert_allclose(value, expected, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(gradient, expected_gradient, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize(
    "observation_dtype,mean_dtype,covariance_dtype,probability_dtype",
    [
        (*fields, probability)
        for fields in product((jnp.int64, jnp.float32, jnp.float64), repeat=3)
        for probability in (jnp.float32, jnp.float64)
    ],
)
def test_admitted_real_dtypes_preserve_original_value_and_return_dtype(
    observation_dtype, mean_dtype, covariance_dtype, probability_dtype
):
    # The independent old derivation owns this compatibility oracle; it uses
    # linalg promotion, not the production TFP adapter's common-dtype rule.
    args = (
        jnp.zeros((2, 2), dtype=observation_dtype),
        jnp.ones(1, dtype=probability_dtype),
        jnp.ones((1, 1), dtype=probability_dtype),
        jnp.zeros((1, 2), dtype=mean_dtype),
        jnp.eye(2, dtype=covariance_dtype)[None, :, :],
    )
    expected = _reference_log_prob(*args)
    actual = gaussian_hmm_log_prob(*args)
    assert actual.dtype == expected.dtype
    # Mixed storage inherits float32 rounding where that type participates;
    # the canonical all-float64 research route keeps the tighter budget.
    tolerance = (
        1e-6
        if jnp.float32 in (observation_dtype, mean_dtype, covariance_dtype)
        else 1e-12
    )
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)
