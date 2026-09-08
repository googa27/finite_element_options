"""Archive-independent synthetic verification for the isolated JAX regime profile."""

from __future__ import annotations

from typing import Any

from .contracts import JaxRegimeStudyConfig, SCHEMA_VERSION
from .hmm.dynamax_adapter import dynamax_marginal_log_prob, sample_dynamax_hmm
from .hmm.experiment import _synthetic_recovery
from .hmm.forward import gaussian_hmm_log_prob
from .hmm.generator_check import check_ctmc_generator
from .hmm.numpyro_model import run_numpyro_hmm
from .pricing.diffrax_sde import simulate_diffrax_terminal_states
from .pricing.exact import simulate_exact_terminal_states
from .utils import stack as _stack


def run_synthetic_verification(config: JaxRegimeStudyConfig) -> dict[str, Any]:
    """Run a bounded synthetic smoke suite without reading PDP or local archives."""

    _jax, jnp, jr = _stack()
    initial = jnp.array([0.65, 0.35])
    transition = jnp.array([[0.96, 0.04], [0.08, 0.92]])
    means = jnp.array([[0.01, -0.005], [-0.015, 0.012]])
    covariances = jnp.array([[[0.7, -0.1], [-0.1, 0.5]], [[1.6, -0.3], [-0.3, 1.2]]])
    _states, observations = sample_dynamax_hmm(
        initial=initial,
        transition=transition,
        means=means,
        covariances=covariances,
        seed=config.seed + 400,
        timesteps=80,
    )
    jax_likelihood = gaussian_hmm_log_prob(observations, initial, transition, means, covariances)
    dynamax_likelihood = dynamax_marginal_log_prob(
        observations, initial, transition, means, covariances
    )
    likelihood_error = float(jnp.abs(jax_likelihood - dynamax_likelihood))
    posterior = run_numpyro_hmm(
        observations,
        initial=initial,
        transition=transition,
        means=means,
        covariances=covariances,
        seed=config.seed + 401,
        warmup=config.warmup,
        samples=config.posterior_samples,
        chains=config.chains,
    )
    regimes = jnp.array([[0, 1, 1, 0], [1, 1, 0, 0]])
    increments = jnp.array(
        [
            [[0.1, -0.2], [0.05, 0.1], [-0.2, 0.3], [0.4, -0.1]],
            [[-0.1, 0.2], [0.2, -0.2], [0.0, 0.1], [0.3, 0.2]],
        ]
    )
    drift = jnp.array([[0.03, 0.01], [-0.02, 0.04]])
    diffusion = jnp.array([[[0.2, 0.0], [-0.05, 0.1]], [[0.35, 0.0], [0.08, 0.22]]])
    exact = simulate_exact_terminal_states(regimes, increments, drift, diffusion, 1.0)
    diffrax = simulate_diffrax_terminal_states(regimes, increments, drift, diffusion, 1.0)
    pathwise_error = float(jnp.max(jnp.abs(exact - diffrax)))
    recovery = _synthetic_recovery(config)
    ctmc = check_ctmc_generator(transition)
    gates = {
        "dynamax_jax_likelihood_parity": likelihood_error <= 1.0e-9,
        "numpyro_two_chains": config.chains >= 2,
        "numpyro_zero_divergence": posterior["divergences"] == 0,
        "numpyro_smoke_rhat": posterior["maximum_rhat"] is not None
        and posterior["maximum_rhat"] <= 1.15,
        "numpyro_smoke_ess": posterior["minimum_ess"] is not None
        and posterior["minimum_ess"] >= 10.0,
        "multi_seed_recovery": bool(recovery["passed"]),
        "ctmc_generator_laws": bool(ctmc["passed"]),
        "diffrax_exact_pathwise_parity": pathwise_error <= 1.0e-10,
    }
    return {
        "schema_version": f"{SCHEMA_VERSION}-synthetic-smoke",
        "status": "passed" if all(gates.values()) else "failed",
        "scope": (
            "bounded archive-independent CI smoke; it does not replace the hash-bound PDP "
            "publication evidence or its stricter R-hat/ESS gates"
        ),
        "config": config.to_dict(),
        "likelihood_absolute_error": likelihood_error,
        "numpyro": {
            "chains": posterior["chains"],
            "draws_per_chain": posterior["draws_per_chain"],
            "divergences": posterior["divergences"],
            "maximum_rhat": posterior["maximum_rhat"],
            "minimum_ess": posterior["minimum_ess"],
        },
        "synthetic_recovery": recovery,
        "ctmc_generator": ctmc,
        "diffrax_exact_maximum_pathwise_error": pathwise_error,
        "gates": gates,
    }
