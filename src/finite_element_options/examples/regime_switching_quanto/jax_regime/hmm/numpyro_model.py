"""NumPyro NUTS for continuous HMM parameters with states marginalized."""

from __future__ import annotations

from functools import partial
from typing import Any

from ..contracts import NumPyroPriorConfig

from .forward import gaussian_hmm_log_prob


def _stack() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import jax
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
        from numpyro.diagnostics import summary
        from numpyro.infer import MCMC, NUTS
        from numpyro.infer.initialization import init_to_value
    except ModuleNotFoundError as exc:
        raise ImportError(
            "NumPyro HMM inference requires finite-element-options[jax-regime]."
        ) from exc
    jax.config.update("jax_enable_x64", True)
    return jax, jnp, numpyro, dist, summary, (MCMC, NUTS, init_to_value)


def _correlations(covariances: Any, jnp: Any) -> Any:
    scales = jnp.sqrt(jnp.diagonal(covariances, axis1=-2, axis2=-1))
    return covariances[:, 0, 1] / (scales[:, 0] * scales[:, 1])


def _model(
    observations: Any,
    initial: Any,
    transition_reference: Any,
    mean_reference: Any,
    covariance_reference: Any,
    prior: NumPyroPriorConfig,
) -> None:
    _jax, jnp, numpyro, dist, _summary, _inference = _stack()
    num_states, dimension = mean_reference.shape
    initial_probs = numpyro.sample(
        "initial_probs",
        dist.Dirichlet(1.0 + prior.initial_weight * initial),
    )
    transition = numpyro.sample(
        "transition_matrix",
        dist.Dirichlet(1.0 + prior.transition_weight * transition_reference),
    )
    reference_scale = jnp.sqrt(jnp.diagonal(covariance_reference, axis1=-2, axis2=-1))
    means = numpyro.sample(
        "means",
        dist.Normal(
            mean_reference,
            jnp.maximum(reference_scale * prior.mean_scale_multiplier, 1.0e-4),
        ).to_event(2),
    )
    log_scales = numpyro.sample(
        "log_scales",
        dist.Normal(jnp.log(reference_scale), prior.log_scale_sd).to_event(2),
    )
    reference_rho = jnp.clip(_correlations(covariance_reference, jnp), -0.95, 0.95)
    reference_rho_raw = jnp.arctanh(reference_rho / 0.98)
    rho_raw = numpyro.sample(
        "rho_raw", dist.Normal(reference_rho_raw, prior.correlation_raw_sd).to_event(1)
    )
    scales = jnp.exp(log_scales)
    rho = 0.98 * jnp.tanh(rho_raw)
    covariances = jnp.zeros((num_states, dimension, dimension))
    covariances = covariances.at[:, 0, 0].set(scales[:, 0] ** 2)
    covariances = covariances.at[:, 1, 1].set(scales[:, 1] ** 2)
    covariances = covariances.at[:, 0, 1].set(rho * scales[:, 0] * scales[:, 1])
    covariances = covariances.at[:, 1, 0].set(rho * scales[:, 0] * scales[:, 1])
    numpyro.deterministic("covariances", covariances)
    numpyro.factor(
        "marginal_hmm_log_likelihood",
        gaussian_hmm_log_prob(observations, initial_probs, transition, means, covariances),
    )


def _diagnostic_extrema(report: dict[str, Any]) -> dict[str, Any]:
    diagnostics: list[dict[str, Any]] = []
    for name, values in report.items():
        rhat = values.get("r_hat")
        ess = values.get("n_eff")
        if rhat is None or ess is None:
            continue
        flat_rhat = rhat.reshape(-1)
        flat_ess = ess.reshape(-1)
        for index in range(len(flat_rhat)):
            diagnostics.append(
                {
                    "parameter": name,
                    "flat_index": index,
                    "rhat": float(flat_rhat[index]),
                    "ess": float(flat_ess[index]),
                }
            )
    maximum_rhat = max((row["rhat"] for row in diagnostics), default=float("nan"))
    minimum_ess = min((row["ess"] for row in diagnostics), default=float("nan"))
    weak = [row for row in diagnostics if row["rhat"] > 1.05 or row["ess"] < 100.0]
    return {
        "maximum_rhat": maximum_rhat,
        "minimum_ess": minimum_ess,
        "worst_rhat": max(diagnostics, key=lambda row: row["rhat"], default=None),
        "worst_ess": min(diagnostics, key=lambda row: row["ess"], default=None),
        "weakly_identified_parameters": weak,
    }


def _required_divergence_count(extra_fields: dict[str, Any]) -> int:
    """Return divergence count, failing closed when NumPyro omits telemetry."""

    if "diverging" not in extra_fields:
        raise RuntimeError("NumPyro extra fields omitted required 'diverging' diagnostics")
    return int(extra_fields["diverging"].sum())


def run_numpyro_hmm(
    observations: Any,
    *,
    initial: Any,
    transition: Any,
    means: Any,
    covariances: Any,
    seed: int,
    warmup: int,
    samples: int,
    chains: int,
    prior: NumPyroPriorConfig | None = None,
) -> dict[str, Any]:
    """Run reference-identified NUTS with all latent HMM states integrated out."""

    jax, jnp, _numpyro, _dist, diagnostics_summary, inference = _stack()
    prior = prior or NumPyroPriorConfig()
    MCMC, NUTS, init_to_value = inference
    reference_scales = jnp.sqrt(jnp.diagonal(covariances, axis1=-2, axis2=-1))
    reference_rho = jnp.clip(_correlations(covariances, jnp), -0.95, 0.95)
    initial_values = {
        "initial_probs": jnp.asarray(initial),
        "transition_matrix": jnp.asarray(transition),
        "means": jnp.asarray(means),
        "log_scales": jnp.log(reference_scales),
        "rho_raw": jnp.arctanh(reference_rho / 0.98),
    }
    kernel = NUTS(
        partial(_model, prior=prior),
        target_accept_prob=0.85,
        init_strategy=init_to_value(values=initial_values),
    )
    mcmc = MCMC(
        kernel,
        num_warmup=warmup,
        num_samples=samples,
        num_chains=chains,
        chain_method="parallel" if jax.local_device_count() >= chains else "sequential",
        progress_bar=False,
    )
    mcmc.run(
        jax.random.key(seed),
        jnp.asarray(observations),
        jnp.asarray(initial),
        jnp.asarray(transition),
        jnp.asarray(means),
        jnp.asarray(covariances),
    )
    grouped = mcmc.get_samples(group_by_chain=True)
    report = diagnostics_summary(grouped, group_by_chain=True)
    diagnostics = _diagnostic_extrema(report)
    extra = mcmc.get_extra_fields(group_by_chain=True)
    divergences = _required_divergence_count(extra)
    flat = mcmc.get_samples(group_by_chain=False)
    posterior_mean = {name: jnp.mean(value, axis=0) for name, value in flat.items()}
    finite = all(bool(jnp.all(jnp.isfinite(value))) for value in flat.values())
    return {
        "chains": chains,
        "draws_per_chain": samples,
        "divergences": divergences,
        "maximum_rhat": diagnostics["maximum_rhat"],
        "minimum_ess": diagnostics["minimum_ess"],
        "worst_rhat": diagnostics["worst_rhat"],
        "worst_ess": diagnostics["worst_ess"],
        "weakly_identified_parameters": diagnostics["weakly_identified_parameters"],
        "finite": finite,
        "diagnostic_fields": sorted(extra),
        "prior": prior.to_dict(),
        "posterior_mean": posterior_mean,
        "samples": flat,
        "grouped_samples": grouped,
    }
