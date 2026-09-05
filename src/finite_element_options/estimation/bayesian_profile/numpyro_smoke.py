"""JAX-native NumPyro inference smoke isolated from the NumPy/SciPy FEM route."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from .contracts import BayesianSmokeConfig, require_supported_python
from .oracle import build_diagnostic_summary


NUMPYRO_HINT = "install finite-element-options[bayesian-jax] in Python 3.12+"


def run_numpyro_smoke(config: BayesianSmokeConfig | None = None) -> dict[str, Any]:
    """Run seeded JAX-native NumPyro NUTS and ArviZ diagnostics."""

    require_supported_python()
    selected = config or BayesianSmokeConfig()
    try:
        import arviz as az
        import jax
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS, Predictive
    except ImportError as exc:  # pragma: no cover - isolated profile test
        raise ModuleNotFoundError(NUMPYRO_HINT) from exc

    observations = jnp.asarray(selected.observations)

    def model(observed: Any = None) -> None:
        mean = numpyro.sample(
            "mean",
            dist.Normal(selected.prior_mean, selected.prior_sigma),
        )
        with numpyro.plate("observations", observations.shape[0]):
            numpyro.sample(
                "observed",
                dist.Normal(mean, selected.known_sigma),
                obs=observed,
            )

    started = perf_counter()
    mcmc = MCMC(
        NUTS(model, target_accept_prob=selected.target_accept),
        num_warmup=selected.warmup,
        num_samples=selected.draws,
        num_chains=selected.chains,
        chain_method="sequential",
        progress_bar=False,
    )
    mcmc.run(
        jax.random.key(selected.numpyro_seed),
        observed=observations,
        extra_fields=("potential_energy",),
    )
    inference_data = az.from_numpyro(mcmc)
    predictive = Predictive(model, posterior_samples=mcmc.get_samples())(
        jax.random.key(selected.numpyro_predictive_seed),
        observed=None,
    )["observed"]
    elapsed = perf_counter() - started
    summary_frame: Any = az.summary(inference_data, var_names=["mean"], round_to=None)
    summary = summary_frame.loc["mean"]
    extras = mcmc.get_extra_fields(group_by_chain=True)
    posterior = jnp.asarray(inference_data.posterior["mean"].values)
    log_density = -jnp.asarray(extras["potential_energy"])
    divergences = int(jnp.asarray(extras["diverging"]).sum())
    return {
        **build_diagnostic_summary(
            engine="numpyro",
            sampler="jax_native_nuts",
            version=numpyro.__version__,
            arviz_version=az.__version__,
            posterior_mean=float(summary["mean"]),
            posterior_sd=float(summary["sd"]),
            posterior_predictive_mean=float(jnp.mean(predictive)),
            posterior_predictive_sd=float(jnp.std(predictive, ddof=1)),
            rhat=float(summary["r_hat"]),
            ess_bulk=float(summary["ess_bulk"]),
            divergences=divergences,
            elapsed_seconds=elapsed,
            finite_log_density=bool(jnp.all(jnp.isfinite(log_density))),
            finite_posterior=bool(jnp.all(jnp.isfinite(posterior))),
            finite_predictive=bool(jnp.all(jnp.isfinite(predictive))),
            config=selected,
        ),
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
        "jax_device_count": jax.device_count(),
        "jax_enable_x64": bool(getattr(jax.config, "jax_enable_x64", False)),
    }


def jax_fem_differentiation_status() -> dict[str, object]:
    """Return the fail-closed boundary for unsupported FEM autodifferentiation."""

    return {
        "status": "unsupported",
        "supported": False,
        "reason": (
            "scikit-fem/SciPy assembly and sparse solves are not a pure JAX trace; "
            "no custom VJP/JVP or implicit differentiation contract exists"
        ),
        "promotion_requirements": [
            "pure-JAX or custom implicit-differentiation operator boundary",
            "finite-difference gradient parity",
            "Taylor remainder convergence test",
            "price and Greek regression evidence",
        ],
    }


def require_jax_fem_differentiation() -> None:
    """Fail rather than implying JAX can automatically differentiate FEM solves."""

    raise NotImplementedError(jax_fem_differentiation_status()["reason"])


__all__ = [
    "NUMPYRO_HINT",
    "jax_fem_differentiation_status",
    "require_jax_fem_differentiation",
    "run_numpyro_smoke",
]
