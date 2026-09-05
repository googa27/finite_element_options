"""Lazy native-PyMC posterior smoke for the isolated Bayesian profile."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from .contracts import BayesianSmokeConfig, require_supported_python
from .oracle import build_diagnostic_summary


PYMC_HINT = "install finite-element-options[bayesian] in Python 3.12+"


def run_pymc_smoke(config: BayesianSmokeConfig | None = None) -> dict[str, Any]:
    """Run seeded native PyMC NUTS, ArviZ diagnostics, and posterior predictive."""

    require_supported_python()
    selected = config or BayesianSmokeConfig()
    try:
        import arviz as az
        import pymc as pm
    except ImportError as exc:  # pragma: no cover - isolated profile test
        raise ModuleNotFoundError(PYMC_HINT) from exc

    observations = np.asarray(selected.observations, dtype=float)
    started = perf_counter()
    with pm.Model():
        mean = pm.Normal("mean", mu=selected.prior_mean, sigma=selected.prior_sigma)
        pm.Normal("observed", mu=mean, sigma=selected.known_sigma, observed=observations)
        inference_data = pm.sample(
            draws=selected.draws,
            tune=selected.warmup,
            chains=selected.chains,
            cores=1,
            random_seed=selected.pymc_seed,
            target_accept=selected.target_accept,
            progressbar=False,
            compute_convergence_checks=False,
            return_inferencedata=True,
        )
        predictive = pm.sample_posterior_predictive(
            inference_data,
            var_names=["observed"],
            random_seed=selected.pymc_predictive_seed,
            progressbar=False,
        )
    elapsed = perf_counter() - started
    summary_frame: Any = az.summary(inference_data, var_names=["mean"], round_to=None)
    summary = summary_frame.loc["mean"]
    posterior = np.asarray(inference_data.posterior["mean"], dtype=float)
    predictive_values = np.asarray(predictive.posterior_predictive["observed"], dtype=float)
    log_density = np.asarray(inference_data.sample_stats["lp"], dtype=float)
    divergences = int(np.asarray(inference_data.sample_stats["diverging"]).sum())
    return build_diagnostic_summary(
        engine="pymc",
        sampler="native_nuts",
        version=pm.__version__,
        arviz_version=az.__version__,
        posterior_mean=float(summary["mean"]),
        posterior_sd=float(summary["sd"]),
        posterior_predictive_mean=float(np.mean(predictive_values)),
        posterior_predictive_sd=float(np.std(predictive_values, ddof=1)),
        rhat=float(summary["r_hat"]),
        ess_bulk=float(summary["ess_bulk"]),
        divergences=divergences,
        elapsed_seconds=elapsed,
        finite_log_density=bool(np.all(np.isfinite(log_density))),
        finite_posterior=bool(np.all(np.isfinite(posterior))),
        finite_predictive=bool(np.all(np.isfinite(predictive_values))),
        config=selected,
    )


__all__ = ["PYMC_HINT", "run_pymc_smoke"]
