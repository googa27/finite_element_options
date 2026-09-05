"""Lazy OpenTURNS adapter for seeded propagation and Sobol indices."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from threading import RLock
from types import ModuleType
from typing import Any

import numpy as np

from finite_element_options.examples.regime_switching_quanto.adoption.optional import (
    require_optional,
)

from .contracts import (
    AdditiveSobolRecovery,
    COMPONENT_NAMES,
    ComponentName,
    UQPilotConfig,
)

_OPENTURNS_LOCK = RLock()


@contextmanager
def openturns_seeded(seed: int) -> Iterator[None]:
    """Coordinate same-process OpenTURNS callers through the shared seeded RNG lock."""

    with _OPENTURNS_LOCK:
        openturns = require_optional("openturns")
        state = openturns.RandomGenerator.GetState()
        try:
            openturns.RandomGenerator.SetSeed(int(seed))
            yield
        finally:
            openturns.RandomGenerator.SetState(state)


def _distribution(openturns: ModuleType) -> tuple[Any, str]:
    marginals = [openturns.Uniform(-1.0, 1.0) for _ in range(4)]
    marginals.append(openturns.Normal(0.0, 1.0))
    constructor = getattr(openturns, "ComposedDistribution", None)
    if constructor is None:
        constructor = openturns.JointDistribution
    distribution = constructor(marginals)
    distribution.setDescription(list(COMPONENT_NAMES))
    return distribution, str(constructor.__name__)


def _as_numpy(sample: Any) -> np.ndarray:
    return np.asarray(sample, dtype=float)


def sample_normalized(seed: int, size: int) -> np.ndarray:
    """Draw a seeded OpenTURNS sample from the five independent marginals."""

    with openturns_seeded(seed):
        openturns = require_optional("openturns")
        distribution, _constructor_name = _distribution(openturns)
        return _as_numpy(distribution.getSample(int(size)))


def saltelli_indices(
    response: Callable[[np.ndarray], float], *, seed: int, base_size: int
) -> tuple[
    dict[ComponentName, float],
    dict[ComponentName, float],
    dict[str, dict[ComponentName, dict[str, float]]],
    str,
    str,
]:
    """Compute raw first and total Saltelli finite-sample estimators and CIs."""

    with openturns_seeded(seed):
        openturns = require_optional("openturns")
        distribution, constructor_name = _distribution(openturns)
        experiment = openturns.SobolIndicesExperiment(distribution, int(base_size), False)
        design = experiment.generate()
        x = _as_numpy(design)
        y = np.asarray([[float(response(row))] for row in x], dtype=float)
        algorithm = openturns.SaltelliSensitivityAlgorithm(
            design, openturns.Sample(y.tolist()), int(base_size)
        )
        algorithm.setUseAsymptoticDistribution(True)
        first_raw = list(algorithm.getFirstOrderIndices())
        total_raw = list(algorithm.getTotalOrderIndices())
        intervals = {
            "first_order": _interval_by_component(algorithm.getFirstOrderIndicesInterval()),
            "total_order": _interval_by_component(algorithm.getTotalOrderIndicesInterval()),
        }
        version = str(openturns.__version__)
    first = _raw_indices(first_raw)
    total = _raw_indices(total_raw)
    return first, total, intervals, version, constructor_name


def _raw_indices(values: list[float]) -> dict[ComponentName, float]:
    """Serialize raw Saltelli point estimates without physical clipping."""

    return {name: float(values[index]) for index, name in enumerate(COMPONENT_NAMES)}


def _interval_by_component(interval: Any) -> dict[ComponentName, dict[str, float]]:
    """Serialize OpenTURNS interval bounds as JSON-safe component records."""

    lower = list(interval.getLowerBound())
    upper = list(interval.getUpperBound())
    return {
        name: {"lower": float(lower[index]), "upper": float(upper[index])}
        for index, name in enumerate(COMPONENT_NAMES)
    }


def component_variance_estimates(
    response: Callable[[np.ndarray], float], config: UQPilotConfig
) -> dict[ComponentName, dict[str, float | int]]:
    """Estimate one-at-a-time variances from seeded draws of the actual marginals."""

    estimates: dict[ComponentName, dict[str, float | int]] = {}
    marginal_sample = sample_normalized(config.component_seed, config.component_size)
    for idx, name in enumerate(COMPONENT_NAMES):
        design = np.zeros((config.component_size, 5), dtype=float)
        design[:, idx] = marginal_sample[:, idx]
        values = np.asarray([response(row) for row in design], dtype=float)
        estimates[name] = {
            "variance": float(np.var(values, ddof=1)),
            "std": float(np.std(values, ddof=1)),
            "mean": float(np.mean(values)),
            "size": int(config.component_size),
        }
    return estimates


def additive_sobol_recovery(config: UQPilotConfig) -> AdditiveSobolRecovery:
    """Verify maintained OpenTURNS Saltelli estimators on a cheap additive function."""

    coefficients = np.asarray([2.0, 1.0, 0.5, 0.0, 1.5], dtype=float)
    variances = np.asarray([1 / 3, 1 / 3, 1 / 3, 1 / 3, 1.0], dtype=float) * coefficients**2
    shares = variances / float(np.sum(variances))
    expected: dict[ComponentName, float] = {
        name: float(shares[index]) for index, name in enumerate(COMPONENT_NAMES)
    }

    def additive(row: np.ndarray) -> float:
        return float(np.dot(coefficients, row))

    estimated_first, estimated_total, _intervals, _version, _constructor = saltelli_indices(
        additive, seed=config.additive_sobol_seed, base_size=config.additive_sobol_base_size
    )
    first_errors = [abs(estimated_first[name] - expected[name]) for name in COMPONENT_NAMES]
    total_errors = [abs(estimated_total[name] - expected[name]) for name in COMPONENT_NAMES]
    tolerance = 0.08
    return AdditiveSobolRecovery(
        expected_first=expected,
        estimated_first=estimated_first,
        estimated_total=estimated_total,
        max_abs_error_first=float(max(first_errors)),
        max_abs_error_total=float(max(total_errors)),
        tolerance=tolerance,
        passed=bool(max(first_errors) <= tolerance and max(total_errors) <= tolerance),
        seed=config.additive_sobol_seed,
        base_size=config.additive_sobol_base_size,
    )


__all__ = [
    "additive_sobol_recovery",
    "component_variance_estimates",
    "openturns_seeded",
    "saltelli_indices",
    "sample_normalized",
]
