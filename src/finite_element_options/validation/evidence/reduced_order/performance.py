"""Timing, amortization, environment, and memory evidence for the ROM pilot."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
import gc
from importlib.metadata import version
from math import ceil, isfinite
import os
import platform
import sys
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sps  # type: ignore[import-untyped]

from .contracts import PymorBlackScholesConfig

if TYPE_CHECKING:
    from .black_scholes import AffineBlackScholesSystem, TrainedPymorROM


@dataclass(frozen=True, slots=True)
class TimingSummary:
    """Robust repeated wall-clock summary with retained raw observations."""

    samples_seconds: tuple[float, ...]
    median_seconds: float
    mad_seconds: float
    p05_seconds: float
    p95_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe timing evidence."""

        return asdict(self)


def benchmark_online(
    system: AffineBlackScholesSystem,
    trained: TrainedPymorROM,
) -> dict[str, Any]:
    """Collect alternated repeated FOM/ROM wall-clock observations."""

    config = system.config
    full_samples: list[float] = []
    reduced_samples: list[float] = []
    setup_samples: list[float] = []
    for volatility in config.holdout_volatilities:
        started = perf_counter()
        prepared_full_order = system.prepare_full_order(volatility)
        setup_samples.append(perf_counter() - started)
        for _ in range(config.benchmark_warmups):
            prepared_full_order.solve()
            trained.solve(volatility)
        for repeat in range(config.benchmark_repeats):
            reduced_call = partial(trained.solve, volatility)
            pair: tuple[tuple[Callable[[], Any], list[float]], ...] = (
                (prepared_full_order.solve, full_samples),
                (reduced_call, reduced_samples),
            )
            if repeat % 2:
                pair = tuple(reversed(pair))
            for function, destination in pair:
                destination.append(_time_call(function))
    full = _timing_summary(full_samples)
    reduced = _timing_summary(reduced_samples)
    setup = _timing_summary(setup_samples)
    return {
        "clock": "time.perf_counter",
        "gc_policy": "cyclic GC disabled during each sample and collected after timing",
        "full_order_policy": "parameter-specific assembly/factorization cached before repeated marches",
        "warmups_per_holdout": config.benchmark_warmups,
        "repeats_per_holdout": config.benchmark_repeats,
        "full_order_factorization_uses_per_holdout": (
            config.benchmark_warmups + config.benchmark_repeats
        ),
        "full_order_factorization_reuses_per_holdout": (
            config.benchmark_warmups + config.benchmark_repeats - 1
        ),
        "full_order_setup": setup.to_dict(),
        "full_order": full.to_dict(),
        "reduced_order": reduced.to_dict(),
        "median_online_speedup": full.median_seconds / reduced.median_seconds,
        "minimum_required_speedup": config.minimum_online_speedup,
    }


def amortization_evidence(
    timing: dict[str, Any],
    offline_seconds: float,
    config: PymorBlackScholesConfig,
) -> dict[str, Any]:
    """Compute ordinary and target-speedup offline-cost amortization counts."""

    full = float(timing["full_order"]["median_seconds"])
    reduced = float(timing["reduced_order"]["median_seconds"])
    saving = full - reduced
    break_even = ceil(offline_seconds / saving) if saving > 0.0 else None
    ten_x_denominator = full - config.minimum_online_speedup * reduced
    ten_x = (
        ceil(config.minimum_online_speedup * offline_seconds / ten_x_denominator)
        if ten_x_denominator > 0.0
        else None
    )
    declared = config.maximum_ten_x_amortization_solves
    amortized_at_declared = (
        declared * full / (offline_seconds + declared * reduced)
        if isfinite(offline_seconds)
        else 0.0
    )
    return {
        "offline_seconds_amortized": offline_seconds,
        "break_even_solve_count": break_even,
        "ten_x_amortization_solve_count": ten_x,
        "declared_amortization_solve_count": declared,
        "amortized_speedup_at_declared_count": amortized_at_declared,
    }


def memory_evidence(
    system: AffineBlackScholesSystem,
    trained: TrainedPymorROM,
) -> dict[str, int]:
    """Report deterministic allocated-array byte counts and a bounded peak estimate."""

    full_sparse = sum(
        _sparse_bytes(matrix)
        for matrix in (system.mass, system.operator_constant, system.operator_variance)
    )
    snapshots = system.interior_dofs * trained.snapshot_count * np.dtype(float).itemsize
    reduced_dimension = trained.basis_size
    online_payload = trained.projection.memory_bytes + trained.output_boundary_weights.nbytes
    workspace = np.dtype(float).itemsize * (
        3 * reduced_dimension * reduced_dimension + 6 * reduced_dimension
    )
    return {
        "full_reference_affine_operator_bytes": full_sparse,
        "offline_snapshot_matrix_bytes": int(snapshots),
        "offline_snapshot_peak_estimate_bytes": int(2 * snapshots),
        "online_rom_numerical_payload_bytes": int(online_payload),
        "online_rom_solve_workspace_estimate_bytes": int(workspace),
    }


def environment_evidence(trained: TrainedPymorROM) -> dict[str, Any]:
    """Return bounded software/hardware metadata without local paths or host names."""

    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unreported",
        "logical_cpu_count": os.cpu_count(),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "scikit_fem": version("scikit-fem"),
        "pymor": trained.projection.library_version,
        "pymor_license": "BSD-2-Clause AND BSD-3-Clause",
        "pymor_cache_policy": (
            "scoped and RLock-serialized disable during adapter construction; prior environment and process-wide policy restored; "
            "no persisted cache read"
        ),
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "byteorder": sys.byteorder,
    }


def _time_call(function: Callable[[], Any]) -> float:
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    started = perf_counter()
    try:
        function()
    finally:
        elapsed = perf_counter() - started
        if gc_was_enabled:
            gc.enable()
        gc.collect()
    return elapsed


def _timing_summary(values: list[float]) -> TimingSummary:
    array = np.asarray(values, dtype=float)
    center = float(np.median(array))
    return TimingSummary(
        samples_seconds=tuple(float(value) for value in array),
        median_seconds=center,
        mad_seconds=float(np.median(np.abs(array - center))),
        p05_seconds=float(np.quantile(array, 0.05)),
        p95_seconds=float(np.quantile(array, 0.95)),
    )


def _sparse_bytes(matrix: sps.spmatrix) -> int:
    sparse = sps.csc_matrix(matrix)
    return int(sparse.data.nbytes + sparse.indices.nbytes + sparse.indptr.nbytes)


__all__ = [
    "TimingSummary",
    "amortization_evidence",
    "benchmark_online",
    "environment_evidence",
    "memory_evidence",
]
