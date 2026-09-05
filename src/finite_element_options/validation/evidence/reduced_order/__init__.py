"""Public-synthetic reduced-order validation boundary."""

from .benchmark import (
    HoldoutEvaluation,
    PymorBenchmarkReport,
    TimingSummary,
    run_pymor_benchmark,
    verify_pymor_benchmark,
)
from .black_scholes import (
    AffineBlackScholesSystem,
    PreparedFullOrderSolver,
    TrainedPymorROM,
    build_affine_black_scholes_system,
    train_pymor_rom,
)
from .contracts import (
    FullOrderSolution,
    OptionOutputs,
    PODProjection,
    PymorBlackScholesConfig,
    ROMEnvelopeError,
    SCHEMA_VERSION,
)

__all__ = [
    "AffineBlackScholesSystem",
    "FullOrderSolution",
    "HoldoutEvaluation",
    "OptionOutputs",
    "PODProjection",
    "PreparedFullOrderSolver",
    "PymorBenchmarkReport",
    "PymorBlackScholesConfig",
    "ROMEnvelopeError",
    "SCHEMA_VERSION",
    "TimingSummary",
    "TrainedPymorROM",
    "build_affine_black_scholes_system",
    "run_pymor_benchmark",
    "train_pymor_rom",
    "verify_pymor_benchmark",
]
