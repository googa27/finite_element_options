"""Public API for statsmodels-backed regime-switching quanto calibration.

This module intentionally belongs to the installed examples tree.  It re-exports
research helpers for cleaning PDP-style S&P 500 / USDCLP levels, fitting a
Markov-switching composite-return model, and deriving a named CTMC generator
from the fitted discrete transition matrix.
"""

from __future__ import annotations

from finite_element_options.examples.regime_switching_quanto._types import (
    DataQualityConfig,
    MarkovSwitchingDiffusionResult,
    RegimeCandidateResult,
    RegimeSummary,
)
from finite_element_options.examples.regime_switching_quanto.fitting import (
    fit_markov_switching_joint_diffusion,
    fit_regime_candidates,
)
from finite_element_options.examples.regime_switching_quanto.generator import (
    discrete_to_continuous_generator,
)
from finite_element_options.examples.regime_switching_quanto.quality import (
    prepare_joint_log_returns,
)

__all__ = [
    "DataQualityConfig",
    "MarkovSwitchingDiffusionResult",
    "RegimeCandidateResult",
    "RegimeSummary",
    "discrete_to_continuous_generator",
    "fit_markov_switching_joint_diffusion",
    "fit_regime_candidates",
    "prepare_joint_log_returns",
]
