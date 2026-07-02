"""Problem presets bundling dynamics, payoffs and boundary sets."""

from .base import Problem
from .option_pricing import OptionPricingProblem
from .credit_risk import CreditRiskProblem

__all__ = ["Problem", "OptionPricingProblem", "CreditRiskProblem"]
