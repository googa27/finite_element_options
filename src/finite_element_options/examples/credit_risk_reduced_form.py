"""Reduced-form credit-risk bond valuation example.

This example uses the analytical constant-intensity reference model.  It does
not route the claim through a spatial FEM solver because a constant hazard-rate
zero-coupon bond has no spatial state variable.
"""

from __future__ import annotations

import numpy as np

from finite_element_options.problems.credit_risk import (
    CreditRiskIntensitySampler,
    DefaultableZeroCouponClaim,
    ReducedFormCreditRiskModel,
)


def analytical_and_sampled_values(n_paths: int = 10) -> tuple[float, float]:
    """Return analytical base value and mean value under intensity uncertainty."""

    model = ReducedFormCreditRiskModel(r=0.03, default_intensity=0.02)
    claim = DefaultableZeroCouponClaim(notional=1.0, recovery_rate=0.4)
    maturity = 1.0
    base_value = model.defaultable_zero_coupon_value(claim, maturity)

    rng = np.random.default_rng(0)
    sampler = CreditRiskIntensitySampler(base_model=model, log_std=0.3)
    sampled_values = [
        sampler.sample(rng).defaultable_zero_coupon_value(claim, maturity)
        for _ in range(n_paths)
    ]
    return float(base_value), float(np.mean(sampled_values))


if __name__ == "__main__":  # pragma: no cover - example script
    analytical, sampled_mean = analytical_and_sampled_values()
    print("Analytical base value:", analytical)
    print("Sampled-intensity mean value:", sampled_mean)
