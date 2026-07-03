# FEM Pinares fixed-price proxy fixture (public)

This directory contains the public-synthetic Project #12 FEM compatibility
fixture for Pinares issue `googa27/finite_element_options#78`.

Files:

- `problem_spec.json` — canonical `quant-problem-spec/v0` payload for the
  supported fixed-price option proxy.
- `result_export.json` — deterministic FEM weak-form convergence rows and
  analytical-oracle comparison for `PINARES-FEM-FIXED-PRICE-PROXY-V0`.
- `unsupported_full_deal_problem_spec.json` — negative contract for
  `PINARES-FEM-FAIL-CLOSED-V0`; full family-contract/ROFR/legal-tax/jump/HJB
  requests must fail before mesh allocation.

This is not private Pinares data and not a legal/tax/family-contract valuation.
It validates only a one-dimensional UF-denominated Black-Scholes-style weak-form
proxy with survival-scaled terminal payoff.

Regenerate with:

```bash
python scripts/export_pinares_fixed_price_proxy_fixture.py
```
