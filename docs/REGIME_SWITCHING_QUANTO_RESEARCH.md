# PDP-calibrated regime-switching equity–FX pricing experiment

**Status:** experimental research example — not a market-calibrated trading model

**Executed:** 2026-09-03/04

**Libraries exercised:** scikit-fem 12.0.2, statsmodels 0.14.6, SciPy 1.17.1, NumPy 2.4.6, pandas 2.3.3

## Result in one sentence

A three-regime Markov AR(2) filter fitted to joint S&P 500/USDCLP returns identifies calm, active, and crisis covariance states. A block-coupled two-factor P1 finite-element PDE produces plausible smooth-payoff values, but only a two-level heuristic extrapolation—not the raw fine grid—falls within the executed Monte Carlo uncertainty bands; digital and dual-trigger FEM values fail parity and remain diagnostic-only.

## Why these libraries

| Capability | Selected route | Evidence and limitation |
|---|---|---|
| Mesh, P1 basis, weak-form assembly | scikit-fem `MeshTri`, `Basis`, `BilinearForm`, `asm` | scikit-fem exposes explicit mesh/basis/assembly primitives and supports boundary-DOF discovery and sparse assembly. It deliberately leaves time stepping, coupled-system policy, and financial boundary conditions to the caller. |
| Latent volatility regimes | statsmodels `MarkovAutoregression` | Supports switching trend, AR coefficients, and innovation variance. It is a **univariate** likelihood, so the subsequent 2×2 equity–FX moments are smoothed-probability estimates rather than a multivariate switching MLE. |
| Diagnostic benchmark | statsmodels `VAR(1)` plus whiteness/stability tests | Useful as a separate linear-dependence diagnostic; it is not part of the switching likelihood. |
| Discrete-to-continuous regime conversion | SciPy `expm`, `logm`, constrained `minimize` | Off-diagonal intensities are constrained nonnegative and rows sum to zero. A discrete transition matrix is not assumed embeddable; the exponential reconstruction residual is reported. |
| Independent pricing oracle | Seeded Monte Carlo | Reuses one simulated terminal-state sample across all payoffs and reports sampling standard errors. It checks the coupled PDE implementation but shares model-form assumptions and uses daily-step CTMC/Euler evolution rather than exact within-step regime-jump simulation. |

Primary library documentation:

- [scikit-fem documentation](https://scikit-fem.readthedocs.io/en/stable/index.html), [how-to guide](https://scikit-fem.readthedocs.io/en/stable/howto.html), [advanced topics](https://scikit-fem.readthedocs.io/en/stable/advanced.html), and [API reference](https://scikit-fem.readthedocs.io/en/stable/api.html)
- [statsmodels MarkovAutoregression](https://www.statsmodels.org/stable/generated/statsmodels.tsa.regime_switching.markov_autoregression.MarkovAutoregression.html), [MarkovRegression](https://www.statsmodels.org/stable/generated/statsmodels.tsa.regime_switching.markov_regression.MarkovRegression.html), and [Markov-switching autoregression example](https://www.statsmodels.org/stable/examples/notebooks/generated/markov_autoregression.html)
- [statsmodels VARResults diagnostics](https://www.statsmodels.org/stable/generated/statsmodels.tsa.vector_ar.var_model.VARResults.html)
- Hamilton, “A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle,” *Econometrica* 57(2), 1989, [DOI 10.2307/1912559](https://doi.org/10.2307/1912559)
- Rannacher, “Finite element solution of diffusion problems with irregular data,” *Numerische Mathematik* 43, 1984, [DOI 10.1007/BF01390130](https://doi.org/10.1007/BF01390130)

## Data contract and quality result

PDP typed source adapters retrieved public daily close levels for `^GSPC` and `CLP=X` over the requested 2010-01-01 to 2026-09-04 half-open window. The immutable merged export contains 4,346 calendar-aligned rows and SHA-256 `aa7ab317266bf37463e27aba9a4e990fa349bb0a6e0aefb5741e93480e0f79f4`.

| Gate | Result |
|---|---:|
| Complete joint level rows before plausibility bounds | 4,187 |
| Calendar/source-alignment exclusions | 159 (153 missing S&P values; 6 missing USDCLP values) |
| Implausible USDCLP levels quarantined | 2 (`5.46` on 2014-04-10; `5.00` on 2016-12-22) |
| Valid joint level rows | 4,185 |
| Joint log-return rows | 4,184 |
| Return sample | 2010-01-05 through 2026-09-03 |
| Classification | public research |

The CLP rate proxy is PDP/Mindicador TPM 4.50% observed 2026-09-03. The USD proxy is FRED DGS2 4.39% observed 2026-09-02; direct FRED CSV retrieval was used because the PDP FRED `httpx` route exhausted its timeout/retry policy. Both rates are flat proxies, not curves. Yahoo observations expose retrieval time but not release/vintage time. Weekend and quarantined-row bridges are reported explicitly, but each retained bridged return is treated as one observation and annualized at 252 rather than rescaled by its calendar gap.

## Statistical formulation

For S&P level `S` and USDCLP level `X`, compute joint log returns and fit the scalar composite

```text
y_t = 100 * (rS_t + rX_t)
y_t = c[Z_t] + phi1[Z_t] y_(t-1) + phi2[Z_t] y_(t-2) + epsilon_t
epsilon_t | Z_t=i ~ Normal(0, s_i^2)
P(Z_t=j | Z_(t-1)=i) = P_ij
```

Statsmodels estimates the latent regimes. Smoothed regime probabilities then weight the bivariate return means and covariance matrices. Regimes are ordered by annualized composite volatility. The daily row-stochastic transition matrix is converted to a CTMC generator by minimizing

```text
|| expm(Q / 252) - P ||_F
```

subject to nonnegative off-diagonal entries and zero row sums.

### Model-selection result

A staged search selected three regimes under AR(1) by BIC, then AR(2) improved the selected three-regime BIC.

| Specification | BIC | Interpretation |
|---|---:|---|
| AR(1), 1 regime | 14,780.21 | Gaussian autoregression baseline |
| AR(1), 2 regimes | 13,917.26 | Large improvement |
| AR(1), 3 regimes | **13,828.56** | BIC winner across regime count |
| AR(1), 4 regimes | 13,870.01 | Worse BIC; includes near-one-day nuisance states |
| AR(2), 3 regimes | **13,820.64** | Final staged winner |

The final fit kept the best of five converged deterministic-seed attempts. This is staged rather than an exhaustive `(AR order, regime count)` grid.

## Calibrated parameters

| Regime | Occupancy | S&P vol | USDCLP vol | Correlation | AR(1) | AR(2) | Expected duration |
|---|---:|---:|---:|---:|---:|---:|---:|
| Calm | 57.45% | 11.14% | 11.12% | −0.111 | −0.217 | −0.087 | 68.27 trading days |
| Active | 37.21% | 18.29% | 18.80% | 0.013 | −0.274 | −0.096 | 36.59 trading days |
| Crisis | 5.33% | 43.25% | 28.15% | 0.076 | −0.330 | 0.017 | 17.95 trading days |

End-of-sample filtered probabilities are 95.58% calm, 4.17% active, and 0.24% crisis. The CTMC embedding residual is `8.20e-16`.

Diagnostic honesty:

- standardized-innovation Ljung–Box p-value at lag 10: **0.0199** — some serial dependence remains;
- squared-standardized-innovation Ljung–Box p-value: **0.9536** — no remaining ARCH signal detected by this gate;
- Jarque–Bera p-value: **0.6839**;
- a separate bivariate VAR(1) is stable but fails its whiteness test (`p=6.55e-36`), showing that richer lag structure remains available.

## Pricing formulation

Under the CLP domestic pricing measure, the example uses correlated regime-dependent lognormal equity and FX factors. The S&P drift includes the quanto covariance adjustment; USDCLP carries the domestic-minus-foreign rate differential. The historical CTMC generator is reused under the pricing measure, explicitly as a research assumption.

In log states `x=log(S/S0)` and `y=log(X/X0)`, each regime has constant diffusion/advection coefficients. The three PDEs are coupled by `Q`. The implementation uses:

- tensor triangular mesh via `MeshTri.init_tensor`;
- continuous P1 elements;
- block mass/operator matrices and `Q ⊗ M` coupling;
- discounted frozen-regime deterministic-continuation Dirichlet values on the finite boundary (an explicit approximation, especially for product and regime-path-dependent far fields);
- four implicit-Euler half steps replacing the first two Crank–Nicolson steps;
- two cached sparse LU factorizations reused for 160 subsequent steps;
- 81×65 fine mesh, 15,795 regime-stacked degrees of freedom, 326,457 nonzeros;
- independent 400,000-path, 126-step seeded Monte Carlo.

## Six-month prices as of 2026-09-03

Spot inputs are S&P 500 `7,747.70996` and USDCLP `937.59998`; the composite is CLP `7,264,252.67`. Values are in CLP.

| Contract | Fine scikit-fem | Richardson* | Monte Carlo ± 2 SE | Gate |
|---|---:|---:|---:|---|
| ATM composite call | 513,505.98 | **510,592.41** | 508,611.54 ± 2,469.76 | heuristic falls within ±2 SE; fine grid does not |
| ATM composite put | 351,943.45 | **348,971.83** | 348,008.73 ± 1,722.01 | heuristic falls within ±2 SE; fine grid does not |
| CLP 1m composite digital | 543,216.36 | — | 521,552.06 ± 1,542.51 | **fail — diagnostic only** |
| ATM fixed-FX quanto call | 396,827.62 | **398,836.41** | 397,828.40 ± 1,824.91 | heuristic falls within ±2 SE |
| CLP 1m dual-trigger protection | 37,015.61 | — | 29,723.64 ± 530.84 | **fail — diagnostic only** |

`*` “Richardson” is a **heuristic two-level** `fine + (fine−coarse)/3` extrapolation. It assumes second-order error from one ratio-two coarse/fine pair; there is no third level establishing observed order, and payoff kinks weaken pointwise P1 convergence. It must not be read as a validated convergence result. The fine-to-expanded-domain ATM **call only** changes by `2.13e-08` CLP, so truncation is negligible for that one contract at this resolution; no equivalent domain study was executed for the other four. Linear solve residuals are below `4.5e-16`.

The discontinuous products expose a real limitation: nodal continuous P1 initialization plus a payoff jump can converge slowly or misleadingly even with Rannacher smoothing. Their Monte Carlo discrepancy is not hidden. A production route needs discontinuity-aligned/adaptive meshes, projection or conservative stabilization, and a fresh convergence/parity study.

### Volatility model-risk sensitivity

Using the coarse mesh for a transparent sensitivity, the ATM composite call moves from CLP 440,475 at 0.8× historical volatility to CLP 522,247 at 1.0× and CLP 624,863 at 1.25×. Historical volatility is not an implied-volatility calibration.

## Scope and non-claims

1. `Q^P = Q^Q` is assumed, not estimated; regime-risk premia are absent.
2. Dividend yield is set to zero and rates are flat proxies.
3. The scalar Markov likelihood does not jointly estimate the two-dimensional covariance process.
4. Historical-close calibration does not reproduce an option surface.
5. The raw fine-grid call and put do not fall inside the Monte Carlo ±2 SE bands; only the explicitly heuristic two-level extrapolations do. Discontinuous-payoff FEM values fail parity outright.
6. This example demonstrates a reproducible research vertical slice, not production, compliance, or tradable-price maturity.

## Executable evidence

Implementation:

- `src/finite_element_options/examples/regime_switching_quanto/quality.py`
- `src/finite_element_options/examples/regime_switching_quanto/fitting.py`
- `src/finite_element_options/examples/regime_switching_quanto/generator.py`
- `src/finite_element_options/examples/regime_switching_quanto/contracts.py`
- `src/finite_element_options/examples/regime_switching_quanto/fem.py`
- `src/finite_element_options/examples/regime_switching_quanto/monte_carlo.py`

Tests:

- `tests/examples/test_regime_switching_quanto_calibration.py`
- `tests/examples/test_regime_switching_quanto_pricing.py`

The generated evidence bundle contains source/data hashes, quality report, calibration JSON, prices CSV/JSON, reproducible reporting script, dark plots, four equation cards, and a vector PDF. It is deliberately kept outside the package because this repository consumes PDP exports but does not own public data.
