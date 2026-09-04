# Iminuit profile-likelihood identifiability diagnostics

Issue: [#133](https://github.com/googa27/finite_element_options/issues/133)
Privacy/scope: public-synthetic research evidence only.

## Scope and terminology

This slice calibrates two parameters (`equity_vol`, `correlation`) to literal supplied fixed-FX quanto call **public-synthetic instrument targets**. The targets are not observed/live market data, not private PDP exports, and not production calibration. The result may use the internal objective label `market_calibration_objective` only because concrete instrument targets are supplied; the evidence should be interpreted as public-synthetic instrument-target calibration.

No capability maturity is promoted. The full multi-regime regime-switching PDE remains outside this evidence.

## Objective

For supplied instruments \(i\), the deterministic weighted least-squares objective is

\[
\chi^2(\sigma,\rho)=\sum_i\left(\frac{P_{BS quanto}(\sigma,\rho;i)-P^{target}_i}{s_i}\right)^2,
\]

with `Minuit.errordef = 1`. The pricing reduction uses the repository analytical Black-Scholes oracle (`EuropeanOptionBs`) and an Actual365Fixed-equivalent year fraction:

\[
T_i = (maturity\_date_i - evaluation\_date_i).days / 365.
\]

The fixed-FX quanto call model price is

\[
P_{BS quanto}=fixed\_fx_i\, C_{BS}(S_i,K_i,T_i,r_{d,i},q_{eff,i},\sigma),
\]

where

\[
q_{eff,i}=q_i+r_{d,i}-r_{f,i}+\rho\,\sigma\,fx\_vol_i.
\]

All dates, spots, strikes, fixed FX rates, target prices, and strictly positive `price_std` values are validated before optimization. Non-finite or out-of-bounds public evaluations return `inf`, record typed diagnostics, and fail closed.

## Bounds and optimizer evidence

Canonical bounds:

- `equity_vol`: `[0.05, 0.60]`, strictly positive;
- `correlation`: `[-0.95, 0.95]`, inside `[-1, 1]`.

The adapter imports `iminuit` lazily through `finite-element-options[identifiability]` and uses the maintained APIs only:

1. `Minuit(objective, equity_vol=..., correlation=...)`;
2. parameter limits and `errordef=1`;
3. `migrad()`;
4. `hesse()`;
5. `minos()` for both parameters;
6. bounded deterministic `mnprofile(..., subtract_min=True, grid=...)` traces.

No raw iminuit object crosses a public contract. Results serialize minimum validity, fval, EDM, nfcn, FMin flags, HESSE covariance quality and symmetric covariance, MINOS intervals/flags, boundary contact, finite-difference gradient/curvature, profile traces, objective diagnostics, and typed failures.

## Identification gate

The gate is fail-closed. `identified=True` requires all of the following:

- finite valid minimum;
- EDM at or below the case threshold;
- no optimizer or configured near-bound contact;
- HESSE covariance present, accurate, positive definite, and not forced positive definite;
- finite positive HESSE standard errors;
- valid two-sided MINOS intervals for every free parameter;
- finite stable profile traces with evidence bracketing \(\Delta\chi^2=1\) on both sides of the minimum;
- finite-difference local positive curvature for both parameters.

A usable point estimate alone is therefore explicitly not sufficient for identification.

## Canonical artifact

Artifact: `docs/evidence/regime_switching_quanto_iminuit_identifiability_2026-09-04.json`

- Artifact SHA-256: `6294b52e9d6aa26aeda39a1809486272223d41ecc7a00e42e670f5dcbba39a3b`
- Study input hash: `a5067f8b5201e2cfee4496dbd480e849822a9d9576cb137d524572ef378c8422`
- Generation/verification command:

```text
uv run --extra identifiability python scripts/run_iminuit_identifiability.py --output docs/evidence/regime_switching_quanto_iminuit_identifiability_2026-09-04.json --verify
```

Verification output:

```text
artifact verification OK
study_input_hash=a5067f8b5201e2cfee4496dbd480e849822a9d9576cb137d524572ef378c8422
all_expected_decisions_passed=True
```

The clean installed-wheel CI profiles rerun the scientific decisions and tolerance assertions on Python 3.11 and 3.12. The committed evidence bytes are guarded separately by their canonical SHA-256 and input hash; full optimizer traces are not required to be bitwise identical across NumPy/SciPy platform builds.

## Results

| Case | Decision | Estimate | HESSE errors | Key evidence |
|---|---:|---|---|---|
| `identified_quanto_surface` | `identified=True` | `equity_vol=0.2300000394`, `correlation=-0.3999994111` | `0.0003480324`, `0.0051604630` | Valid minimum, EDM `1.35e-08`, accurate positive-definite HESSE covariance, valid two-sided MINOS, no bound contact, both bounded profiles finite and bracketing `Delta-chi2=1`. |
| `weak_rho_fxvol_zero` | `identified=False` | usable sigma point estimate `equity_vol=0.2300000000`; rho remains arbitrary near initial `0.2500014547` | zero/missing HESSE errors | All targets have `fx_vol=0`, so `rho` is structurally absent from `q_eff`. HESSE/MINOS evidence fails, rho finite-difference curvature is zero, and the rho profile is flat with `max_delta_chi2=0`. |

Negative weak-identification is the expected and scientifically valid result for the second case.
