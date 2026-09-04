# Regime-switching quanto volatility challengers (#131)

Status: adoption-only research benchmark; no maturity promotion.

This report records the immutable benchmark comparing `arch` GJR-GARCH and EGARCH
challengers against the existing statsmodels Markov AR(2) baseline for the
regime-switching equity–FX quanto example. The route consumes a caller-provided
CSV export and never imports PDP internals.

## Reproducibility

Immutable input contract:

- Expected input SHA-256: `aa7ab317266bf37463e27aba9a4e990fa349bb0a6e0aefb5741e93480e0f79f4`
- Columns: `date`, `sp500`, `usdclp`
- Response: `100 * (sp500_log_return + usdclp_log_return)`
- First rolling train window: `2023-02-28` through `2026-03-05`
- Hold-out: `2026-03-06` through `2026-09-03`
- Artifact SHA-256: `3ef33542865cc7370bc639b15b60aba207a2be3981ad77b9d1132f5f0e15f9ad`

Regenerate from any local checkout with the optional research profiles installed:

```bash
python scripts/run_regime_volatility_benchmark.py \
  --input <PDP_JOINT_LEVELS_CSV> \
  --expected-sha256 aa7ab317266bf37463e27aba9a4e990fa349bb0a6e0aefb5741e93480e0f79f4 \
  --output <CANONICAL_OUTPUT_JSON>

python scripts/run_regime_volatility_benchmark.py \
  --input <PDP_JOINT_LEVELS_CSV> \
  --expected-sha256 aa7ab317266bf37463e27aba9a4e990fa349bb0a6e0aefb5741e93480e0f79f4 \
  --output <CANONICAL_OUTPUT_JSON> \
  --verify
```

The checked-in artifact is
`docs/evidence/regime_switching_quanto_volatility_benchmark_2026-09-03.json`.
It is canonical JSON, has no local input path, no current timestamp, no raw time
series, and summarizes quarantined rows and bridged gaps instead of serializing
row-level level data.

## Data-quality bounds

The benchmark used the existing public-synthetic level bounds:

- `sp500`: `[100.0, 20000.0]`
- `usdclp`: `[100.0, 2000.0]`

Summary diagnostics from the canonical artifact:

| Field | Value |
|---|---:|
| input rows | 4346 |
| valid level rows | 4185 |
| return rows | 4184 |
| quarantined row count | 161 |
| bridged return gap count | 159 |
| max bridged calendar gap days | 5 |

Quarantine reasons: `sp500_nonfinite=153`, `usdclp_nonfinite=6`,
`usdclp_out_of_bounds=2`.

## Forecast design

All challenger metrics are held-out. Each rolling refit block uses an identical
training boundary and then one-step-ahead forecasts aligned to the observed
hold-out target. For ARCH challengers, the model object is built with data
through the current block, fit with `last_obs` at the train/hold-out boundary,
and forecast with `horizon=1`, `align="target"`, `method="analytic"`. This avoids
leaking future parameters and avoids replacing Student-t or skewed-t innovations
with Gaussian simulation draws. Predictive log scores and VaR use the fitted
`arch` distribution (`StudentsT` or `skewstudent`) and its fitted shape/skew
parameters.

The Markov AR(2) baseline is fitted with statsmodels on the same rolling blocks.
For each hold-out observation, it propagates filtered regime probabilities with
statsmodels' transition matrix converted explicitly to row-stochastic
previous-regime-to-next-regime orientation, forms AR(2) conditional means from
training or earlier hold-out observations only, emits Gaussian-mixture mean,
variance, log density, and exact mixture VaR, and only then filters regime
probabilities with the just-observed value. Its metrics remain diagnostic when
any baseline fit fails or does not converge; they cannot gate promotion.

## Held-out metrics

| Model | Converged | QLIKE | Mean log score | VaR exceedances | VaR rate | Parameter stability |
|---|---:|---:|---:|---:|---:|---:|
| Markov AR(2), Gaussian mixture | false | 1.2325125394503185 | -1.5419174161882798 | 5 | 0.03968253968253968 | 0.41720016730193454 |
| GJR-GARCH, Student-t | true | 1.214396280136161 | -1.520515388393554 | 4 | 0.031746031746031744 | 0.09111058485088412 |
| GJR-GARCH, skewed-t | true | 1.214010621890624 | -1.518739638805828 | 3 | 0.023809523809523808 | 0.09463658337600428 |
| EGARCH, Student-t | true | 1.218907030475227 | -1.5231085639576043 | 5 | 0.03968253968253968 | 0.049816509368429365 |
| EGARCH, skewed-t | true | 1.2164941339635733 | -1.5198045913289726 | 5 | 0.03968253968253968 | 0.048668962652579216 |

## Changepoint comparison

`ruptures.Pelt(model="rbf")` on rolling volatility/covariance features produced
59 total breakpoints. The JSON artifact serializes a bounded sample of 20
breakpoints and records `breakpoint_count=59` and `breakpoints_truncated=true`.
Against the full-sample highest-variance Markov smoothed-regime probabilities,
the nearest gap was 1 calendar day; 41 of 59 breakpoints were within the configured
63-day window, giving overlap rate `0.6949152542372882`.

The changepoint comparison is descriptive and does not enter the promotion gate.
If full-sample Markov regime probabilities fail, the artifact records a typed
`regime_probability_failure` instead of fabricating zero probabilities.

## Decision

Decision: `reject`

Selected diagnostic challenger: `GJR-GARCH/skewed-t`

Fail-closed reason:

> invalid Markov AR(2) baseline: optimizer one or more fits did not converge;
> challenger promotion is disabled until the baseline converges

The ARCH challengers showed better diagnostic held-out QLIKE/log scores in this
run, but promotion is impossible while the baseline is invalid/nonconverged. The
baseline metrics above are kept only to aid diagnosis.

## Fairness caveats and limitations

- ARCH challenger densities and Markov Gaussian-mixture scores are predictive
  scores on the same scalar response but are not nested likelihoods.
- The Markov block forecast is sequential one-step-ahead. Each step uses only
  fitted parameters, prior filtered regime probabilities, and observed training
  or earlier hold-out AR(2) lags; the current hold-out value updates the filter
  only after that step's forecast and score are emitted. Its VaR is an exact
  Gaussian-mixture quantile, not a normal moment approximation.
- The ARCH route benchmarks scalar composite-return volatility, not a production
  bivariate quanto pricing model.
- Changepoints are descriptive covariance/volatility diagnostics and are not a
  promotion criterion.
- This is an optional adoption study behind `calibration`, `volatility`, and
  `changepoints` extras; it does not change the capability matrix maturity.
