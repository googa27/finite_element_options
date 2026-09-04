# Regime-switching quanto QuantLib oracle slice (#132)

Privacy class: public-synthetic research, architecture, dependency-boundary, and numerical evidence only.

## Scope

This slice adds an isolated `quantlib` adoption oracle for one-regime European calls:

- `vanilla`: standard Black-Scholes-Merton European call.
- `fixed_fx_quanto`: fixed-FX quanto call, priced independently with QuantLib `QuantoVanillaOption` and `QuantoEuropeanEngine`.

It is **not** validation evidence for the full multi-regime regime-switching PDE. The existing research FEM and seeded MC engines are exercised only in a one-regime reduction.

## Public contract and boundary

The public spec/result contracts live under:

```text
finite_element_options.examples.regime_switching_quanto.adoption.quantlib_oracle
```

They contain only stdlib/domain/JSON-safe values: `datetime.date`, strings, floats, bools, lists, and dictionaries. QuantLib handles, dates, calendars, quotes, term structures, processes, options, engines, and exceptions remain inside the adapter. The adoption facade remains lazy, and executing the adapter without QuantLib raises the exact install hint `finite-element-options[quantlib]` for `QuantLib>=1.43,<2`.

Unsupported conventions fail before QuantLib state mutation through `QuantLibConventionError(field, received, supported)`. Vanilla reduction invariants also fail before any QuantLib evaluation-date mutation through JSON-safe `QuantLibReductionError(field, received, expected, kind)`: vanilla requires `foreign_rate == domestic_rate`, `fx_vol == 0`, `correlation == 0`, and `fixed_fx == 1`.

Supported conventions:

| Field | Supported value |
|---|---|
| Calendar | `TARGET` |
| Day count | `Actual365Fixed` |
| Business-day convention | `Unadjusted` |
| Rate compounding | `Continuous` |
| Exercise | `European` |
| Option type | `call` |
| Kind | `vanilla`, `fixed_fx_quanto` |

## Formula and QuantLib engines

Let

```text
adjustment = rho * sigmaS * sigmaFX
q_eff = q + rd - rf + adjustment
```

The existing two-factor model's equity drift is:

```text
rf - q - rho * sigmaS * sigmaFX
```

The equivalent domestic-discounted one-factor BSM yield is `q_eff` above. Vanilla cases set `rf=rd`, `sigmaFX=0`, `rho=0`, and `fixed_fx=1`, so `q_eff=q`.

QuantLib routes:

- Vanilla: `EuropeanOption` + `AnalyticEuropeanEngine` on a `BlackScholesMertonProcess` with domestic flat risk-free curve, ordinary dividend yield, and flat equity Black vol.
- Fixed-FX quanto: `QuantoVanillaOption` + `QuantoEuropeanEngine(process, foreignRiskFreeRate, exchangeRateVolatility, correlation)` with the process carrying domestic risk-free and ordinary dividend curves. QuantLib returns the unscaled foreign-equity payoff NPV, so the public oracle price multiplies by `fixed_fx`.

Flat curves/volatilities are anchored at the explicit evaluation date; the explicit maturity date is used unadjusted. QuantLib's process-global evaluation date is serialized by an `RLock` and restored in `finally` on both success and failure. The context manager accepts stdlib `datetime.date` and converts it only after lazy QuantLib import while preserving existing QuantLib `Date`/sentinel compatibility.

## Canonical artifact

Artifact:

```text
docs/evidence/regime_switching_quanto_quantlib_oracle_2026-09-04.json
```

Artifact SHA-256: `ca2789e8f686a2f25b9abebc076f18ce7596673b038e52b681478cad22c4a056`

Canonical matrix input/spec hash: `123e1535987e14e282e4ab755fe7fdc6e3b73c7764aeb03ba72dab9c7965c934`

Regenerate/verify:

```bash
uv run --extra quantlib python scripts/run_quantlib_oracle_matrix.py \
  --output docs/evidence/regime_switching_quanto_quantlib_oracle_2026-09-04.json
uv run --extra quantlib python scripts/run_quantlib_oracle_matrix.py \
  --output docs/evidence/regime_switching_quanto_quantlib_oracle_2026-09-04.json \
  --verify
```

Verification output recorded during this slice:

```text
artifact verification OK
matrix_spec_hash=123e1535987e14e282e4ab755fe7fdc6e3b73c7764aeb03ba72dab9c7965c934
all_passed=True
```

## Matrix results

| Case | Kind | T (Actual365Fixed) | q_eff | QuantLib | Analytical | FEM | MC ± SE | Gates |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `vanilla_atm_one_year` | vanilla | 1.000000000000 | 0.010000 | 8.827321225352 | 8.827321225352 | 8.783647662261 | 8.849073941425 ± 0.045529007469 | pass |
| `vanilla_otm_six_month` | vanilla | 0.498630136986 | 0.004000 | 3.647899595827 | 3.647899595827 | 3.646790145948 | 3.670403886082 ± 0.027315636874 | pass |
| `quanto_positive_correlation` | fixed-FX quanto | 1.254794520548 | 0.038400 | 5420.764273987145 | 5420.764273987138 | 5414.581396116463 | 5378.701113623561 ± 32.049072915232 | pass |
| `quanto_negative_correlation` | fixed-FX quanto | 0.747945205479 | -0.022680 | 15219.546276302546 | 15219.546276302553 | 15233.045354954824 | 15211.863944758199 ± 56.952055522024 | pass |

Summary metrics:

| Metric | Value |
|---|---:|
| Cases | 4 |
| Vanilla cases | 2 |
| Quanto cases | 2 |
| Max `abs(QuantLib - analytical)` | 7.275957614183426e-12 |
| Max `abs(FEM - analytical)` | 13.499078652270327 |
| Max MC error in standard errors | 1.3124610647812724 |
| Overall gate | pass |

Case gates:

- QuantLib vs analytical tolerance: `1.0e-9` absolute.
- FEM tolerances: `0.05`, `0.03`, `7.0`, and `15.0` absolute by case; the quanto tolerances are below 0.13% of option value and reflect the deliberately small one-regime P1 research grid.
- MC gates: `max(abs_floor, 4 * reported_standard_error)` with seeded paths. This accounts for reported sampling uncertainty and does not treat MC as a deterministic oracle.

## Executable tests and CI

Added/updated gates:

```bash
uv run --extra quantlib pytest -q \
  tests/examples/test_regime_switching_quanto_quantlib_oracle.py \
  tests/examples/test_regime_switching_quanto_adoption_boundaries.py --no-cov
uv run --extra quantlib python scripts/run_quantlib_oracle_matrix.py \
  --output docs/evidence/regime_switching_quanto_quantlib_oracle_2026-09-04.json \
  --verify
```

The `optional_imports` CI matrix now executes the QuantLib oracle/state tests on the `quantlib` profile instead of only importing QuantLib.

## Limitations

- The matrix is one-regime only and does not validate CTMC coupling, calibrated multi-regime parameters, full two-factor PDE behavior, barriers, digitals, American exercise, or production model risk.
- QuantLib is optional and remains outside base dependencies.
- The FEM rows use the existing research P1 two-factor code as a comparison route, not as new maturity evidence for multi-regime pricing.
