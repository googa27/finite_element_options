# OpenTURNS FEM uncertainty decomposition pilot (#134)

Privacy class: public-synthetic research evidence only. This report covers the optional OpenTURNS UQ pilot for the regime-switching quanto example; it does not promote a production pricing capability or consume PDP internals.

## Decision

Decision: **retain OpenTURNS as an optional, experimental UQ adapter** (`finite-element-options[uncertainty]`) because the canonical run passed all evidence gates:

- real one-regime fixed-FX quanto FEM propagation was reproducible;
- direct NumPy reference parity passed with declared pooled-null sampling tolerances;
- a cheap additive model recovered known Sobol variance shares using OpenTURNS/Saltelli;
- raw real-FEM Saltelli point estimates and confidence intervals passed finite-sample sanity checks;
- every component, grid, model, payoff, and calibration source is hash-bound;
- OpenTURNS global RNG state is serialized and restored on success, failure, and concurrent calls.

The lightweight NumPy route remains the direct-reference baseline because it samples the same five normalized marginals and evaluates the same FEM response without an optional dependency. OpenTURNS uniquely adds maintained distribution composition, Sobol design generation, raw Saltelli first/total finite-sample estimators, and confidence-interval diagnostics. The maturity remains `experimental_optional_non_production`; no base/core dependency and no capability-matrix maturity change.

## Dependency boundary

Package extra: `uncertainty = ["openturns>=1.27,<2"]`.

Observed execution used OpenTURNS `1.27.post1`. The adapter imports OpenTURNS lazily via `require_optional("openturns")`; missing dependency errors name `finite-element-options[uncertainty]` exactly. Public contracts/results are immutable dataclasses containing only JSON-safe values; no OpenTURNS objects or reprs cross the boundary.

OpenTURNS RNG is process-global, so `uncertainty.openturns_adapter` wraps all OpenTURNS seeded calls in a re-entrant lock, saves `RandomGenerator.GetState()`, calls `RandomGenerator.SetSeed(...)`, and restores `RandomGenerator.SetState(...)` in `finally`.

## Canonical artifact

- Artifact: `docs/evidence/regime_switching_quanto_openturns_uq_2026-09-04.json`
- Artifact SHA-256: `f8c733faf469f15d99dbcea160e26ffc87b8f33d25be119be5f184bbdeea2460`
- Canonical input hash: `89b931ad4c307c03367a98a03fdf6e0c5b0da0828be9b458a0405d0cd7836457`
- CLI: `python scripts/run_openturns_uq_pilot.py --output docs/evidence/regime_switching_quanto_openturns_uq_2026-09-04.json --verify`

The CLI verifies predecessor evidence before execution. The baseline model/payoff conventions derive from `docs/evidence/regime_switching_quanto_quantlib_oracle_2026-09-04.json` SHA `ca2789e8f686a2f25b9abebc076f18ce7596673b038e52b681478cad22c4a056` (`quanto_positive_correlation`). The iminuit predecessor artifact SHA `6294b52e9d6aa26aeda39a1809486272223d41ecc7a00e42e670f5dcbba39a3b` is verified as a sequencing prerequisite but is not used as a parameter source.

## Scientific design

Response model: the existing `price_contract_fem` regime-switching-quanto scikit-fem solver with **one regime**, a fixed-FX quanto call, and the fine grid below. No analytical surrogate is used for propagation. The combined output distribution is an **estimator/validation diagnostic**, not a risk-neutral payoff distribution and not intrinsic fair-value uncertainty.

Baseline:

- maturity: `458/365 = 1.254794520547945` years;
- equity spot: `100.0`;
- strike: `105.0`;
- fixed FX payout multiplier: `850.0`;
- domestic/foreign rates: `0.035 / 0.015`;
- dividend yield: `0.010`;
- equity volatility: `0.20`;
- FX volatility: `0.12`;
- full quanto correlation endpoint: `0.35`;
- CTMC generator/probability: one absorbing regime, `[[0.0]]`, `[1.0]`.

Model-form mapping: normalized `z_model_form ~ U(-1,1)` maps to `weight=(z_model_form+1)/2`. `weight=0` is the `zero_correlation_independent_equity_fx_generator` endpoint; `weight=1` is the `full_quanto_correlation_generator` endpoint; correlation used by the FEM solver is `rho = 0.35 * weight`. This is a continuous homotopy between zero-coupling and full-coupling generators, not a posterior probability over models.

## Component table

| Component | Normalized marginal and mapping | Role | Source |
|---|---|---|---|
| `data` | `z_data ~ U(-1,1)`, `spot = 100 * (1 + 0.05*z_data)` | FEM perturbation | public-synthetic spot/input-state band |
| `parameter` | `z_parameter ~ U(-1,1)`, `sigmaS = 0.20 * (1 + 0.15*z_parameter)` | FEM perturbation | equity-volatility-only band |
| `model_form` | `z_model_form ~ U(-1,1)`, correlation inclusion `weight=(z+1)/2` | FEM perturbation | zero-correlation vs full-quanto coupling interpolation |
| `numerical` | `z_numerical ~ U(-1,1)`, additive `half_width*z_numerical` | additive validation-estimator error | coarse-vs-fine FEM discrepancy |
| `monte_carlo` | `z_monte_carlo ~ N(0,1)`, additive `standard_error*z_monte_carlo` | additive validation-estimator error | seeded direct MC standard error |

Numerical error is **not included** in parameter uncertainty. Monte Carlo error is **estimator uncertainty** in the seeded direct MC validation estimator, not intrinsic fair-value uncertainty.

## Calibration before propagation

Fine grid identity: Lagrange-P1 triangular tensor grid, `nx=31`, `ny=7`, `time_steps=16`, domain `x=[-1.6,1.6]`, `y=[-0.7,0.7]`, four Rannacher half-steps then Crank-Nicolson.

Coarse grid identity: same domain/element, `nx=21`, `ny=5`, `time_steps=10`.

Calibration values:

- baseline fine FEM price: `5711.946360858297`;
- baseline coarse FEM price: `5679.906373622954`;
- numerical half-width formula: `max(1.5 * abs(fine_fem_price - coarse_fem_price), 1e-12)`;
- numerical half-width: `48.05998085301553`;
- MC calibration: seed `134011`, paths `4096`, steps/year `32`, realized steps `41`;
- MC calibration price: `5507.523029993026`;
- MC standard error: `166.43859006226404`.

## Propagation results

OpenTURNS propagation controls: sample seed `134101`, sample size `64`, Sobol seed `134201`, Sobol base size `128`; all `64` propagated prices were finite. The canonical runner wrote the artifact in `19.083011` seconds in the local verification environment.

Price summary:

| Statistic | Value |
|---|---:|
| mean | `5575.921813648002` |
| std | `1261.6666138891171` |
| q01 | `3195.7187677162465` |
| q05 | `3712.7287270361494` |
| median | `5514.233154184234` |
| q95 | `7728.005162008185` |
| q99 | `7976.752792904593` |

OpenTURNS/Saltelli results below are **raw finite-sample estimators**, not constrained physical Sobol values. Small negative values and confidence intervals that cross zero are accepted sampling noise in this pilot when finite and within the declared sanity envelope for point estimates.

| Component | Raw first | First 95% CI | Raw total | Total 95% CI | Standalone variance |
|---|---:|---:|---:|---:|---:|
| `data` | `0.643108983272484` | `[0.44096304489700544, 0.8452549216479626]` | `0.7480793278673056` | `[0.5913488422972422, 0.904809813437369]` | `1252607.0498698635` |
| `parameter` | `0.2266460256514087` | `[0.0611779906865349, 0.3921140606162825]` | `0.2591108219249379` | `[0.15540183744490443, 0.36281980640497136]` | `390695.6500075577` |
| `model_form` | `-0.019071398272156582` | `[-0.18021024573086622, 0.14206744918655304]` | `0.029784085341486986` | `[0.008931262567658228, 0.050636908115315744]` | `14690.954139377493` |
| `numerical` | `-0.03014132278088872` | `[-0.19789793599100935, 0.13761529042923193]` | `0.00019626540807452078` | `[-0.005461504863314352, 0.005854035679463393]` | `843.6043926635601` |
| `monte_carlo` | `-0.02351920804591386` | `[-0.1907998177347331, 0.14376140164290535]` | `-0.00047658364820923617` | `[-0.032258565961463564, 0.03130539866504509]` | `29032.749639657857` |

Sobol validation gate:

- point sanity envelope: `[-0.25, 1.25]`;
- point violations: none;
- non-finite point estimates: none;
- interval bound failures: none;
- interval bounds outside the point envelope: none.

The standalone component variance estimates vary one normalized input at a time around zero with a deterministic design. They are not Sobol indices and do not include interactions.

## Direct NumPy reference parity

Direct reference controls: seed `134301`, size `64`, same five marginals, same FEM response, independent random sequence.

Tolerances are statistical rather than bitwise because OpenTURNS and NumPy generate independent sequences. Mean tolerance is `3*sqrt(var_ot/n_ot + var_np/n_np)`. Std tolerance uses a normal-theory 3-sigma standard error for sample standard deviations. Quantile tolerances use a fixed-seed pooled-null bootstrap 99.5% envelope: pool the two empirical samples, resample two independent size-n samples from that null distribution, then add one combined mean standard error.

Parity differences all passed:

- mean difference `202.6888442428708` <= tolerance `637.451895490632`;
- std difference `122.47337992861435` <= tolerance `454.3098273286806`;
- q01 difference `267.14510194892773` <= tolerance `1062.4656437861463`;
- q05 difference `253.98278938563635` <= tolerance `1019.3575536088321`;
- median difference `452.85960160579725` <= tolerance `993.4658605220817`;
- q95 difference `8.255430176451227` <= tolerance `1111.4012813417735`;
- q99 difference `99.37602687454182` <= tolerance `947.5651564253385`.

## Additive Sobol recovery

Cheap synthetic additive model coefficients `[2.0, 1.0, 0.5, 0.0, 1.5]` over the same marginals have known variance shares:

- expected: `data=0.3333333333333333`, `parameter=0.08333333333333333`, `model_form=0.020833333333333332`, `numerical=0.0`, `monte_carlo=0.5625`;
- estimated first: `data=0.3133217438575161`, `parameter=0.09774296682873383`, `model_form=0.01653721956850784`, `numerical=-0.004215718653051791`, `monte_carlo=0.5943025974033155`;
- estimated total: `data=0.3260133928962037`, `parameter=0.07650008228450561`, `model_form=0.018182041449111855`, `numerical=-1.3350837932513583e-07`, `monte_carlo=0.5839978282444354`;
- max first-order error: `0.03180259740331548`;
- max total-order error: `0.021497828244435357`;
- tolerance: `0.08`;
- gate: pass.

## Verification commands

```text
uv run --extra uncertainty python scripts/run_openturns_uq_pilot.py --output docs/evidence/regime_switching_quanto_openturns_uq_2026-09-04.json --verify
uv run --extra uncertainty pytest -q tests/examples/test_regime_switching_quanto_openturns_uq.py --no-cov
python scripts/check_ai_hierarchy_policy.py
python scripts/check_ci_contract.py
```
