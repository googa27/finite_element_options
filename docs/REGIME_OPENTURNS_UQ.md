# OpenTURNS FEM uncertainty decomposition pilot (#134)

Privacy class: public-synthetic research evidence only. This report covers the optional OpenTURNS UQ pilot for the regime-switching quanto example; it does not promote a production pricing capability or consume PDP internals.

## Decision

Decision: **retain OpenTURNS as an optional, experimental UQ adapter** (`finite-element-options[uncertainty]`) because the canonical run passed all evidence gates:

- real one-regime fixed-FX quanto FEM propagation was reproducible;
- direct NumPy reference parity passed with declared pooled-null sampling tolerances;
- a cheap additive model recovered known Sobol variance shares using OpenTURNS/Saltelli;
- raw real-FEM Saltelli point estimates and confidence intervals passed finite-sample sanity checks;
- every component, grid, model, payoff, and calibration source is hash-bound; nested public evidence metadata is recursively immutable and `to_dict()` emits fresh JSON-safe copies;
- OpenTURNS global RNG state is restored for success, failure, and concurrent calls that coordinate through the shared public `openturns_seeded(...)` context; uncoordinated direct same-process calls are outside this guarantee.

The lightweight NumPy route remains the direct-reference baseline because it samples the same five normalized marginals and evaluates the same FEM response without an optional dependency. OpenTURNS uniquely adds maintained distribution composition, Sobol design generation, raw Saltelli first/total finite-sample estimators, and confidence-interval diagnostics. The maturity remains `experimental_optional_non_production`; no base/core dependency and no capability-matrix maturity change.

## Dependency boundary

Package extra: `uncertainty = ["openturns>=1.27,<2"]`.

Observed execution used OpenTURNS `1.27.post1`. That release exposes `JointDistribution`, which the canonical artifact records as the constructor actually used; the adapter will select and report `ComposedDistribution` only when a future compatible OpenTURNS release exposes it. The adapter imports OpenTURNS lazily via `require_optional("openturns")`; missing dependency errors name `finite-element-options[uncertainty]` exactly. Public contracts/results are immutable dataclasses containing only JSON-safe values; no OpenTURNS objects cross the boundary.

OpenTURNS RNG is process-global, so `uncertainty.openturns_adapter` wraps seeded calls in the shared public `openturns_seeded(...)` re-entrant-lock context, saves `RandomGenerator.GetState()`, calls `RandomGenerator.SetSeed(...)`, and restores `RandomGenerator.SetState(...)` in `finally`. All same-process callers that need this guarantee must use that context; uncoordinated direct OpenTURNS calls can still interfere, so such workloads require separate-process isolation. This deliberately serializes coordinated OpenTURNS use within one interpreter; reproducibility takes priority over thread-level parallelism for this bounded pilot.

## Canonical artifact

- Artifact: `docs/evidence/regime_switching_quanto_openturns_uq_2026-09-04.json`
- Artifact SHA-256: `c7e90d2857b9d43da5ed65221c2b85969aecfb62a2be45966bade82cd336b6a3`
- Canonical input hash: `e6a668c45eba43618c21451febd775cffb1d351c08c544826acce705e2160ecb`
- CLI: `python scripts/run_openturns_uq_pilot.py --output docs/evidence/regime_switching_quanto_openturns_uq_2026-09-04.json --verify`

The CLI verifies predecessor evidence before execution. The baseline model/payoff conventions derive from `docs/evidence/regime_switching_quanto_quantlib_oracle_2026-09-04.json` SHA `ca2789e8f686a2f25b9abebc076f18ce7596673b038e52b681478cad22c4a056` (`quanto_positive_correlation`). The iminuit predecessor artifact SHA `6294b52e9d6aa26aeda39a1809486272223d41ecc7a00e42e670f5dcbba39a3b` is verified as a sequencing prerequisite but is not used as a parameter source.

The installed-wheel API does not require repository-only `docs/evidence/` resources: `run_openturns_uq_pilot()` with no `root` executes the FEM/UQ study, records predecessor provenance as `declared_digest_only`, and fails the retention decision closed because those files were not independently verified. Canonical evidence generation passes an explicit checkout root, records `file_sha256`, and requires both predecessor digests to match.

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
- baseline midpoint correlation: `0.175`;
- CTMC generator/probability: one absorbing regime, `[[0.0]]`, `[1.0]`.

Model-form mapping: normalized `z_model_form ~ U(-1,1)` maps to `weight=(z_model_form+1)/2`. `weight=0` is the `zero_correlation_independent_equity_fx_generator` endpoint; `weight=1` is the `full_quanto_correlation_generator` endpoint; correlation used by the FEM solver is `rho = 0.35 * weight`. This is a continuous homotopy between zero-coupling and full-coupling generators, not a posterior probability over models.

## Component table

| Component | Normalized marginal and mapping | Role | Source |
|---|---|---|---|
| `data` | `z_data ~ U(-1,1)`, `spot = 100 * (1 + 0.05*z_data)` | FEM perturbation | public-synthetic spot/input-state band |
| `parameter` | `z_parameter ~ U(-1,1)`, `sigmaS = 0.20 * (1 + 0.15*z_parameter)` | FEM perturbation | equity-volatility-only band |
| `model_form` | `z_model_form ~ U(-1,1)`, correlation inclusion `weight=(z+1)/2` | FEM perturbation | zero-correlation vs full-quanto coupling interpolation |
| `numerical` | `z_numerical ~ U(-1,1)`, additive `domain_half_width*z_numerical` | independent additive validation-estimator error | 275-point declared-domain analytical-oracle screen plus 10% safety margin |
| `monte_carlo` | `z_monte_carlo ~ N(0,1)`, additive `standard_error*z_monte_carlo` | additive validation-estimator error | seeded direct MC standard error |

Numerical error is **not included** in parameter uncertainty. It uses an independent normalized coordinate and one fixed conservative envelope calibrated across the declared spot/volatility/correlation domain, preserving an additive first-order component rather than coupling its scale to those inputs. Monte Carlo error is **estimator uncertainty** in the seeded direct MC validation estimator, not intrinsic fair-value uncertainty.

## Calibration before propagation

Fine grid identity: Lagrange-P1 triangular tensor grid, `nx=31`, `ny=7`, `time_steps=16`, domain `x=[-1.6,1.6]`, `y=[-0.7,0.7]`, four Rannacher half-steps then Crank-Nicolson.

Coarse grid identity: same domain/element, `nx=21`, `ny=5`, `time_steps=10`.

Calibration values:

- baseline fine FEM price: `5711.946360858297`;
- baseline coarse FEM price: `5679.906373622954`;
- exact one-regime analytical-oracle price at the midpoint correlation: `5615.51290798328`;
- fine/coarse absolute oracle errors: `96.43345287501779 / 64.3934656396741`;
- domain screen: `11 x 5 x 5 = 275` tensor points over spot `[95,105]`, sigma `[0.17,0.23]`, and correlation weight `[0,1]`;
- maximum screened fine-FEM/oracle error: `123.19714346893579` at spot `100`, sigma `0.17`, correlation weight `0`;
- domain-screen grid hash: `486f1ca0ffa9f29949c34d263a4742f0f50c049f571507f3824d98abc6b71d32`;
- numerical half-width formula: `max(abs(fine - oracle), abs(coarse - oracle), 1.5*abs(fine - coarse), 1.10*max_domain_grid_error, 1e-12)`;
- fixed independent numerical half-width: `135.51685781582938`;
- analytical-oracle source hash: `c09a86ec3e9f83f522c73dc5bd798fd2c7cab5f55ceab6d24e68219303228b21`;
- MC calibration: seed `134011`, paths `4096`, steps/year `32`, realized steps `41`;
- MC calibration price: `5507.523029993026`;
- MC standard error: `166.43859006226404`.

## Propagation results

OpenTURNS propagation controls: sample seed `134101`, sample size `64`, Sobol seed `134201`, Sobol base size `128`; all `64` propagated prices were finite.

Price summary:

| Statistic | Value |
|---|---:|
| mean | `5567.332938195312` |
| std | `1265.7503663828593` |
| q01 | `3213.4421186874547` |
| q05 | `3712.1224328663984` |
| median | `5492.149141747067` |
| q95 | `7704.743763785033` |
| q99 | `7969.336693611863` |

OpenTURNS/Saltelli results below are **raw finite-sample estimators**, not constrained physical Sobol values. Small negative values and confidence intervals that cross zero are accepted sampling noise in this pilot when finite and when point estimates remain inside the family-specific sanity envelopes.

| Component | Raw first | First 95% CI | Raw total | Total 95% CI | Standalone variance |
|---|---:|---:|---:|---:|---:|
| `data` | `0.6380523643168031` | `[0.4375495546575634, 0.8385551739760428]` | `0.7470347598004511` | `[0.5895140419771316, 0.9045554776237705]` | `1083790.755181209` |
| `parameter` | `0.21912162636033178` | `[0.05355207054511796, 0.3846911821755456]` | `0.2638696927854309` | `[0.16106031202567242, 0.36667907354518936]` | `396464.8514628893` |
| `model_form` | `-0.02420195040569668` | `[-0.18565115137673008, 0.13724725056533674]` | `0.029441674542451855` | `[0.008623319875681792, 0.050260029209221915]` | `15962.866796680411` |
| `numerical` | `-0.023414191571266504` | `[-0.19137769769391816, 0.14454931455138514]` | `0.003104526074151346` | `[-0.012777644323068668, 0.01898669647137136]` | `5273.7642821064` |
| `monte_carlo` | `-0.029629947462902727` | `[-0.19703909871364456, 0.1377792037878391]` | `-0.0002476570902998604` | `[-0.03171516253764713, 0.03121984835704741]` | `22639.199033575667` |

Sobol validation gate:

- first-order point sanity envelope: `[-0.05, 1.0]`;
- total-order point sanity envelope: `[-0.05, 1.05]`;
- point violations: none;
- non-finite point estimates: none;
- interval bound failures: none;
- three first-order confidence intervals extend below the point-estimate envelope; this is reported, not used as a point-estimate gate.

The standalone component variance estimates vary one normalized input at a time around zero using seed `134401` and draws from the same OpenTURNS marginals used by propagation. They are not Sobol indices and do not include interactions.

## Direct NumPy reference parity

Direct reference controls: seed `134301`, size `64`, same five marginals, same FEM response, independent random sequence.

Tolerances are statistical rather than bitwise because OpenTURNS and NumPy generate independent sequences. Mean tolerance is `3*sqrt(var_ot/n_ot + var_np/n_np)`. Std tolerance uses a normal-theory 3-sigma standard error for sample standard deviations. Quantile tolerances use a fixed-seed pooled-null bootstrap 99.5% envelope: pool the two empirical samples, resample two independent size-n samples from that null distribution, then add one combined mean standard error.

Parity differences all passed:

- mean difference `216.24785932520717` <= tolerance `638.6544718805808`;
- std difference `126.29755935412345` <= tolerance `455.16689948727344`;
- q01 difference `215.11178477208705` <= tolerance `1052.9584210388837`;
- q05 difference `263.5832184299261` <= tolerance `1065.9547184867517`;
- median difference `456.6200887164814` <= tolerance `997.7314513958579`;
- q95 difference `9.447411720895616` <= tolerance `1144.1865519394996`;
- q99 difference `122.27674939428562` <= tolerance `909.9598862061035`.

## Additive Sobol recovery

Cheap synthetic additive model coefficients `[2.0, 1.0, 0.5, 0.0, 1.5]` over the same marginals have known variance shares:

- expected: `data=0.3333333333333333`, `parameter=0.08333333333333333`, `model_form=0.020833333333333332`, `numerical=0.0`, `monte_carlo=0.5625`;
- estimated first: `data=0.3133217438575161`, `parameter=0.09774296682873383`, `model_form=0.01653721956850784`, `numerical=-0.004215718653051791`, `monte_carlo=0.5943025974033155`;
- estimated total: `data=0.3260133928962037`, `parameter=0.07650008228450561`, `model_form=0.018182041449111855`, `numerical=-1.3350837932513583e-07`, `monte_carlo=0.5839978282444354`;
- max first-order error: `0.03180259740331548`;
- max total-order error: `0.021497828244435357`;
- tolerance: `0.08`;
- gate: pass.

## Limitations

- The fixed numerical-error envelope is calibrated on a declared 275-point tensor screen with a 10% safety margin; it covers every screened point but is not a mathematical uniform-error proof between grid nodes or a reusable multi-regime error model.
- The pilot's coarse/fine grids are not claimed to form a convergent sequence; analytical-oracle domain screening, rather than refinement difference alone, controls the independent numerical envelope.
- The Monte Carlo standard error comes from one declared seed/path budget and is validation-estimator uncertainty only.
- The model-form coordinate is a continuous generator homotopy, not a calibrated posterior over discrete model classes.
- Sample sizes and asymptotic Saltelli intervals are appropriate only for an experimental screening pilot; the raw estimates are not production sensitivity estimates.
- OpenTURNS calls coordinated through `openturns_seeded(...)` are serialized inside one interpreter because its RNG state is process-global; uncoordinated direct calls require process isolation.

## Verification commands

```text
uv run --extra uncertainty python scripts/run_openturns_uq_pilot.py --output docs/evidence/regime_switching_quanto_openturns_uq_2026-09-04.json --verify
uv run --extra uncertainty pytest -q tests/examples/test_regime_switching_quanto_openturns_uq.py --no-cov
python scripts/check_ai_hierarchy_policy.py
python scripts/check_ci_contract.py
```
