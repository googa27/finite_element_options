# OpenTURNS FEM uncertainty decomposition pilot (#134)

Privacy class: public-synthetic research evidence only. This report covers the optional OpenTURNS UQ pilot for the regime-switching quanto example; it does not promote a production pricing capability or consume PDP internals.

## Decision

Decision: **retain OpenTURNS as an optional, experimental UQ adapter** (`finite-element-options[uncertainty]`) because the canonical run passed all evidence gates:

- real one-regime fixed-FX quanto FEM propagation was reproducible;
- direct NumPy reference parity passed with declared pooled-null sampling tolerances;
- a cheap additive model recovered known Sobol variance shares using OpenTURNS/Saltelli;
- raw real-FEM Saltelli point estimates and confidence intervals passed finite-sample sanity checks;
- every component, grid, model, payoff, and calibration source is hash-bound;
- OpenTURNS global RNG state is restored for success, failure, and concurrent calls that coordinate through the shared public `openturns_seeded(...)` context; uncoordinated direct same-process calls are outside this guarantee.

The lightweight NumPy route remains the direct-reference baseline because it samples the same five normalized marginals and evaluates the same FEM response without an optional dependency. OpenTURNS uniquely adds maintained distribution composition, Sobol design generation, raw Saltelli first/total finite-sample estimators, and confidence-interval diagnostics. The maturity remains `experimental_optional_non_production`; no base/core dependency and no capability-matrix maturity change.

## Dependency boundary

Package extra: `uncertainty = ["openturns>=1.27,<2"]`.

Observed execution used OpenTURNS `1.27.post1`. That release exposes `JointDistribution`, which the canonical artifact records as the constructor actually used; the adapter will select and report `ComposedDistribution` only when a future compatible OpenTURNS release exposes it. The adapter imports OpenTURNS lazily via `require_optional("openturns")`; missing dependency errors name `finite-element-options[uncertainty]` exactly. Public contracts/results are immutable dataclasses containing only JSON-safe values; no OpenTURNS objects cross the boundary.

OpenTURNS RNG is process-global, so `uncertainty.openturns_adapter` wraps seeded calls in the shared public `openturns_seeded(...)` re-entrant-lock context, saves `RandomGenerator.GetState()`, calls `RandomGenerator.SetSeed(...)`, and restores `RandomGenerator.SetState(...)` in `finally`. All same-process callers that need this guarantee must use that context; uncoordinated direct OpenTURNS calls can still interfere, so such workloads require separate-process isolation. This deliberately serializes coordinated OpenTURNS use within one interpreter; reproducibility takes priority over thread-level parallelism for this bounded pilot.

## Canonical artifact

- Artifact: `docs/evidence/regime_switching_quanto_openturns_uq_2026-09-04.json`
- Artifact SHA-256: `0806dd9dcef4663207b101a6c9f8c463598a0907f1cbc42dfeaf70c96d4b2fe3`
- Canonical input hash: `c124653a04ef0e1137ebf1af7299cbdc3f8179fde527427f22161a10f1e43a09`
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
| `numerical` | `z_numerical ~ U(-1,1)`, additive `abs(fine_FEM(input)-analytical_oracle(input))*z_numerical` | input-dependent additive validation-estimator error | exact one-regime analytical oracle at each propagated input plus baseline coarse/fine evidence |
| `monte_carlo` | `z_monte_carlo ~ N(0,1)`, additive `standard_error*z_monte_carlo` | additive validation-estimator error | seeded direct MC standard error |

Numerical error is **not included** in parameter uncertainty. Its scale is evaluated at every propagated spot/volatility/correlation input, so it may interact with those inputs instead of reusing the baseline scalar. Monte Carlo error is **estimator uncertainty** in the seeded direct MC validation estimator, not intrinsic fair-value uncertainty.

## Calibration before propagation

Fine grid identity: Lagrange-P1 triangular tensor grid, `nx=31`, `ny=7`, `time_steps=16`, domain `x=[-1.6,1.6]`, `y=[-0.7,0.7]`, four Rannacher half-steps then Crank-Nicolson.

Coarse grid identity: same domain/element, `nx=21`, `ny=5`, `time_steps=10`.

Calibration values:

- baseline fine FEM price: `5711.946360858297`;
- baseline coarse FEM price: `5679.906373622954`;
- exact one-regime analytical-oracle price at the midpoint correlation: `5615.51290798328`;
- fine/coarse absolute oracle errors: `96.43345287501779 / 64.3934656396741`;
- numerical half-width formula: `max(abs(fine - oracle), abs(coarse - oracle), 1.5 * abs(fine - coarse), 1e-12)`;
- numerical half-width: `96.43345287501779` (baseline descriptor only);
- propagated numerical-error formula: `abs(fine_FEM(input)-analytical_oracle(input))*z_numerical`, evaluated at every input;
- analytical-oracle source hash: `c09a86ec3e9f83f522c73dc5bd798fd2c7cab5f55ceab6d24e68219303228b21`;
- MC calibration: seed `134011`, paths `4096`, steps/year `32`, realized steps `41`;
- MC calibration price: `5507.523029993026`;
- MC standard error: `166.43859006226404`.

## Propagation results

OpenTURNS propagation controls: sample seed `134101`, sample size `64`, Sobol seed `134201`, Sobol base size `128`; all `64` propagated prices were finite.

Price summary:

| Statistic | Value |
|---|---:|
| mean | `5573.963082795815` |
| std | `1261.284499263254` |
| q01 | `3192.151858256343` |
| q05 | `3703.427627456702` |
| median | `5502.894944029444` |
| q95 | `7741.802591881391` |
| q99 | `7963.839913420134` |

OpenTURNS/Saltelli results below are **raw finite-sample estimators**, not constrained physical Sobol values. Small negative values and confidence intervals that cross zero are accepted sampling noise in this pilot when finite and when point estimates remain inside the family-specific sanity envelopes.

| Component | Raw first | First 95% CI | Raw total | Total 95% CI | Standalone variance |
|---|---:|---:|---:|---:|---:|
| `data` | `0.6406739567470172` | `[0.43892007358534924, 0.8424278399086852]` | `0.7478764707010801` | `[0.5917043127677847, 0.9040486286343755]` | `1083790.755181209` |
| `parameter` | `0.22647574084626937` | `[0.06161365504791569, 0.391337826644623]` | `0.26026930266440695` | `[0.1566251881114778, 0.3639134172173361]` | `396464.8514628893` |
| `model_form` | `-0.018927819130213387` | `[-0.17976702019725074, 0.14191138193682398]` | `0.029523493495423873` | `[0.008687645682116229, 0.050359341308731514]` | `15962.866796680411` |
| `numerical` | `-0.029185995466321217` | `[-0.1957827656687558, 0.13741077473611335]` | `0.0050095166348126975` | `[-0.0013355231761408518, 0.011354556445766247]` | `2670.4810626966723` |
| `monte_carlo` | `-0.023041945256812076` | `[-0.1897635223769182, 0.14367963186329405]` | `-0.0005141361903830875` | `[-0.032290451098936475, 0.031262178718170296]` | `22639.199033575667` |

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

- mean difference `204.72876496393292` <= tolerance `637.637052095383`;
- std difference `120.93185593180783` <= tolerance `454.4417878197048`;
- q01 difference `276.1570715048406` <= tolerance `1062.9840027187274`;
- q05 difference `261.7973863761358` <= tolerance `1020.5478656437019`;
- median difference `467.5672212565505` <= tolerance `976.5915676562073`;
- q95 difference `1.340650878662018` <= tolerance `1134.6593562441192`;
- q99 difference `116.6537456970209` <= tolerance `971.5697787380033`.

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

- Numerical-error scaling uses the exact analytical reduction at every propagated input and therefore applies only to this one-regime fixed-FX diagnostic; it is not a reusable error model for multi-regime PDEs without such an oracle.
- The pilot's coarse/fine grids are not claimed to form a convergent sequence; the input-local analytical discrepancy, rather than refinement difference alone, controls each numerical perturbation.
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
