# Isolated Python 3.12 Bayesian/JAX Research Profile

## Decision

**Adopt two isolated research extras—`bayesian` and `bayesian-jax`—without expanding the base wheel or claiming automatic differentiation through FEM.**

| Gate | Observed | Required | Result |
|---|---:|---:|:---:|
| Python | `3.12.3` | `>=3.12,<3.13` | PASS |
| Package origin | clean installed wheel | wheel, not checkout/editable | PASS |
| Hash-pinned lock | SHA `1f97148d…e1f7f` | present and verified | PASS |
| PyMC posterior smoke | all 10 checks true | all true | PASS |
| NumPyro posterior smoke | all 10 checks true | all true | PASS |
| JAX runtime route | CPU / 2 devices | documented and decision-gated | PASS |
| Cross-engine posterior mean / SD differences | `0.002` / `0.003` | ≤ `0.04` / `0.02` | PASS |
| Cross-engine predictive mean / SD differences | `0.000194` / `0.00199` | ≤ `0.04` / `0.04` | PASS |
| JAX/FEM differentiation | explicitly unsupported | fail closed | PASS |
| Predecessor continuity | PETSc artifact exact hash | verified | PASS |

This is an environment and diagnostic capability only. It is not market calibration, a production Bayesian model, or a capability-matrix maturity upgrade.

## Evidence

- Artifact: [`evidence/bayesian_jax_profile_2026-09-05.json`](evidence/bayesian_jax_profile_2026-09-05.json)
- Artifact SHA-256: `3a4fde478b4f8b43c8eed31774f2086638105249f2029239ab7cb25d49d2c876`
- Study-input SHA-256: `4271c962f2bc9ab0d4844ac9d27557868e0dcc6eb62a8f4539198352c5d6b6bf`
- Synthetic-data SHA-256: `1f5474034d123b0ee9fd7e67f3ae9c0e37e7ea8babe61e5090f56e6f83c364b3`
- Combined lock: [`../environments/bayesian-jax-py312/requirements.lock`](../environments/bayesian-jax-py312/requirements.lock)
- Combined lock SHA-256: `1f97148d8501965688e450aff6563abd0172c7098c622cf50bd9a0848d9e1f7f`
- PyMC-only lock: [`../environments/bayesian-py312/requirements.lock`](../environments/bayesian-py312/requirements.lock)
- PyMC-only lock SHA-256: `86ef1f8939370f48573bc9ddf2733536c91c58b2ce8d78e2e727ec0b1628004d`
- Predecessor PETSc artifact SHA-256: `f81d29c63625138fd5c1a2ee124c4398b578db80b1a603f113859a0db7dc1368`
- Privacy: `public_synthetic`

## Dependency split

| Profile | Purpose | Dependencies |
|---|---|---|
| base | FEM/analytical core | NumPy, SciPy, scikit-fem, SymPy, Pydantic |
| `calibration` | lightweight deterministic and statsmodels calibration | pandas, statsmodels |
| `bayesian` | native PyMC inference and ArviZ diagnostics | PyMC, ArviZ |
| `jax` | existing narrow JAX Greek experiments | JAX |
| `bayesian-jax` | JAX-native probabilistic inference | PyMC, ArviZ, JAX, NumPyro |

PyMC/ArviZ are no longer installed by the `calibration` extra. This is an explicit `0.2.0` dependency-interface break: migrate Bayesian workflows to Python 3.12 and `[bayesian]` (or `[bayesian-jax]` when NumPyro is required), as recorded in `CHANGELOG.md`. Both Bayesian extras carry `>=3.12,<3.13` dependency markers and their runners fail closed outside Python 3.12. `bayesian-jax` is intentionally distinct from the existing narrow `jax` extra and from scikit-fem/SciPy runtime arrays.

The root `PyMCCalibrator` and `sample_pymc_calibration` names remain as lazy compatibility APIs; when optional dependencies are absent they fail before importing the pandas-backed Heston module and report the `0.2` migration to Python 3.12 plus `[calibration,bayesian]`.

## Recreate from a clean location

```bash
REPO="$PWD"
VENV=/tmp/feo-bayes-wheel
DIST=/tmp/feo-bayes-dist

rm -rf build "$VENV" "$DIST"
uv build --wheel --out-dir "$DIST"
uv venv --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" --require-hashes \
  -r environments/bayesian-jax-py312/requirements.lock
uv pip install --python "$VENV/bin/python" --no-deps \
  "$DIST"/finite_element_options-*.whl
uv pip install --python "$VENV/bin/python" pytest pytest-cov

cd /tmp
PYTHONPATH="" JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=2 \
  "$VENV/bin/python" -m pytest -q \
  "$REPO/external_tests/bayesian_profile" --no-cov

PYTHONPATH="" JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=2 \
  "$VENV/bin/python" "$REPO/scripts/run_bayesian_jax_profile.py" \
  --output "$REPO/docs/evidence/bayesian_jax_profile_2026-09-05.json" --verify
```

The verified wheel resolved:

| Package | Version |
|---|---:|
| CPython | `3.12.3` |
| PyMC | `5.26.1` |
| ArviZ | `0.23.4` |
| JAX / jaxlib | `0.9.2` / `0.9.2` |
| NumPyro | `0.21.0` |
| finite-element-options | `0.2.0` |
| NumPy | `2.5.1` |
| SciPy | `1.18.0` |

The lock is generated from `uv.lock` with:

```bash
uv export --frozen --extra bayesian-jax --no-dev --no-emit-project \
  --format requirements-txt \
  --output-file environments/bayesian-jax-py312/requirements.lock
uv export --frozen --extra bayesian --no-dev --no-emit-project \
  --format requirements-txt \
  --output-file environments/bayesian-py312/requirements.lock
```

`--verify` is semantic: it reruns both stochastic engines and checks stable input/lock/package-version/decision identities, Python major/minor, the documented CPU backend with two host devices, and current diagnostics. Python patch and host platform remain provenance rather than equality gates, so equivalent Python 3.12 CI hosts replay portably. Exact artifact bytes include timing observations and are independently pinned by the ordinary static evidence test.

## Identifiable synthetic model

The smoke uses 20 literal public-synthetic observations under a known-variance normal model:

```text
μ ~ Normal(0, 2)
yᵢ | μ ~ Normal(μ, 0.4)
```

The conjugate posterior is available exactly:

| Quantity | Exact | PyMC | NumPyro |
|---|---:|---:|---:|
| posterior mean | `1.516467066` | `1.514` | `1.512` |
| posterior sd | `0.08935341032` | `0.091` | `0.094` |
| R-hat | — | `1.00` | `1.00` |
| bulk ESS | — | `231` | `237` |
| divergences | — | `0` | `0` |
| posterior-predictive mean | `1.516467066` | `1.512912276` | `1.512718439` |
| posterior-predictive sd | `0.4098585511` | `0.4117076243` | `0.4136976898` |

Both routes use 2 sequential chains, 300 warmup iterations, 300 retained draws, target acceptance `0.9`, and explicit sampler/predictive seeds.

### Diagnostic gates

Each engine must provide:

- finite posterior samples;
- finite log density (PyMC `lp`) or finite negative potential energy (NumPyro);
- zero divergences;
- R-hat ≤ `1.05`;
- bulk ESS ≥ `100`;
- posterior mean and standard deviation close to the exact conjugate result;
- finite posterior predictive samples whose mean and standard deviation are close to the exact predictive result;
- posterior and predictive mean/standard-deviation parity across engines.

The configuration contract rejects convergence, sampling-budget, posterior, predictive, and cross-engine controls weaker than these evidence-backed defaults; callers may only tighten the adoption gates.

This is deliberately a bounded smoke, not proof that all future hierarchical/PDE calibration models mix well.

## JAX-native boundary

NumPyro owns the JAX-native NUTS route. `numpyro_smoke.py` is separate from `pymc_smoke.py`; an AST architecture test forbids NumPy, SciPy, scikit-fem, and FEM-layer imports in the JAX-native adapter. The profile facade uses dependency-light provenance contracts and is subprocess-tested against any NumPy/SciPy/scikit-fem import traversal.

Automatic FEM differentiation remains fail-closed:

```text
status = unsupported
reason = scikit-fem/SciPy assembly and sparse solves are not a pure JAX trace
```

`require_jax_fem_differentiation()` raises `NotImplementedError`. Promotion requires a pure-JAX or custom implicit-differentiation boundary, finite-difference gradient parity, Taylor remainder convergence, and price/Greek regression evidence.

## CI and absence behavior

- `bayesian` installed-wheel profile: Python 3.12, installs its PyMC-only hash lock before the no-dependencies wheel, imports PyMC/ArviZ, and executes the native posterior smoke.
- `bayesian-jax` installed-wheel profile: Python 3.12, installs the combined hash lock before the no-dependencies wheel, imports PyMC/ArviZ/JAX/NumPyro, executes NumPyro smoke, and replays the semantic profile gate.
- A separate Python 3.12 supply-chain job installs the combined lock and then the no-dependencies `0.2.0` release wheel before audit/SBOM generation, so the CycloneDX document includes the project and its declared relationships; the Python 3.11 supply-chain job audits every published non-Bayesian extra, including calibration, IO, visualization, UI, and JAX.
- External tests live outside default `tests/` collection and raise on missing dependencies; selected profiles never pass through skips.
- Base-wheel subprocess tests block pandas, statsmodels, PyMC, ArviZ, JAX, and NumPyro imports while importing the profile facade.
- The source-distribution contract includes both locks, profile docs/evidence, replay script, and external/static tests.
- Python 3.11 FEM package/test workflows remain separate and unchanged.

Sources: [PyMC documentation](https://www.pymc.io/projects/docs/en/stable/), [ArviZ](https://python.arviz.org/), [JAX version policy](https://docs.jax.dev/en/latest/deprecation.html), and [NumPyro getting started](https://num.pyro.ai/en/stable/getting_started.html).
