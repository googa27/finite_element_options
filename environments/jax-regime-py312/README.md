# JAX regime Python 3.12 environment

This environment isolates the experimental DYNAMAX/NumPyro/Diffrax regime study from the base FEM wheel and the older `bayesian-jax` profile. The two extras intentionally conflict because they pin different validated JAX minor lines.

## Recreate

```bash
uv venv --python 3.12 /tmp/feo-jax-regime
uv pip install --python /tmp/feo-jax-regime/bin/python \
  --require-hashes -r environments/jax-regime-py312/ci-requirements.lock
uv pip install --python /tmp/feo-jax-regime/bin/python \
  --require-hashes -r environments/jax-regime-py312/requirements.lock
uv pip install --python /tmp/feo-jax-regime/bin/python \
  --require-hashes -r environments/jax-regime-py312/test-requirements.lock
/tmp/feo-jax-regime/bin/python -m build --wheel --no-isolation --outdir dist
uv pip install --python /tmp/feo-jax-regime/bin/python --no-deps \
  dist/finite_element_options-*.whl
```

Verify from the repository root:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
XLA_FLAGS=--xla_force_host_platform_device_count=2 \
/tmp/feo-jax-regime/bin/python -m pytest -q \
  external_tests/jax_regime/ --no-cov

JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
XLA_FLAGS=--xla_force_host_platform_device_count=2 \
/tmp/feo-jax-regime/bin/python scripts/run_jax_regime_study.py \
  --synthetic --warmup 75 --samples 75 --chains 2 --pricing-paths 32 --verify
```

## Frozen stack and limitations

- JAX/jaxlib 0.11.1; NumPyro 0.21.0; DYNAMAX 1.0.2; Diffrax 0.7.2.
- CI build/audit/SBOM tooling is independently frozen: build 1.6.0, CycloneDX 7.3.1, packaging 26.2, pip 26.2.1, pip-audit 2.10.1, setuptools 84.1.0, wheel 0.48.0, and the CI HTTP tooling security floor urllib3 2.8.0 (issue 172). The 2026-10-31 security repair also pins certifi 2026.10.19, filelock 4.2.1 and msgpack 1.2.3; scientific/runtime/test/visual pins are unchanged.
- Test tooling is independently frozen at packaging 26.2, pytest 9.0.3, and pytest-cov 7.0.0 with every transitive dependency hash-pinned.
- `fastprogress` is fixed at version `1.0.3` as an explicit lean compatibility pin. Later 1.1.x releases add an unrelated web stack that this research profile does not need.
- DYNAMAX 1.0.2 leaves TensorFlow Probability unconstrained. This lock freezes `tfp-nightly` at version `0.26.0.dev20260907`; a JAX/DYNAMAX/TFP upgrade requires the full external profile replay.
- DYNAMAX 1.0.2 cannot initialize a one-state HMM because its TFP Dirichlet path rejects event size one. The study therefore uses an exact closed-form Gaussian baseline for $K=1$.
- TFP emits deprecation warnings under JAX 0.11.1. They are a reassessment trigger, not suppressed support evidence.
- Diffrax is forced onto every daily regime boundary and checked pathwise against the exact log-diffusion update. It is an abstraction/extension seam, not an accuracy improvement.

The runtime lock SHA-256 is `42f83eb5da5716b7f228bdb94338beb5b552d9fe0fdb866449e5cb31b8c46a7c`; the test-tool lock SHA-256 is `ab7d270889b7d1b74e7723668d972173b86e2e5d763d6385ad6566d5ac418af0`; and the CI-tool lock SHA-256 is `faa3ac74a6d187a494962c2ff5dc922befa51bf2c59bad6a22b16eb8fd9e1f99`. Release acceptance requires replaying the external profile and `pip-audit` commands in the current pull request; this document does not substitute for their live output.

The historical 2026-09-07 CI-tool lock checksum was `5dbd4f3f15dce41e455b4cde0cb453c23782379cc4b37fef0db526ec75e0580b`. The 2026-10-07 security repair (issues 172/174) produced `7779e0c2ea2b64c957093d76e1ae24401088e801feaa770cb675e72727a5607c`. The current checksum adds only the four audited transitive/tool patches above. The numerical study JSON, sidecar, figures and scientific runtime/test/visual lock identities are unchanged. Current isolated profile/audit and bounded synthetic checks do not imply a new canonical research run.

Primary [certifi release metadata](https://pypi.org/pypi/certifi/2026.10.19/json)
retains MPL-2.0 and Python >=3.7; [msgpack 1.2.3 metadata](https://pypi.org/pypi/msgpack/1.2.3/json)
retains Apache-2.0 and Python >=3.10. Filelock 4.2.1 and setuptools 84.1.0
retain MIT. These are CI/supporting dependencies; DYNAMAX, NumPyro, JAX and
Diffrax versions and all declared scientific budgets remain fixed.

The universal/UI and historical visual locks are outside this repair's audited
cohorts. Their separate migration and actual visual replay are owned by
[issue 184](https://github.com/googa27/finite_element_options/issues/184);
historical lock and figure bytes remain evidence of their original execution.

## Scalar likelihood ownership

Current NumPyro and chronological scalar likelihoods use the already locked TFP JAX `HiddenMarkovModel.log_prob` through a lazy adapter; no dependency or lock changes accompany this boundary. The original scalar recursion lives only in `external_tests/jax_regime/test_hmm_likelihood.py`. Run that suite alongside the existing profile tests. It preserves the existing1e-300 input-probability floor and checks the strict-interior tail/gradient case that rejects direct DYNAMAX1.0.2 likelihood reuse. Explicit one-state parameters work for this likelihood, independently of the separate DYNAMAX prior-fit initializer limitation. The pricing filter and historical numerical evidence remain unchanged.


The scalar likelihood seam uses public JAX inexact/common dtype promotion for integer and mixed real emission arrays. Covariance promotion follows the original linalg route; homogeneous float32/float64 return precision is retained, and mixed arithmetic is compared within its input-precision budget. The canonical float64 NumPyro model, priors, and probability floor are unchanged. The external likelihood suite checks all54 admitted real-storage combinations against the independent original derivation.
