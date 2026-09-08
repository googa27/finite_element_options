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
  external_tests/jax_regime/test_profile.py --no-cov

JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
XLA_FLAGS=--xla_force_host_platform_device_count=2 \
/tmp/feo-jax-regime/bin/python scripts/run_jax_regime_study.py \
  --synthetic --warmup 75 --samples 75 --chains 2 --pricing-paths 32 --verify
```

## Frozen stack and limitations

- JAX/jaxlib 0.11.1; NumPyro 0.21.0; DYNAMAX 1.0.2; Diffrax 0.7.2.
- CI build/audit/SBOM tooling is independently frozen: build 1.6.0, CycloneDX 7.3.1, packaging 26.2, pip 26.2.1, pip-audit 2.10.1, setuptools 84.0.0, and wheel 0.48.0.
- Test tooling is independently frozen at packaging 26.2, pytest 9.0.3, and pytest-cov 7.0.0 with every transitive dependency hash-pinned.
- `fastprogress` is fixed at version `1.0.3` as an explicit lean compatibility pin. Later 1.1.x releases add an unrelated web stack that this research profile does not need.
- DYNAMAX 1.0.2 leaves TensorFlow Probability unconstrained. This lock freezes `tfp-nightly` at version `0.26.0.dev20260907`; a JAX/DYNAMAX/TFP upgrade requires the full external profile replay.
- DYNAMAX 1.0.2 cannot initialize a one-state HMM because its TFP Dirichlet path rejects event size one. The study therefore uses an exact closed-form Gaussian baseline for $K=1$.
- TFP emits deprecation warnings under JAX 0.11.1. They are a reassessment trigger, not suppressed support evidence.
- Diffrax is forced onto every daily regime boundary and checked pathwise against the exact log-diffusion update. It is an abstraction/extension seam, not an accuracy improvement.

The runtime lock SHA-256 is `42f83eb5da5716b7f228bdb94338beb5b552d9fe0fdb866449e5cb31b8c46a7c`; the test-tool lock SHA-256 is `ab7d270889b7d1b74e7723668d972173b86e2e5d763d6385ad6566d5ac418af0`; and the CI-tool lock SHA-256 is `5dbd4f3f15dce41e455b4cde0cb453c23782379cc4b37fef0db526ec75e0580b`. Release acceptance requires replaying the external profile and `pip-audit` commands in the current pull request; this document does not substitute for their live output.
