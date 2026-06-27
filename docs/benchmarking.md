# Benchmarking

This project uses [`pytest-benchmark`](https://pytest-benchmark.readthedocs.io/) to
track solver runtime on fixed meshes.  Benchmarks are executed both locally and
on the continuous integration server.

## Running Benchmarks Locally

Run the benchmark tests in isolation to focus on performance measurements:

```bash
pytest tests/test_benchmark_black_scholes.py --benchmark-only
```

To store results for later comparison, write them to a JSON file:

```bash
pytest tests/test_benchmark_black_scholes.py --benchmark-json=benchmark.json
```

## Interpreting Results

The JSON output records statistics such as mean, median and minimum runtime for
each benchmarked function.  Lower values indicate faster solver performance.
`pytest-benchmark` can compare multiple runs to track regressions:

```bash
pytest-benchmark compare benchmark.json path/to/previous.json
```

CI runs publish the latest `benchmark.json` file as a workflow artifact, making
it easy to download and contrast with local measurements.

## Payoff microkernel policy

The finite-difference terminal-payoff initialization path intentionally uses
NumPy vectorized `maximum` calls rather than an optional Numba microkernel.  The
old acceleration module only wrapped elementwise intrinsic call/put payoffs, so
it added import/startup and cold-compile risk without targeting the measured
solver bottleneck: assembly and time stepping.  Dependency-heavy acceleration
work should instead be justified by end-to-end solver benchmarks such as the
Black-Scholes benchmark above and by the factorization/assembly work tracked in
the performance roadmap.

Verification for this decision is intentionally simple and reproducible:

```bash
.venv/bin/pytest -q --override-ini addopts= tests/test_fd_black_scholes.py \
  tests/test_benchmark_black_scholes.py
```

The payoff tests cover call/put parity against the NumPy formulas, including
non-contiguous grid views.  Because Numba is not imported by the payoff path and
is no longer a runtime requirement, absent or incompatible Numba installations
cannot silently change solver behavior.
