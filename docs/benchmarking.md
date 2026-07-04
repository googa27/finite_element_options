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

## Solver-cache/factorization acceptance evidence

Project #17 issues #47/#71 add executable residual evidence for invariant
one-dimensional line-uniform FEM solves.  The released route uses SciPy sparse
LU (`splu`) once per theta-system and reuses that factorization for every time
step right-hand side.  Each benchmark row records DOFs, nonzeros, assembly count,
factorization count, factorization reuse count, solve count, max linear residual,
cache keys and stage timings.  The direct SciPy route is validated; banded, AMG
and PETSc routes remain fail-closed in the capability manifest until their own
equal-error residual benchmarks are added.

Run the focused evidence test with:

```bash
pytest -q --override-ini addopts= tests/test_solver_cache_benchmark.py
```

Programmatic access:

```python
from finite_element_options.validation.solver_cache_benchmark import run_solver_cache_benchmark

report = run_solver_cache_benchmark()
assert report.accepted
```

## Public Pinares fixed-price proxy fixture

Project #12 issue #78 adds a public-synthetic FEM weak-form fixture for the
Pinares fixed-price purchase-option proxy. It is intentionally a numerical
compatibility benchmark, not a family-contract/legal/tax valuation. Refresh the
published JSON artifacts with:

```bash
python scripts/export_pinares_fixed_price_proxy_fixture.py
```

The script writes:

- `tests/fixtures/fem_pinares_fixed_price_proxy_v1/problem_spec.json`
- `tests/fixtures/fem_pinares_fixed_price_proxy_v1/result_export.json`
- `tests/fixtures/fem_pinares_fixed_price_proxy_v1/unsupported_full_deal_problem_spec.json`
- `tests/fixtures/quant_problem_specs/pinares_fixed_price_proxy.json`

Verification is the deterministic analytical-oracle and fail-closed test slice:

```bash
pytest -q --override-ini addopts= tests/test_pinares_fem_proxy.py tests/test_fem_backend_capabilities.py
```

The supported route is only one-dimensional, public-synthetic, `Q*` proxy,
UF-denominated, Lagrange-P2 on a uniform line mesh, theta/Crank-Nicolson time
stepping, SciPy direct solve, endpoint Dirichlet/linear-growth boundaries, and
value/Delta/Gamma outputs. Full Pinares deal routes (ROFR, legal coordination,
liquidity jumps/default, HJB/control, obstacle/free boundary or legal/tax output)
must produce diagnostics before mesh allocation.
