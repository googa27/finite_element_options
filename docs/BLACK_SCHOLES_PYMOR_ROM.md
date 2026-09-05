# pyMOR Black–Scholes Reduced-Order Adoption Pilot

## Decision

**Promote pyMOR only as an optional experimental adapter for this validated affine class.**

The canonical public-synthetic run passed every promotion gate:

| Gate | Required | Observed | Result |
|---|---:|---:|:---:|
| Exact affine reconstruction | relative error ≤ `1e-11` | `2.045146659e-16` | PASS |
| Maximum ROM/FOM price error | ≤ `1e-7` | `2.637945418e-11` | PASS |
| Maximum ROM/FOM Delta error | ≤ `1e-6` | `1.005116472e-7` | PASS |
| Maximum ROM/FOM Gamma error | ≤ `1e-5` | `5.587854562e-6` | PASS |
| Maximum FOM / ROM linear residual | ≤ `1e-9` | `1.580398139e-14` / `1.421085472e-14` | PASS |
| Median online speedup | ≥ `10x` | `28.6038821x` | PASS |
| Queries to amortize offline cost at `10x` | ≤ `1000` | `367` | PASS |
| Out-of-envelope behavior | refuse and name FOM fallback | both sides refused | PASS |
| Predecessor continuity | issue #134 artifact hash verified | exact SHA-256 match | PASS |

This does **not** promote reduced-order modeling as a general solver capability. The capability matrix remains unchanged. Full-order FEM remains authoritative and is the mandatory fallback outside the declared volatility envelope.

## Reproducible evidence

- Artifact: [`evidence/black_scholes_pymor_rom_2026-09-05.json`](evidence/black_scholes_pymor_rom_2026-09-05.json)
- Artifact SHA-256: `f30d712e054937ac7e17ea452fc2bcbc0a874087b1ec180caa5e63dc190ea4b7`
- Study-input SHA-256: `d56805683c07bd8ef5bd7a54b39c3faca3bcd48fd01366ef0de3f7e7a97a0044`
- Affine-decomposition SHA-256: `196f8dd754d88268a081555605ff77bdfd3448f4da2feeb48a102d68ec8e2070`
- Predecessor artifact SHA-256: `d488ea1d2300b3cd1da882479a5a475b22732145335ca3e4a3abd4393e80463f`
- Environment: CPython `3.11.15`, pyMOR `2026.1.0`, NumPy `2.4.6`, SciPy `1.17.1`, scikit-fem `12.0.2`
- Privacy class: `public_synthetic`

```bash
uv run --extra reduction python scripts/run_pymor_rom_benchmark.py
uv run --extra reduction python scripts/run_pymor_rom_benchmark.py --verify
```

`--verify` deliberately does not compare timing bytes. It checks stable schema/input/decomposition/library identities and reruns all current scientific, fallback, speed, and amortization gates because wall-clock observations are noisy.

## Why the decomposition is valid

For backward time `τ`, the Black–Scholes weak-form operator depends on volatility only through variance `η = σ²`:

```text
M du/dτ = K(η)u,
K(η) = K_constant + η K_variance.
```

The benchmark uses one fixed line/P2 mesh, one fixed time grid, fixed rate/maturity/strike, and a volatility-independent asymptotic European-call boundary:

```text
V(0, τ) = 0,
V(S_max, τ) = S_max - K exp(-rτ).
```

That boundary choice matters: it prevents hidden non-affine `σ` dependence in the eliminated Dirichlet terms. Direct scikit-fem assembly and affine reconstruction were compared at every training and holdout volatility plus the domain midpoint; the maximum relative matrix error was `2.045146659e-16`.

The time integrator is implicit Euler (`θ=1`) over 160 steps. This damps high-frequency error from the nonsmooth call payoff and gives stable finite-bump Greeks. The full-order and reduced-order routes use the same time grid, boundary policy, and price/Greek output functionals. Eliminated Dirichlet values are added back to all output functionals, including stencils near the right boundary; the analytical Gamma oracle uses the full `S σ √T` denominator.

## Offline/online construction

| Item | Canonical choice |
|---|---|
| Volatility envelope | `[0.10, 0.35]` |
| Training values | 9 equally spaced values, endpoints included |
| Holdout values | 6 disjoint interior values |
| Full FEM dimension | 4,097 DOFs; 4,095 interior DOFs |
| Snapshot schedule | every 4th state from 160 time steps |
| Snapshot count | 369 |
| POD product | FEM mass matrix |
| Maximum basis size | 40 |
| Effective numerical basis | 36 modes |
| Dimension reduction | `113.81x` |
| Online parameter | `η = σ²` |

pyMOR owns the maintained numerical building blocks:

- `pymor.algorithms.pod.pod` for mass-product POD;
- `pymor.algorithms.projection.project` for Galerkin projection;
- `NumpyVectorSpace`, `NumpyMatrixOperator`, and `to_matrix` at the adapter boundary.

Repository code owns the domain semantics: FEM assembly, affine decomposition, boundary elimination, implicit-Euler stepping, price/Greek output functionals, envelope refusal, benchmark timing, amortization policy, and evidence schema. No pyMOR object crosses the public result contract.

## Holdout accuracy

Greeks use the same centered output policy in both routes: scikit-fem `Basis.probes` supplies the actual containing-element P2 interpolation weights at `S₀-h`, `S₀`, and `S₀+h`, with `h=0.02`, followed by centered Delta and Gamma differences.

| `σ` | Price error | Delta error | Gamma error | FOM/oracle price | FOM/oracle Delta | FOM/oracle Gamma |
|---:|---:|---:|---:|---:|---:|---:|
| 0.1125 | `7.9904e-12` | `1.0051e-7` | `5.5879e-6` | `3.2847e-5` | `1.3067e-3` | `4.0665e-3` |
| 0.1375 | `9.4252e-12` | `5.2203e-8` | `9.2923e-7` | `4.2523e-5` | `8.7523e-4` | `4.2242e-3` |
| 0.1875 | `6.2533e-12` | `5.5670e-8` | `2.8247e-6` | `6.1032e-5` | `5.0756e-4` | `3.9698e-3` |
| 0.2375 | `1.9086e-12` | `1.3731e-8` | `5.6589e-7` | `7.8966e-5` | `3.6116e-4` | `3.5038e-3` |
| 0.3125 | `2.0147e-11` | `3.3870e-8` | `1.2013e-6` | `1.0549e-4` | `2.6627e-4` | `2.8813e-3` |
| 0.3375 | `2.6379e-11` | `1.8394e-8` | `4.6735e-7` | `1.1430e-4` | `2.4891e-4` | `2.7100e-3` |

The analytical Black–Scholes comparison validates that the FOM reference itself remains within the declared discretization tolerances: price `2e-4`, Delta `2e-3`, Gamma `5e-3`.

Each holdout records 160 linear solves, 16,375 FOM interior nonzeros, and final infinity-norm algebraic residuals. Maxima were `1.580398139e-14` (FOM) and `1.421085472e-14` (ROM), both gated against `1e-9`. The repeated timing path builds one parameter-specific FOM factorization and reuses it 13 times after its first solve.

## Timing and amortization

Timing uses `time.perf_counter`, 3 warmups per holdout, and 11 measured repetitions per holdout: 66 FOM and 66 ROM samples. The primary FOM comparator is deliberately strict: each holdout's parameter-specific sparse operator is assembled and factorized once, then the benchmark times cached repeated full-order marches. The six untimed parameter-setup samples had median `0.002951470669 s`. Execution order alternates, and cyclic GC is disabled during each sample and collected immediately afterward.

| Metric | FOM | ROM |
|---|---:|---:|
| Median | `0.1084771429 s` | `0.003792392323 s` |
| MAD | `0.001253380673 s` | `0.0001251709182 s` |
| 5th percentile | `0.1063878465 s` | `0.00362235948 s` |
| 95th percentile | `0.1227074434 s` | `0.004254847532 s` |

Offline total: `2.587373165 s`, including system construction, direct affine validation, 9 training FOM solves, lazy pyMOR import/vector-space setup, snapshot conversion, POD, and projection.

- Raw median online speedup: `28.6038821x`.
- Ordinary break-even: 25 solves.
- First solve count reaching an amortized `10x`: 367 solves.
- Amortized speedup at the declared 1,000-query horizon: `17.00331197x`.

## Memory evidence

| Allocation | Bytes |
|---|---:|
| Full-reference three sparse affine matrices | 639,036 |
| Offline snapshot matrix | 12,088,440 |
| Estimated offline snapshot assembly peak | 24,176,880 |
| Standalone online ROM numerical payload | 1,212,816 |
| Estimated per-solve reduced workspace | 32,832 |

The trained ROM deliberately does not retain the full-order system; the caller keeps FOM separately as fallback. The tiny 1D problem is a speed win, **not a sparse-storage win**: retaining the dense POD basis costs about `1.90x` the three sparse affine matrices. Counts cover numerical arrays, sparse matrix buffers, and estimated dense reduced solve workspace—not Python object headers or process RSS. This pilot therefore makes no memory-reduction claim. Larger/multidimensional systems need a separate empirical memory gate that includes sparse factorization and peak RSS.

## Fail-closed envelope

The trained ROM accepts only `σ ∈ [0.10, 0.35]`. Calls at `0.05` and `0.3675` raised `ROMEnvelopeError` with:

```text
reason   = parameter_out_of_envelope
fallback = full_order_fem
```

There is no extrapolation flag and no silent clipping.

## Transitive security exception

pyMOR `2026.1.0` declares `diskcache` as a mandatory dependency. PyPI currently provides only `diskcache 5.6.3`, affected by `PYSEC-2026-2447` / `CVE-2025-69872`: unsafe pickle deserialization if an attacker can write a cache directory later read by the process. No patched PyPI release exists.

This adapter does not accept cache paths or persisted cache objects. It serializes construction with a process-local `RLock`, temporarily sets `PYMOR_CACHE_DISABLE=1`, calls pyMOR's public `disable_caching()` before constructing any pyMOR object, then restores both the caller's prior environment policy and process-wide cache state in `finally`. The reduction CI jobs execute that path, and the supply-chain audit carries one exact, commented `--ignore-vuln PYSEC-2026-2447`; all other findings remain blocking. Remove the ignore as soon as a patched DiskCache release is available or pyMOR removes the dependency.

## Limitations and next evidence

- Only a one-parameter, one-dimensional, European-call system has been validated.
- The exact affine decomposition relies on fixed mesh/time grids and the volatility-independent asymptotic boundary. Analytical sigma-dependent boundaries, local/stochastic volatility, moving meshes, early exercise, jumps, and regime coupling require new decomposition evidence.
- Timing is host-specific. Semantic replay must pass the speed and amortization thresholds on the target machine; committed timing bytes are not golden outputs.
- POD captures the observed training trajectory manifold; the parameter envelope is not a proof of error between sampled values. The six disjoint holdouts and hard refusal are the current controls.
- Promotion is optional and experimental. A capability-matrix upgrade requires at least one real repeated-solve consumer, broader model coverage, residual/error estimators, persistence/versioning of trained bases, and target-hardware replay.

## Library decision

| Capability | Selected | Alternatives | Reason |
|---|---|---|---|
| POD + Galerkin projection | pyMOR `==2026.1.*` | custom NumPy/SciPy, RBniCSx, neural surrogate | Maintained public POD/operator-projection APIs; avoids hand-written MOR algorithms; clean lazy adapter boundary |
| FOM assembly/time stepping | existing scikit-fem/SciPy stack | pyMOR discretizer rewrite | Reuses the repository’s validated FEM semantics and keeps FOM authoritative |
| Evidence and gates | repository contracts | benchmark prose only | Domain-specific tolerances, fallback policy, lineage, and amortization are repository responsibilities |

Sources: [pyMOR 2026.1 release](https://github.com/pymor/pymor/releases/tag/2026.1.0), [pyMOR 2026.1 docs](https://docs.pymor.org/2026-1-0/), [PyPI](https://pypi.org/project/pymor/), and [license](https://github.com/pymor/pymor/blob/2026.1.0/LICENSE.txt).
