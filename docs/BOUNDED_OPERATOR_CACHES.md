# Bounded operator and sparse-factor reuse

Issue [#153](https://github.com/googa27/finite_element_options/issues/153) replaces two unbounded dictionaries with explicit least-recently-used entry limits. This changes retention and recomputation policy only. The weak form, assembled values, endpoint times, theta arithmetic, boundary enforcement, matrix hashes and SciPy sparse LU calls are unchanged.

## Public policy

- `SpaceSolver(..., operator_cache_size=2)` retains at most two time-keyed operators. `space.operator_cache_info()` returns an immutable snapshot of capacity, current/peak entries, hits, misses, evictions and explicit clears, counted over that space's lifetime.
- `ThetaScheme(..., factorization_cache_size=2)` retains at most two exact-key sparse factors **within each solve**. Its last linear-solve diagnostics add capacity, peak entries, entries at completion and eviction count. Completion counts describe the last instant before return; factors are released when the solve returns. An unsuccessful solve still raises the original failure; `last_solve_diagnostics` continues to describe the last completed solve.
- Capacities are nonnegative Python `int` values. Booleans, floats, strings and negative values are refused before assembly. Zero retains no entries and recomputes through the same numerical route. `reuse_factorization=False` retains its existing `fem.solve` path and has effective cache capacity zero. The existing `factorization_reuse_enabled` field records that configured route flag; actual reuse is always given by hit counts.
- Hits move an entry to most recent position. Inserting a new completed operator into a full cache releases the least recently used entry. Failed assembly/factorization never installs a partial entry. Evicted keys are recomputed if requested again.
- Numerical keys remain exact. A repeated scalar step width does not authorize reuse when matrix values, mesh, theta or boundary identity changed. A larger working set can require a larger explicit capacity. Earlier factor-once-per-key behavior now means once **while the key remains resident**.

The two-entry default covers adjacent start/end spatial operators and the demonstrated constant, alternating-two-width and Rannacher-startup factor working sets. It does not infer that a coefficient is time invariant: constant coefficients may still be assembled at distinct endpoint times. No TTL, approximate time matching, cross-solve factor sharing, persistent storage or concurrent-use guarantee is introduced.

## Invalidation and mutation

`space.invalidate_operator_cache()` clears resident time operators and refreshes `space.stiffness` at time zero. Call it after changing coefficient forms or model inputs, or construct a new space. Coefficient closures and their external state must be deterministic for their inputs during a solve; in-place mutations are not detected automatically. Do not mutate returned sparse matrices while they remain cached.

The refinement API rebuilds the mesh bases and mass matrix before invoking the same invalidation method. Changing mass-form semantics, element type or mesh outside that API requires constructing a new space. The method is not a transaction over arbitrary user callbacks: assembly failures propagate; repair the coefficient input and explicitly invalidate again before reuse. The separately retained mass and initial stiffness are outside the cache count.

## Maintained implementation boundary

| Capability | Selected implementation | Alternative and decision |
|---|---|---|
| Recency ordering and eviction | Python [`collections.OrderedDict`](https://docs.python.org/3/library/collections.html#ordereddict-objects), standard-library PSF license | `functools.lru_cache` hides numerical ownership/invalidation and `cachetools` adds an unnecessary dependency for these two serial owners. A small `core.operator_cache` adapter owns explicit policy and immutable counts. |
| Sparse numerical factors and solves | Existing SciPy [`splu`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html) / [`SuperLU`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.SuperLU.html), existing BSD-3-Clause dependency | No custom LU, factor serialization or solver change. Only bound references to completed factors are retained. |

`__getitem__` has normal exact-key lookup/`KeyError` semantics and `__setitem__` retains a caller-owned completed numerical payload. Explicit `clear()` and `info()` express policy and diagnostics. The adapter supplies no arithmetic, hash approximation or generic persistence framework. Space owns endpoint identity and invalidation; theta integration owns enforced-system identity and factor lifetime.

## Numerical and resource evidence

The standalone benchmark loads the two original solver modules byte-for-byte from Git revision `c875010ac1e995fbaa00f575868cd81988fe22b7`, saves those source snapshots/hashes and compares them with the imported current implementation:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/benchmark_bounded_caches.py --output /tmp/fem-cache-evidence
```

Its manufactured problem is the forward-time reaction equation `du/dt = -c(t)u` on `[0,1]`, with `u(0)=1`, zero source/natural boundary terms and 1,025 P1 degrees of freedom. Mass and reaction forms are `(u,v)` and `-(c(t)u,v)`. Continuous solutions are `exp(-0.2t-0.1t²)` for `c(t)=0.2+0.2t`, or `exp(-0.2t)` for constant `c=0.2`. An independent discrete theta product checks every case's final field. No financial calibration or model change is involved.

The measured default policy produced bitwise-identical complete histories in six cases: varying coefficients at 50, 200 and 400 steps; a constant 200-step system; repeated widths `0.25,0.5,0.25`; and a 20-output-step Rannacher schedule with 22 internal steps. For the 400-step varying case:

| Retained payload | Original | Bounded default |
|---|---:|---:|
| Time operators after return | 401 | 2 |
| Resident operator CSR arrays | 16,432,980 bytes | 81,960 bytes |
| Peak cached sparse factors | 400 | 2 |
| Cached factor L/U/permutation representation | 26,243,200 bytes | 131,216 bytes |
| Operator assemblies / factorizations | 401 / 400 | 401 / 400 |

Both histories occupy 3,288,200 bytes. The constant workload retains one factor and records 199 hits; the two-width workload retains two factors and records one hit; Rannacher retains two factors and records 20 hits. The benchmark records environment and source hashes rather than a timing threshold.

CSR bytes are actual retained sparse array sizes. Factor bytes are the sizes of SciPy's exported L/U/permutation representations, **not RSS or native allocator usage**. Active assembly/factorization can temporarily coexist with resident entries. Mass, initial stiffness, output history and scalar time/hash diagnostics remain outside the bounded caches. No whole-solver constant-memory or speedup claim follows.

Regression tests additionally observe actual matrix/LU-owner lifetimes, zero/one/two capacities, eviction/rebuild, post-failure restart, changed boundary matrices, repeated solves, coefficient invalidation and mesh refinement:

```bash
pytest -q tests/unit/test_operator_cache.py tests/integration/test_bounded_solver_caches.py --no-cov
```

Rollback is removal of the cache-policy change and its additive API/diagnostics. Increasing the explicit capacity is a workload-specific tradeoff; it does not change numerical identity or make mutable coefficients safe.
