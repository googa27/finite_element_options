# PETSc American Variational-Inequality Assessment

## Decision

**Promote a single-rank PETSc SNES-VI adapter only as an explicit external profile. Keep the existing scikit-fem assembly and SciPy projected-SOR route canonical.**

The trigger is real: `FEM-AMERICAN-LCP-REFERENCE` already solves a lower-obstacle American-option complementarity problem. PETSc therefore received a real, non-skipped functional and equal-discretization trial rather than an import-only probe.

| Gate | Observed | Required | Result |
|---|---:|---:|:---:|
| Existing American-VI trigger | validated lower-obstacle route | present | PASS |
| Runtime KSP / SNES-VI / TS | all converged | real execution | PASS |
| Package origin | installed wheel from clean `/tmp` venv | wheel, not editable checkout | PASS |
| Grid max error vs projected SOR | `4.059721096e-7` | ≤ `5e-7` | PASS |
| Price error | `7.60093746e-8` | ≤ `2e-7` | PASS |
| Delta error | `5.251533163e-7` | ≤ `2e-6` | PASS |
| Gamma error | `1.577888479e-6` | ≤ `2e-5` | PASS |
| American ≥ European price | premiums `0.00511545303` / `0.005115529039` | nonnegative | PASS |
| Projected-SOR residual | `2.172082961e-10` | ≤ `1e-9` | PASS |
| PETSc SNES-VI residual | `9.242360606e-10` | ≤ `1e-9` | PASS |
| Typed failure path | `SNES_DIVERGED_LINEAR_SOLVE` → `LCPConvergenceError` | required | PASS |
| Execution scope | `COMM_SELF`, size 1 | single rank only | PASS |

This is **not** a capability-matrix upgrade, a distributed-assembly claim, a PETSc TS production time integrator, or a general American-product qualification.

## Evidence

- Artifact: [`evidence/petsc_vi_assessment_2026-09-05.json`](evidence/petsc_vi_assessment_2026-09-05.json)
- Artifact SHA-256: `b0ebd55b748c2c36382854ad6624f3f983b8a1ee25cdb1e34419df7fa9da5b35`
- Predecessor pyMOR artifact SHA-256: `f30d712e054937ac7e17ea452fc2bcbc0a874087b1ec180caa5e63dc190ea4b7`
- Runtime: CPython `3.12.3`, PETSc `3.24.6`, petsc4py `3.24.6`, NumPy `2.4.6`, SciPy `1.18.1`, scikit-fem `12.0.2`
- Privacy: `public_synthetic`

## Explicit external environment

There is intentionally no `finite-element-options[petsc]` extra. PyPI builds PETSc/petsc4py from source and must match an installed PETSc ABI; pretending this is a normal wheel profile would be misleading.

The verified Linux/Homebrew route was:

```bash
REPO="$PWD"
PETSC_DIR="$(brew --prefix petsc)"       # verified PETSc 3.24.6
VENV=/tmp/feo-petsc-wheel
DIST=/tmp/feo-petsc-dist
uv build --wheel --out-dir "$DIST"
uv venv --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" \
  'setuptools>=77' wheel 'cython==3.0.12' 'numpy==2.4.6'
PETSC_DIR="$PETSC_DIR" uv pip install \
  --python "$VENV/bin/python" --no-build-isolation 'petsc4py==3.24.6'
uv pip install --python "$VENV/bin/python" "$DIST"/finite_element_options-*.whl pytest pytest-cov

cd /tmp
PYTHONPATH="" PETSC_DIR="$PETSC_DIR" "$VENV/bin/python" -m pytest -q \
  "$REPO/external_tests/petsc_vi/test_petsc_vi_external.py" --no-cov
PYTHONPATH="" PETSC_DIR="$PETSC_DIR" "$VENV/bin/python" \
  "$REPO/scripts/run_petsc_vi_assessment.py" \
  --output "$REPO/docs/evidence/petsc_vi_assessment_2026-09-05.json" --verify
```

`--verify` is an explicit **semantic gate replay**, not a byte-for-byte timing reproduction: it compares stable input/runtime/decision identities and reruns every current gate. Wall-clock fields are expected to change. The committed artifact's exact SHA-256 is separately enforced by `tests/validation/test_petsc_vi_evidence.py` in the ordinary base suite.

A default isolated build selected Cython `3.3.0` and failed compiling petsc4py `3.24.6` (`PC.pyx: Invalid index type 'int'`). Pinning the verified Cython build dependency and using `--no-build-isolation` against the matched external PETSc fixed the build. The project base wheel still imports without petsc4py.

Other supported PETSc installation routes include conda-forge, distro packages, Spack, and source builds. Every route must prove that `petsc4py.__version__` matches `PETSc.Sys.getVersion()` and rerun the real tests; an import or skipped test is not support evidence.

## Ownership and method

The adapter consumes the existing canonical LCP contract:

```text
x >= obstacle
A x - b >= 0
(x - obstacle) · (A x - b) = 0 componentwise
```

Ownership remains deliberately split:

| Layer | Owner |
|---|---|
| Mesh, P2 basis, mass/operator assembly, Dirichlet enforcement | scikit-fem / repository FEM code |
| Canonical LCP convention and residual diagnostics | `time_integration.lcp` |
| SciPy reference solve | projected SOR |
| Optional external solve | PETSc `SNESVINEWTONRSLS` |
| Nested linear solve | PETSc KSP `preonly` + PC `lu` |
| Time loop | repository `ThetaScheme`; PETSc TS is doctor-only |

The PETSc adapter uses `COMM_SELF` and converts each SciPy CSR time-step matrix into a PETSc AIJ matrix. It does not claim distributed mesh partitioning or distributed assembly.

`ThetaScheme` now accepts the structural `LCPSolver` protocol. Existing callers remain on `ProjectedSORSolver`; PETSc is injected explicitly only by the external profile.

## Real runtime doctor

| Component | Configuration | Evidence |
|---|---|---|
| KSP | `preonly` + `lu`, 2×2 dense system | 1 iteration, zero infinity-norm residual |
| SNES VI | `vinewtonrsls`, coupled 2×2 lower-obstacle LCP | 1 nonlinear + 1 linear iteration, zero projected residual |
| TS | backward Euler, `u'=-u`, `dt=0.01`, `T=0.1` | 10 steps, `TS_CONVERGED_TIME`, absolute error `4.49536657e-4` |

TS execution proves the installed runtime surface, but the American benchmark intentionally retains repository-owned time stepping and uses SNES VI only for each discrete complementarity solve.

## Equal-discretization American put

Both solvers used exactly:

- one scikit-fem P2 mesh with 513 DOFs;
- 80 implicit-Euler time steps;
- rate `0.05`, volatility `0.20`, maturity `1`, strike/spot `1`;
- domain `[0,4]`;
- canonical LCP tolerance `1e-9`;
- identical mesh, assembled operator family, obstacle/boundary policy, and initial payoff.

The two full trajectories evolve independently. Their first-step LCP is identical, but later right-hand sides and warm starts inherit each backend's preceding numerical state and therefore are not byte-identical. Exact same-LCP semantics are separately covered by the coupled 2×2 integration test; the 80-step result is correctly interpreted as full-trajectory equal-discretization parity.

| Output | Projected SOR | PETSc SNES-VI | Absolute error |
|---|---:|---:|---:|
| Price | `0.06073451755` | `0.06073459356` | `7.60093746e-8` |
| Delta | `-0.4116379827` | `-0.4116374575` | `5.251533163e-7` |
| Gamma | `2.308939413` | `2.30894099` | `1.577888479e-6` |

The same discretization's European put price was `0.05561906452`, so both American routes produced positive early-exercise premiums (`0.00511545303` and `0.005115529039`).

Projected SOR used 6,646 iterations across 80 solves (maximum 97 per step). PETSc used 119 nonlinear and 119 nested linear iterations (maximum 21 nonlinear iterations per step). Different final exercise counts—241 versus 235 at the diagnostic tolerance—occur only at near-boundary nodes; the value grid and price/Greek tolerances pass.

## Runtime

An untimed initial parity solve warms each backend. Three full repeated solves then alternate backend order; each repetition uses fresh solver objects and excludes mesh/operator construction from both routes. The artifact retains every sample plus median, MAD, and 5th/95th percentiles:

| Backend | Samples (s) | Median (s) |
|---|---|---:|
| Projected SOR | `15.7567`, `15.7174`, `15.7681` | `15.75673886` |
| PETSc SNES-VI | `0.3300`, `0.3369`, `0.3429` | `0.3369057309` |

PETSc's median solve time was `0.0213817×` the Python projected-SOR reference, or a **46.77× speedup**. This is evidence against the current Python reference on this host—not a comparison with an optimized native active-set implementation or a distributed scaling claim.

## Memory

| Allocation | Bytes |
|---|---:|
| SciPy assembled mass + operator sparse buffers | 53,288 |
| Last-step CSR transfer to PETSc | 26,644 |
| Five PETSc vector payloads (estimate) | 20,520 |
| PETSc `Mat.getInfo()['memory']` | `0` (runtime did not report allocator bytes) |

Memory evidence is therefore bounded buffer accounting, not peak RSS. The zero PETSc allocator report is retained rather than replaced with an invented estimate.

## Failure behavior

A singular incompatible LCP produced:

```text
SNES_DIVERGED_LINEAR_SOLVE
projected_residual_max = 1.0
```

With `fail_on_nonconvergence=False`, the adapter returned a failed `LCPResult`; with the default policy, it raised the existing typed `LCPConvergenceError`. Unsupported SNES/KSP/PC settings fail validation before PETSc allocation.

## Limitations and next trigger

- Single rank only (`COMM_SELF`); no MPI scaling evidence.
- No distributed scikit-fem assembly, partitioned mesh, AMG, GPU, or PETSc TS American-VI route.
- PETSc is not installed by the base package or ordinary CI profile. CI validates the committed public-synthetic artifact; the external command above is the real support gate.
- Only one normalized American put and one fixed discretization are covered.
- SciPy remains canonical because it is portable, base-installed, and fully exercised in regular CI.
- Promote broader PETSc capability only after a maintained external environment can run in CI and either (a) multi-rank equal-error scaling exceeds the SciPy memory/time envelope or (b) a production American-VI consumer requires PETSc SNES/TS behavior.

Sources: [PETSc SNES manual](https://petsc.org/release/manual/snes/), [SNESVINEWTONRSLS](https://petsc.org/release/manualpages/SNES/SNESVINEWTONRSLS/), [petsc4py SNES API](https://petsc.org/release/petsc4py/reference/petsc4py.PETSc.SNES.html), and [PETSc installation guide](https://petsc.org/release/install/).
