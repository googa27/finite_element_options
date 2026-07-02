# AGENTS.md — Finite Element Options Operating Contract

**Canonical for:** human contributors and coding/review agents  
**Audit baseline:** 2026-06-26  
**Default branch:** `master`  
**Portfolio epic:** [`haircut-engine` #62](https://github.com/googa27/haircut-engine/issues/62)  
**Local modernization epic:** [#43](https://github.com/googa27/finite_element_options/issues/43)

> The prior guide is preserved at
> [`docs/archive/AGENTS.pre-federated-audit-2026-06-26.md`](docs/archive/AGENTS.pre-federated-audit-2026-06-26.md).

## 1. Mission

Build an installable, reusable and numerically trustworthy finite-element library. Weak-form correctness, boundary algebra, convergence, diagnostics, reproducibility and dependency boundaries take priority over feature count or speed.

A module or demo is not a validated capability. A faster solve with worse or unknown error is not an improvement.

## 2. Required reading

Before a non-trivial change, read:

1. `docs/PRD.md`.
2. `docs/ARCHITECTURE.md`.
3. The owning GitHub issue and its parent.
4. Relevant tests and implementation.
5. User-facing README and benchmark documentation when behavior changes.

GitHub issues, canonical docs, tests and release manifests are authoritative. Local agent scratch files, caches and untracked notebooks are not.

## 3. Ownership

This repository owns meshes, finite-element spaces, weak forms, assembly, boundary treatment, time integration, adaptivity, sparse solver policies, sensitivities and FEM diagnostics.

It does not own Haircut Engine domain or CASCADE policy, PDP internals, or a second production finite-difference implementation.

Cross-repository integration uses released wheels, a versioned solver contract, entry points and parity fixtures. Do not add production Git submodules, local-path release dependencies, branch dependencies or repository-relative imports.

## 4. Workflow

1. Start from current `master`.
2. Link the change to the correct parent/child issue.
3. Add a characterization or failing numerical test before changing semantics.
4. Separate package moves from mathematical changes.
5. Update contracts, tests, benchmarks, docs and deprecation notes together.
6. Run targeted numerical tests, then architecture, packaging and profile gates.
7. Describe mathematical risk, compatibility, performance and rollback in the PR.
8. Do not increase a capability's maturity without the required evidence.

## 5. Dependency direction

```text
contracts and diagnostic schemas
             ↑
mesh / spaces / forms / boundary conditions
             ↑
assembly / time integration / linear solvers / adaptivity / sensitivities
             ↑
validation and integration adapters
             ↑
examples / apps / research workflows
```

Rules:

- The stable import package is `finite_element_options`; no new public `src.*` imports.
- Contracts do not import scikit-fem internals, JAX, FEniCSx, PETSc, pandas, PyMC, Streamlit, plotting or Haircut.
- Numerical core does not import products, calibration, UI, examples or integration adapters.
- Optional imports are lazy and name the required extra.
- Examples and applications import the installed public package.
- Compatibility shims live only at package boundaries and have removal milestones.
- Add architecture tests for every new package boundary.

## 6. Mathematical preflight

For every PDE or FEM change, document:

- strong-form PDE and operator sign;
- forward/backward-time transform;
- domain and coordinates;
- initial or terminal condition;
- boundary partition, type and formula;
- weak form and integration-by-parts terms;
- coefficient regularity and quadrature policy;
- element family/order and expected spatial convergence;
- time integrator and expected temporal convergence;
- linear or nonlinear solver and stopping criterion;
- requested value, functional or sensitivity convention.

Product classes and UI defaults are not substitutes for this specification.

## 7. Mesh, forms and boundary rules

- Validate mesh dimension, topology, element orientation and boundary markers.
- Record mesh identity, refinement lineage and degrees of freedom.
- Element family, order, continuity and quadrature are explicit.
- Keep mass, diffusion, advection, reaction, source and coupling terms distinguishable.
- Evaluate variable coefficients using a documented quadrature or projection policy.
- Preserve sparse structure unless a measured small-system exception is documented.
- Essential-boundary elimination must update the right-hand side correctly.
- Neumann and Robin contributions must be retained in the weak form where required.
- Test boundary residuals, corners, time-dependent data and unsupported conditions.
- Generic FEM code must not guess financial far-field boundaries.

## 8. Time integration and linear solvers

- New canonical code uses `time_integration`, not a `time` package.
- State the theta-scheme operator sign and formula used by implementation.
- Test spatial and temporal convergence separately.
- Time-dependent operators and boundaries have explicit assembly and reuse behavior.
- Rannacher smoothing, adaptive time stepping and remeshing are separate capabilities.
- A solver policy declares matrix assumptions, method, preconditioner, tolerances, iteration limit, reuse and diagnostics.
- Broad exception-based solver switching is prohibited.
- Return or raise a typed failure with residual and context when convergence fails.
- Record iterations, residual, factorization/preconditioner reuse and regularization.

## 9. Adaptivity and sensitivities

Adaptive work identifies estimator or goal functional, norm, marking policy, mesh limits, transfer operator and stopping criterion. Evidence includes DOF growth, estimator behavior, transfer error and convergence. A visual demo alone is not validation.

Every Greek or sensitivity states the differentiated state or parameter, units, evaluation point, method, bump or smoothing policy and reference/error evidence. Finite-difference, tangent, adjoint and JAX routes are separate capabilities. JAX remains optional and does not define core array types.

## 10. Backend plugin

The Haircut adapter must:

- expose lightweight identity, solver-contract range, maturity and capabilities;
- validate the request before mesh or assembly work;
- map generic records into canonical native FEM contracts without reinterpretation;
- use only validated native policies;
- return solution, sensitivities and complete diagnostics;
- work from a clean installed wheel;
- import no Haircut domain/application, PDP, UI or calibration modules.

Advertise only capabilities backed by repository-local tests and shared parity evidence.

## 11. Ownership cleanup

Classify modules as stable core, optional, example, research, compatibility, migrate or delete.

- Production FD behavior belongs in `finite_difference_options`.
- Product and market models are thin validation clients unless genuinely FEM-specific.
- Calibration, dataframes, plotting and Streamlit stay outside core.
- JAX, Numba, FEniCSx and PETSc are explicit optional profiles.
- Consolidate duplicate examples so they use the installed package.
- Preserve behavior with tests before moving code; change numerics separately.

## 12. Packaging and dependencies

Packaging changes must:

- use PEP 621 metadata and an explicit build backend;
- place the package under `src/finite_element_options`;
- declare `requires-python`, bounded core dependencies, license and project URLs;
- use published extras for optional capabilities and dependency groups for development tools;
- keep core to NumPy, SciPy and minimal scikit-fem unless evidence justifies more;
- avoid `scikit-fem[all]` in core;
- isolate mesh, visualization, JAX, Numba, calibration, Bayesian, columnar, FEniCSx, PETSc, UI and validation profiles;
- maintain a reproducible lock but publish compatible runtime ranges;
- test minimum-supported and latest-compatible profiles separately;
- build and install sdist/wheel outside the repository;
- test missing-extra messages and wheel contents;
- produce release supply-chain evidence or an owned exception.

Evaluate a dependency by API stability, maintenance, platform support, transitive size, license, security history and measured value.

## 13. Performance

Optimization order:

1. Prove analytical or manufactured correctness and convergence.
2. Profile assembly, boundary application, factorization, solve, transfer and post-processing.
3. Remove dense conversions, repeated assembly and unnecessary solution history.
4. Reuse invariant matrices and factorizations with complete invalidation keys.
5. Compare alternatives at equal numerical error.
6. Establish noise-aware regression budgets.

A benchmark records problem/config hash, DOFs, nonzeros, time steps, versions, hardware/BLAS/threads, dtype/device, cold/warm/JIT/cache state, stage timings, memory, residual/iterations, achieved error and reuse count.

## 14. Tests and verification

Required layers are architecture, contract, unit, numerical, integration, shared parity, performance and packaging.

Current repository commands include:

```bash
python -m pip install -e '.[dev]' -c constraints.txt
python -m build --sdist --wheel
python -m twine check dist/*
python scripts/check_architecture_contract.py
python scripts/check_ci_contract.py
pytest -q tests/architecture tests/test_packaging_contract.py --no-cov
pytest -q
pytest tests/test_black_scholes_1d.py tests/test_fd_black_scholes.py tests/test_fenics_solver.py -q
pytest tests/test_benchmark_black_scholes.py --benchmark-json=benchmark.json
ruff check src tests scripts
mypy --ignore-missing-imports --follow-imports=silent src/finite_element_options/contracts src/finite_element_options/validation scripts/check_ci_contract.py
python -m pip_audit --progress-spinner=off --skip-editable
cyclonedx-py environment --of JSON -o sbom.json
pydocstyle src/finite_element_options
```

CI must keep third-party Actions pinned to full commit SHAs, run clean-wheel package checks on Python 3.11/3.12, run separate optional-profile imports for advertised extras, upload package/test/benchmark/SBOM artifacts, and keep least-privilege permissions plus explicit job timeouts.

Do not report an unconfigured or unrun gate as passing. Record the gap and owner issue.

## 15. Evidence by change type

| Change | Minimum evidence |
|---|---|
| Strong/weak form or coefficients | Mathematical statement, reference/manufactured and residual tests |
| Boundary algebra | Boundary residual, RHS correction, corner and time-dependent cases |
| Mesh or adaptivity | Geometry, transfer and convergence/effectivity evidence |
| Time integration | Temporal convergence and start-up/stability behavior |
| Linear solver | Residual, failure behavior and equal-accuracy benchmark |
| Sensitivities | Convention, method, bump/smoothing and reference error |
| Package topology | Characterization, clean wheel and legacy import mapping |
| Optional dependency | Missing-extra and isolated profile test |
| Haircut adapter | Entry-point discovery, capability negatives, parity and compatibility update |
| Breaking public semantics | Version bump, migration/deprecation note and downstream evidence |

## 16. Compatibility, docs and review

Distribution API and Haircut solver-contract versions are separate. Unknown combinations are unsupported. Public deprecations name replacement, warning version, removal date/version and migration example; shims contain no new numerical behavior.

Update `docs/PRD.md`, `docs/ARCHITECTURE.md`, `AGENTS.md`, tests, benchmarks and issues together when responsibilities or semantics change. Keep README examples aligned with the installed namespace. Do not create competing roadmaps or agent state stores.

Before review, confirm:

- [ ] The change belongs in the FEM repository.
- [ ] Strong/weak form, time and boundary conventions are explicit.
- [ ] Numerical evidence includes convergence, residuals and negative behavior.
- [ ] No optional dependency leaked into core.
- [ ] No new checkout-only import assumption was added.
- [ ] Solver failure and fallback behavior are explicit.
- [ ] Performance compares equal error.
- [ ] API, compatibility and deprecation impact are handled.
- [ ] Examples use the installed package.
- [ ] Docs, issues and release metadata are synchronized.

---

*This file is the canonical Finite Element Options contributor and agent contract.*
