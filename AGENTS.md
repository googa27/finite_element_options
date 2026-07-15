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

Issue #39 domain/boundary rule: use explicit `DomainSpec`/`DomainAxis` or bound pairs whenever lower bounds, transformed coordinates or tail truncation matter. Meshes must carry named facets (`s_min`, `s_max`, `v_min`, `v_max`, ...), Dirichlet BCs must validate those names before enforcement, and boundary oracles must be evaluated at finite-element degrees of freedom rather than through an `L2` projection.

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
- Heston variance domains use the shared CIR diagnostics: exact time-average boundary variance, exact terminal conditional variance mean, exact CIR variance, Feller ratio and an explicit tail-mass policy.

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

- expose Haircut's public `BackendIdentity` and `BackendCapabilityManifest` shapes, solver-contract range, maturity and capabilities;
- register exactly one canonical `haircut.solver_backends` entry point named `finite_element_options`;
- import only Haircut's public solver protocol seam at factory time, without adding a local-path or VCS runtime dependency, and fail closed when that seam is absent or contract majors drift;
- validate the request before mesh or assembly work;
- reject unsupported, private, unsupported-benchmark or mutated public-fixture requests before numerical validation runners or assembly are imported;
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

- Run focused #117 evidence: `pytest -q tests/validation/test_manufactured_solutions.py tests/validation/test_black_scholes_convergence.py`.
- Emit the deterministic FEM evidence bundle: `fem-options validation run-benchmark fem-bs-001 --out /tmp/fem-bs-001-evidence.json`; omit `--out` for JSON stdout.
- Do not report an unconfigured or unrun gate as passing. Record the gap and owner issue.

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

<!-- PORTFOLIO-CONSTITUTION:START -->
## Portfolio engineering constitution

This repository follows [Portfolio Project #24](https://github.com/users/googa27/projects/24) and [finite_element_options rollout issue](https://github.com/googa27/finite_element_options/issues/114). A repository-specific, evidence-backed exception in `docs/ARCHITECTURE.yaml` may specialize a rule; undocumented drift is not an exception.

### Research and maintained-library preference

- Research domain theory, maintained libraries, standards, interfaces, datasets, licenses, adjacent repositories, and probable extension paths before design or implementation.
- **Maintained-library preference:** use well-maintained libraries for solved algorithms, protocols, parsers, persistence, orchestration, dataframes, numerical methods, and security controls instead of implementing them from scratch. Record capability, selected library, alternatives, maintenance/API/license evidence, adapter boundary, and any custom-code justification.
- Custom code belongs to domain semantics, composition, adapters/contracts, or genuinely missing algorithms and must be tested against an oracle/reference.
- Turn reusable findings into maintained Hermes skills and concise support files. Add a plugin or MCP server only when stable CLI/contracts have multiple measured consumers or real interoperable external-tool needs.

### Clean and evolutionary architecture

After the dependency route is sound, apply SOLID, DRY knowledge ownership, suitable design patterns, explicit dependencies, low coupling, cohesive modules, extensibility, maintainability, and technical-debt minimization. Design for probable extensions, not speculative frameworks. Every meaningful change reduces named debt or adds an executable fitness function.

`docs/ARCHITECTURE.yaml` is the machine-readable source of truth. Update it in the same change as architecture, public API, test, CI, data, AI-interface, or exception changes.

- At each Python `src/` level, count immediate runtime `.py` files and package directories, excluding `__init__.py` and architecture/readme/typing metadata. Default maximum: 10. Deepen hierarchy around stable responsibilities instead of widening it.
- Default Python module maximum: 500 physical lines. Larger legacy files are exact no-growth exceptions with reason, owner/context, risk, accepted ceiling, and refactoring trigger.
- Keep `tests/unit`, `tests/integration`, `tests/e2e`, and `tests/architecture`; mirror source where useful. Empty suites document their intended boundary and activation trigger.
- Architecture tests enforce the YAML contract, source fan-out, module-size ratchets, exception metadata, required docs/suites, and repository-specific import/public-API rules.

### Two first-class users

1. **Hermes Agent and compatible agents:** this concise root file, deterministic CLI/public contracts, exact verification commands, and capability discovery are the baseline. Skills encode recurring procedures. Plugins/MCP are optional escalation layers, never substitutes for a stable public interface; mutation tools must be explicit, typed, least-privileged, and separately verifiable.
2. **Human programmer/notebook user:** provide a typed, documented importable API independent of CLI/UI internals and deterministic public-synthetic notebook examples where the repository is a library. Use only lawful Python protocols: compact `__repr__`, value equality/hash for deeply immutable objects, true collection/context/NumPy protocols, and pure IPython display hooks. Prefer named methods for policy, configuration, I/O, diagnostics, expensive/stateful behavior, or ambiguous mathematics. Test every claimed algebraic law and named-method/operator parity.

### AI-assisted change controls

- Treat agent output as untrusted until a human reviews it and executable repository gates verify it. The human author remains accountable.
- Keep agent changes small, single-purpose, and completely reviewable. Generated tests are not a sufficient sole oracle for generated implementation.
- New dependencies require human approval plus package-existence, maintenance, API, license, vulnerability, and typosquat checks; lock reproducibly.
- Security-sensitive code (authentication, cryptography, parsers, serialization, SQL, filesystem, subprocess, network, permissions, or private data) requires dedicated human review.
- Use least privilege: workspace-scoped writes, network/secret access only when approved, no autonomous merge/deploy, and exact command/result provenance.
- Measure AI impact with lead time, review time, CI failures, reverts, escaped defects, and churn; do not infer productivity from self-report.

### Semantic source-tree hierarchy

- Do **not** balance source folders like AVL/B-trees. Package boundaries follow information hiding, cohesion, coupling, public contracts, ownership, and change patterns; naturally heavy-tailed sizes are expected.
- Empty marker packages and speculative folder scaffolds are forbidden unless an exact, dated structural-role exception exists. Keep future plans in architecture/roadmap documents.
- `__init__.py` is a compatibility/public facade only: imports, re-exports, `__all__`, metadata, and bounded lazy hooks. Domain classes and business functions belong in cohesive modules.
- Severe branch concentration is a review trigger, not a command to redistribute files. Fix it only when dependency, churn, ownership, or comprehension evidence shows a bad boundary.


### GitHub Actions supply-chain controls

- Pin every third-party action to a full-length commit SHA; keep the human-readable release in a comment.
- Declare least-privilege workflow `permissions`; read-only `contents` is the default.
- Set `persist-credentials: false` on checkout and provide narrowly scoped credentials only to the step that needs mutation.
- Validate workflow changes with `pinact run --check` and `uvx zizmor --offline --min-severity medium .github/workflows`.

### Data and core-repository boundaries

For data-consuming work, design `source registry -> typed acquisition -> immutable Bronze -> canonical Silver -> curated Gold/features -> formulation/model -> governed output -> read-only UI/API/notebook` before implementation. Record grain, units, classification, lineage, quality, freshness/vintage/effective time, identity, replay, and validation.

- `PDP` owns reusable/public data acquisition and products.
- `financial_problem_formulations` owns general problem/formulation/formula/workflow semantics.
- `ui_and_artifacts` owns reusable audience-aware rendering and artifact QA.
- Consume stable public contracts/CLIs, not repository internals. Keep canonical names theoretical/general rather than deal/product-specific.

Repository posture: Consume FPF contracts; avoid PDP/UI runtime dependencies; emit solver evidence bundles. Data posture: No data ownership; public-synthetic/convergence fixtures only.

### Exact commands

- Setup: `python -m pip install -e '.[dev]'`
- Tests: `python -m pytest -q`
- Lint/format: `ruff check . && ruff format --check .`
- Portfolio architecture: `python scripts/check_portfolio_architecture.py`
- AI/hierarchy policy: `python3 scripts/check_ai_hierarchy_policy.py`

If a command is declared unavailable, the activation trigger and replacement command belong in `docs/ARCHITECTURE.yaml`; do not fabricate successful output.
<!-- PORTFOLIO-CONSTITUTION:END -->
