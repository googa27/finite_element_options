# Finite Element Options — Product Requirements Document

**Status:** Canonical target-state PRD; current repository is an experimental library/application hybrid  
**Audit baseline:** 2026-06-26  
**Repository:** `googa27/finite_element_options`  
**Default branch:** `master`  
**Portfolio epic:** [`haircut-engine` #62](https://github.com/googa27/haircut-engine/issues/62)  
**Local modernization epic:** [#43](https://github.com/googa27/finite_element_options/issues/43)

---

## 1. Product statement

Finite Element Options is a reusable, library-first finite-element toolkit for parabolic and related pricing PDEs. Its stable responsibility is the FEM numerical mechanism: meshes, approximation spaces, weak forms, assembly, boundary treatment, time integration, adaptivity, sparse linear-solver policies, sensitivities and FEM diagnostics.

Option pricing and credit-risk examples are validation clients of the numerical library, not the definition of its internal architecture. Streamlit, plotting, calibration, JAX experiments, FEniCSx/PETSc integrations and product-specific workflows are optional profiles or examples.

Issue #55 narrows the credit-risk example to a constant-intensity reduced-form defaultable zero-coupon claim with fractional recovery of par paid at default. Because that claim has no spatial state, its supported route is an analytical/ODE reference that separates bond value, default-free value, credit loss, survival/default probabilities, recovery leg and LGD; attempts to route it through spatial FEM assembly fail closed until a stochastic-intensity PDE is explicitly specified.

The repository is one numerical backend in a federated portfolio. It may implement the Haircut Engine solver contract through a thin plugin, but it must not depend on Haircut domain entities or PDP internals.

Issue #34 fixes Heston variance moment semantics before any higher-dimensional domain claims are advertised. The two- and three-dimensional Heston helpers share one CIR kernel for

\[
E[V_{t+\tau}\mid V_t=v]=\theta+(v-\theta)e^{-\kappa\tau},
\]

including zero-horizon, zero-mean-reversion, small-product and long-horizon limits. Boundary pricing now uses the exact CIR time-average mean variance, while solver evidence includes conservative variance-domain diagnostics built from the exact terminal first two CIR moments, the Feller ratio and an explicit Chebyshev tail-mass bound.

Issue #36 preserves state- and time-dependent reaction/source coefficients in the generic weak form. Dynamics models may expose vectorized `discount(state, time)` and `source(state, time)` fields evaluated at scikit-fem quadrature points. The theta stepper passes interval endpoint times into spatial assembly so nonconstant coefficients are not sampled once and frozen; 3D Heston uses the short-rate state coordinate as the reaction field and collapses to the 2D constant-rate discount limit when that coordinate is fixed.

Issue #39 makes state domains and boundary facets explicit before advertising broader Heston or transformed-domain solves. Mesh construction accepts `DomainSpec`/`DomainAxis` records, `(lower, upper)` pairs and legacy extents, attaches canonical named facets, and preserves transformed-coordinate bounds. Dirichlet boundary enforcement validates facet names, rejects duplicate labels, and evaluates oracle values at degrees of freedom so carry, strike, maturity and coordinate-transform semantics are not distorted by projection.

Issue #40 makes theta stepping consume validated, strictly increasing time grids. Nonuniform grids use each local step, roundoff-uniform `linspace` grids canonicalize to one reusable invariant system, source terms use theta-consistent endpoint timing, Dirichlet data are enforced at the new time node, and Rannacher startup is encoded through the same theta schedule diagnostics.

## 2. Portfolio role and boundaries

| Concern | This repository owns | This repository must not own |
|---|---|---|
| FEM core | Meshes, spaces, basis/quadrature, weak forms, assembly and boundary enforcement | PDP ingestion or Haircut business logic |
| Time and linear algebra | Time integration, state transfer, sparse-solver/factorization policy and diagnostics | Hidden solver selection in UI/product code |
| Adaptivity | Error estimators, refinement, marking and transfer | Unvalidated production claims from demos |
| Sensitivities | FEM-consistent values, Greeks and parameter sensitivities with explicit convention | Generic portfolio policy or regulatory reporting |
| Validation | Manufactured solutions, analytical benchmarks and FEM convergence evidence | Production finite-difference ownership |
| Integration | Thin versioned backend adapter and capability manifest | Direct imports from Haircut domain/application modules |
| Delivery | Optional examples, CLI or Streamlit exploration | Mandatory UI/plot/calibration dependencies in core |

Target dependency direction:

```text
Haircut generic solver contract
            │
            ▼
FEM integration adapter
            │
            ▼
public FEM contracts → mesh/space/forms/assembly/time/linear solvers

No dependency from FEM core into Haircut Engine, PDP, UI, calibration or examples.
No direct production dependency on finite_difference_options.
```

## 3. Users and jobs to be done

| User | Job | Required outcome |
|---|---|---|
| Numerical developer | Add or improve FEM methods | Explicit weak form, convergence, residual and boundary evidence |
| Quant/model developer | Solve a pricing PDE | Stable typed problem/config/result API and reproducible diagnostics |
| Model validator | Assess numerical correctness | Manufactured/analytical references, observed order and failure semantics |
| Performance engineer | Improve assembly or solve speed | Accuracy-adjusted benchmarks, memory and reuse evidence |
| Backend consumer | Integrate FEM into another application | Clean wheel, capability manifest and no private-module coupling |
| Researcher | Explore JAX, FEniCSx, PETSc or Bayesian workflows | Isolated optional profile with explicit experimental maturity |

## 4. Maturity language

| Status | Meaning |
|---|---|
| `implemented` | Code exists and focused tests pass |
| `validated` | Reference, convergence, residual/boundary and negative tests pass |
| `production` | API, packaging, compatibility, performance and release gates pass |
| `experimental` | Unstable optional research route |
| `deprecated` | Replacement and removal version/date are published |
| `unsupported` | Request fails before mesh/assembly/solve |

A module, example or benchmark script does not itself establish a validated capability.

## 5. Functional requirements

### FR-FEM-001 — Installable package and public namespace

The project must use modern `pyproject.toml` metadata and a real import package such as `finite_element_options` under `src/`. A built wheel installed outside the repository must expose the same public API as development installs.

The literal package name `src`, checkout-relative imports and `sys.path` compensation are prohibited in the stable API. Owner: #44.

### FR-FEM-002 — Immutable numerical problem contracts

Public native contracts must describe:

- state domain and coordinate system;
- time interval and forward/backward orientation;
- coefficient fields and regularity assumptions;
- initial or terminal condition;
- source/running term;
- boundary specifications;
- requested values, fields, Greeks and estimators;
- discretization, tolerance and resource policy;
- dtype/device/determinism;
- result diagnostics and provenance.

Product-specific models may translate into these records but cannot redefine their mathematical semantics. Owner: #48 and adapter #49.

### FR-FEM-003 — Meshes and approximation spaces

The core must support documented structured or unstructured meshes and finite-element spaces with explicit dimension, element family/order, coordinate transform, quadrature policy and geometric validity checks.

Mesh refinement, coarsening and mappings must preserve deterministic identity sufficient for diagnostics and restart/reproduction.

### FR-FEM-004 — Weak forms and assembly

Every operator must be defined through an explicit strong and weak form, sign convention and coefficient interpretation. Assembly must distinguish mass, diffusion, advection, reaction, source and optional coupling terms.

Coordinate transforms are model terms, not display metadata. If a state transform is used, assembly must apply the full componentwise chain rule: transformed diffusion, Hessian drift correction, transformed diffusion divergence, singular-Jacobian rejection and PSD covariance diagnostics where relevant. Matrix and vector dimensions, symmetry, definiteness, sparsity and boundary modifications must be testable. Variable coefficients and mixed terms cannot be approximated by hidden constants.

### FR-FEM-005 — Boundary conditions

Dirichlet, Neumann and Robin conditions must have typed representations and mathematically correct weak/algebraic treatment. Essential-boundary elimination must modify both matrix and right-hand side consistently. Unsupported boundary classes fail before solve.

Boundary residuals and far-field/asymptotic assumptions are part of validation evidence.

### FR-FEM-006 — Time integration

The library must provide explicit time-orientation semantics and validated theta-family integration, including Crank–Nicolson and backward Euler. Rannacher startup uses `startup_theta`, `startup_steps`, and `startup_substeps` on the same theta-step API. Each solve records the supplied output grid, local and internal step widths, theta schedule, uniform-grid status, and forward orientation.

Time grids must be finite and strictly increasing. Nonuniform grids use every local `dt_n`; roundoff-uniform grids are canonicalized only to prevent `np.linspace` ulps from defeating invariant-system factorization reuse. Boundary/source values are evaluated with scheme-consistent theta endpoint timing, and essential Dirichlet data are enforced at the new time node.

### FR-FEM-007 — Linear solver and factorization policy

Direct, iterative, preconditioned, AMG or PETSc policies must be selected by typed capability and problem metadata—not broad exception fallback. The result records convergence status, iterations, residuals, factorization/preconditioner reuse, regularization and failure reason.

Invariant matrices and factorizations should be reused where mathematically valid and invalidated deterministically when coefficients, mesh, time step or boundaries change.

### FR-FEM-008 — Adaptivity and error estimation

Adaptive refinement requires an explicit estimator, marking policy, refinement limits, transfer operator and goal quantity. Experimental demos cannot be exposed as production capability without effectivity, convergence and stability evidence.

The current validated core supports residual or gradient-driven refinement through `AdaptiveMesh.refine_with_transfer`: it checks positive simplex measures, preserves total domain measure, carries named boundary/domain metadata across scikit-fem remeshing, transfers nodal values by interpolation, emits estimator/marking/transfer diagnostics and disables coarsening until a reversible hierarchy and coverage proof exist.

Goal-oriented estimators must identify the target functional and adjoint assumptions.

### FR-FEM-009 — Values, Greeks and sensitivities

The public result must identify:

- quantity differentiated;
- coordinate and units;
- differentiation method;
- smoothing or nonsmooth-payoff policy;
- interpolation/projection location;
- error estimate or reference evidence.

Finite-difference, adjoint and automatic-differentiation routes remain distinct capabilities. JAX support is optional and cannot make JAX a mandatory core dependency.

Black-Scholes analytical-oracle APIs must name whether an input is volatility `sigma` or variance `sigma**2`. Volatility Vega `dV/dsigma` and variance sensitivity `dV/d(sigma**2)` are separate quantities connected by the chain rule only when variance is strictly positive. Zero-maturity, zero-volatility, zero-spot and invalid-input limiting cases are part of the public numerical contract.

### FR-FEM-010 — Numerical validation

The repository must maintain method-specific tests for:

- heat and advection–diffusion–reaction manufactured solutions;
- one-dimensional Black–Scholes call/put values and boundaries;
- spatial and temporal convergence orders;
- variable coefficients;
- nonuniform/adaptive meshes;
- Greeks away from and around payoff kinks;
- multidimensional mixed terms where advertised;
- invalid mesh, coefficient, boundary and solver requests.

A scalar endpoint comparison alone is insufficient.

### FR-FEM-011 — Haircut Engine backend plugin

After package and typed-interface foundations are stable, publish an optional entry point in `haircut.solver_backends`. The adapter must:

- expose backend identity, solver-contract version, maturity and capabilities;
- translate generic problem records without changing conventions;
- reject unsupported requests before mesh or assembly work;
- return complete FEM diagnostics and evidence metadata;
- use only canonical public FEM APIs;
- import no Haircut domain/application, PDP, UI or calibration modules.

Owner: #49.

Issue #64 provides the first executable Project #5 compatibility slice before the full entry-point plugin: a public-synthetic QuantProblemSpec adapter manifest, fail-closed route diagnostics and the `fem-bs-001` Black-Scholes parity fixture. This slice is deliberately narrow and validates only the one-dimensional uniform-line/Lagrange-P2/theta/SciPy-direct route against the Haircut analytical oracle with named endpoint boundary enforcement and value/Delta/Gamma error evidence; unsupported dimensions, adaptive meshes, unvalidated elements, American exercise, jump terms and HJB/control terms fail before mesh or weak-form allocation.

Issue #74 extends this to a public arxiv-lab compatible FEM oracle artifact set for the same European call benchmark. It adds deterministic public problem/contract files, explicit weak-form sign metadata, typed boundary metadata ($S=0$, $S=S_{\max}$), mesh/time-step and results exports, and an equal-error comparison policy so parity consumers can compare results against matching error budgets using public files/contracts.

Issue #78 adds the Pinares-specific public-synthetic fixed-price proxy weak-form fixture. It consumes/exports a `quant-problem-spec/v0` record with UF units, `Q*` proxy measure, maturity date/time domain, survival-scaled terminal payoff, one-dimensional drift/diffusion/reaction terms, endpoint Dirichlet/linear-growth boundary metadata, Lagrange-P2 line mesh controls, theta stepping and SciPy-direct solve evidence. The route is validated only for the fixed-price option proxy; full family-contract, ROFR, obstacle/free-boundary, jump/liquidity, HJB/control and legal/tax-output requests fail closed with diagnostics before mesh allocation.

Project #17 publishes a released public FEM solver contract for Pinares fixed-price proxy parity and closes the residual solver-cache evidence for #47/#71. The validated solver route remains `scipy_direct`; repeated one-dimensional line-uniform theta solves reuse one sparse LU factorization per invariant system and record factorization reuse counts plus max residual. Banded, AMG and PETSc routes are explicit unsupported capabilities until their own dependency/profile, convergence and equal-error benchmark evidence exists.

Issue #116 consumes FPF `pde_ir.v0` compiled weak-form artifacts through serialized public fixtures. The supported VQPW v0 route is exactly `tests/fixtures/compiled_weak_form/black_scholes_call_v0.json`: public-synthetic Black-Scholes, one-dimensional finite spot domain `[0, 400]`, lower/upper essential endpoint enforcement (natural split empty), Lagrange-P2 line mesh, theta/Crank-Nicolson time stepping with backward orientation, 80 positive steps, tau `[0, 1]` and theta `0.5`, SciPy-direct sparse solves, and value/Delta/Gamma analytical evidence. The adapter rejects private, mutated, unsupported mesh/BC/dimension/exercise/output/schema/hash/sign/unit/measure/numeraire/route/expression fixtures before assembly by exact golden-fixture comparison of `pde_ir`, `compiled_operator` and route subobjects, nested unknown-field checks, and typed `CompiledWeakFormDiagnostic` conversion errors. It exposes deterministic `fem-options qps screen|solve` JSON.

Issue #117 adds `VQPW-FEM-VERIFICATION-EVIDENCE-V0` for the `fem-bs-001` route. The evidence generator must produce deterministic accepted JSON, validate it before writing, recompute immutable section hashes and the full `evidence_hash`, and reject tampered sections/hash fields. Required checks include a SymPy manufactured residual, at least three spatial and temporal refinement levels, negative perturbations for operator sign/source/reaction/boundary, Black-Scholes price/Delta/Gamma tolerance gates, no-arbitrage checks, and CLI access via `fem-options validation run-benchmark fem-bs-001 --out PATH` or stdout JSON when `--out` is omitted.

### FR-FEM-012 — Module ownership and duplicate retirement

The stable package owns FEM numerical mechanics. The embedded FD implementation, full application/product workflows, duplicate examples and heavy UI/calibration responsibilities must be classified as core, optional, example, compatibility, migrate or delete.

Production FD behavior migrates to `finite_difference_options`; only time-bounded benchmark or compatibility code may remain. Owner: #50.

Issue #51 repairs the remaining benchmark-oracle role for `fdsolver.py`: it is now documented as a narrow one-dimensional Black-Scholes reference over validated uniform spot/time-to-maturity grids, not a production FD backend. The route fails closed for invalid/nonuniform grids and time grids, uses carry-aware endpoint Dirichlet boundaries with a nonnegative call far-field clamp, eliminates nonzero Dirichlet columns from interior equations, propagates payoff domain errors instead of broad scalar fallback, reuses a constant factorization, and returns labeled xarray coordinates plus route/time/residual/convergence metadata.

### FR-FEM-013 — Optional research and application profiles

JAX, Numba, PyMC, Statsmodels, pandas/xarray/Arrow, plotting, Streamlit, FEniCSx, PETSc and mesh-I/O capabilities are optional profiles. Each has explicit import boundaries, maturity and CI policy.

Examples and apps import the installed public package and cannot become alternate canonical APIs.

## 6. Non-functional requirements

### 6.1 Numerical correctness

- Strong/weak forms, time transforms and boundaries are documented.
- Observed convergence agrees with the expected asymptotic order or the deviation is explained.
- Linear solver failure is never represented as a valid numerical field.
- Unsupported dimensions, coefficients, BCs and outputs fail closed.
- Floating-point regularization is explicit and quantified.

### 6.2 Reproducibility

- Problem, mesh, discretization and solver configurations are immutable or snapshot-able.
- Deterministic routes repeat within documented tolerance.
- Result metadata includes package/backend versions, mesh/grid identity, dtype/device and solver policy.
- JIT, cache and factorization reuse are disclosed.

### 6.3 Performance and scalability

- Sparse assembly and solve are the default for nontrivial systems.
- Avoid repeated assembly, dense conversion and unnecessary solution-history retention.
- Benchmark assembly, boundary application, factorization, solve, adaptivity/transfer and post-processing separately.
- Compare performance at equal error, not equal element count alone.
- Track memory and factorization reuse as well as wall time.

### 6.4 Maintainability and modularity

- One import package and one public API inventory.
- Core has no UI, calibration, dataframe, JAX, FEniCSx or Haircut dependency.
- Dependencies point inward and are architecture-tested.
- Compatibility shims contain no new numerical logic and have removal dates.
- Examples, notebooks and generated assets are not sources of architectural truth.

### 6.5 Reliability and security

- No unbounded mesh refinement or solver iteration without limits.
- Optional native/HPC dependencies have documented platform and license constraints.
- Release profiles generate vulnerability, SBOM and license evidence or owned exceptions.
- Test and benchmark artifacts contain no private downstream data.

## 7. Distribution and dependency profiles

The current single `requirements.txt` combines core, FD, UI, calibration, dataframe, JAX and test packages. The target uses PEP 621 metadata, published extras and development dependency groups.

| Profile | Purpose | Representative boundary |
|---|---|---|
| `core` | Native contracts and validated FEM baseline | Bounded NumPy, SciPy and minimal `scikit-fem` |
| `mesh` | Optional mesh readers/generators | Mesh-specific packages only |
| `viz` | Plotting and export | Matplotlib/visualization packages |
| `jax` | Experimental differentiable FEM/Greeks | JAX profile |
| `calibration` | Deterministic statistical calibration | pandas/xarray/Statsmodels as required |
| `bayes` | Bayesian research workflow | PyMC and its stack |
| `columnar` | Arrow/Parquet experiment exchange | PyArrow and optional dataframe packages |
| `fenicsx` | Optional FEniCSx backend | FEniCSx-compatible environment; contract tests run in CI even without runtime |
| `petsc` | PETSc/petsc4py solver policy | Platform-specific HPC environment; KSP convergence diagnostics fail closed |
| `ui` | Streamlit demonstration | Streamlit plus visualization |
| `validation` | Reference and benchmark tooling | Test/reference packages |
| development groups | Test, lint, typing, docs, build and audit | pytest, Hypothesis, Ruff, mypy, build, twine |

Do not depend on `scikit-fem[all]` in the minimal profile. The development lock reproduces CI; wheel metadata uses compatible runtime ranges. Minimum-supported and latest-compatible profiles are tested separately.

## 8. Release, compatibility and deprecation

- Distribution and solver-contract versions are independent.
- Wheels and sdists must install and test outside the repository root.
- Haircut compatibility is recorded in the matrix owned by Haircut #65.
- Unknown contract/version combinations are unsupported.
- Public deprecations include replacement, warning version, removal version/date and migration example.
- No production Git submodule, branch dependency or local-path dependency.
- Breaking numerical convention or public API changes require semantic versioning, migration notes and downstream parity evidence.

## 9. Production gates

| Dimension | Required gate |
|---|---|
| Packaging | Real package namespace, PEP 621 metadata, wheel content/import smoke |
| Architecture | Import/dependency contracts and no core heavyweight imports |
| FEM correctness | Manufactured/analytical, convergence, boundary and negative evidence |
| Diagnostics | Residual, convergence, solver, mesh, timing and provenance metadata |
| Plugin | Clean-wheel discovery, capability rejection and shared parity |
| Performance | Accuracy-adjusted stage and memory benchmarks |
| Optional profiles | Isolated extras and explicit maturity/platform CI |
| Release | Compatibility matrix, changelog, SBOM/vulnerability/license evidence |

## 10. Roadmap and issue ownership

| Workstream | Owner issues |
|---|---|
| Numerical verification and model correctness | #33–#42 |
| Repository modernization and package foundation | #43, #44 |
| Typed interfaces, calibration, Greeks and solver policy | #45–#48 |
| Haircut backend adapter | #49 |
| Ownership and duplicate retirement | #50 |
| Portfolio architecture and release governance | `haircut-engine` #62–#65 |

New work must attach to an existing owner or create a scoped child with dependencies, acceptance criteria and migration impact.

## 11. Non-goals

- A full collateral recovery or regulatory-risk platform.
- A second production finite-difference library.
- A mandatory environment containing all UI, Bayesian, JAX, FEniCSx and PETSc stacks.
- Product-specific business entities in the FEM core.
- Git submodules or source-tree imports as integration contracts.
- Production claims based on examples, demos or speed alone.

## 12. Change policy

Any change affecting strong/weak-form convention, mesh/space semantics, boundaries, time integration, solver policy, adaptivity, sensitivities, public APIs, dependency profiles, backend capabilities or compatibility must update this PRD, `docs/ARCHITECTURE.md`, `AGENTS.md`, relevant fixtures/benchmarks and issue/release metadata in the same change set.

---

## References

- PyPA `pyproject.toml`: https://packaging.python.org/en/latest/guides/writing-pyproject-toml/
- PyPA `src` layout: https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/
- PyPA plugin discovery: https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/
- Python entry points: https://docs.python.org/3/library/importlib.metadata.html
- uv dependency profiles: https://docs.astral.sh/uv/concepts/projects/dependencies/
- scikit-fem documentation: https://scikit-fem.readthedocs.io/
- SciPy sparse linear algebra: https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html

*End of canonical Finite Element Options PRD.*
