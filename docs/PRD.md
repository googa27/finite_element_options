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

The repository is one numerical backend in a federated portfolio. It may implement the Haircut Engine solver contract through a thin plugin, but it must not depend on Haircut domain entities or PDP internals.

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

Matrix and vector dimensions, symmetry, definiteness, sparsity and boundary modifications must be testable. Variable coefficients and mixed terms cannot be approximated by hidden constants.

### FR-FEM-005 — Boundary conditions

Dirichlet, Neumann and Robin conditions must have typed representations and mathematically correct weak/algebraic treatment. Essential-boundary elimination must modify both matrix and right-hand side consistently. Unsupported boundary classes fail before solve.

Boundary residuals and far-field/asymptotic assumptions are part of validation evidence.

### FR-FEM-006 — Time integration

The library must provide explicit time-orientation semantics and validated theta-family integration, including Crank–Nicolson and backward Euler. Rannacher or adaptive time stepping may be added only with documented start-up, stability and error behavior.

State transfer after remeshing must be explicit and included in diagnostics.

### FR-FEM-007 — Linear solver and factorization policy

Direct, iterative, preconditioned, AMG or PETSc policies must be selected by typed capability and problem metadata—not broad exception fallback. The result records convergence status, iterations, residuals, factorization/preconditioner reuse, regularization and failure reason.

Invariant matrices and factorizations should be reused where mathematically valid and invalidated deterministically when coefficients, mesh, time step or boundaries change.

### FR-FEM-008 — Adaptivity and error estimation

Adaptive refinement requires an explicit estimator, marking policy, refinement limits, transfer operator and goal quantity. Experimental demos cannot be exposed as production capability without effectivity, convergence and stability evidence.

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

### FR-FEM-012 — Module ownership and duplicate retirement

The stable package owns FEM numerical mechanics. The embedded FD implementation, full application/product workflows, duplicate examples and heavy UI/calibration responsibilities must be classified as core, optional, example, compatibility, migrate or delete.

Production FD behavior migrates to `finite_difference_options`; only time-bounded benchmark or compatibility code may remain. Owner: #50.

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
| `fenicsx` | Optional FEniCSx backend | FEniCSx-compatible environment |
| `petsc` | PETSc/petsc4py solver policy | Platform-specific HPC environment |
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
