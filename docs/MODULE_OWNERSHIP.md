# Module Ownership and Compatibility Inventory

**Owner issue:** [finite_element_options #50](https://github.com/googa27/finite_element_options/issues/50)  
**Status:** executable architecture inventory  
**Machine contract:** [`docs/architecture_contract.toml`](architecture_contract.toml), `[[module_ownership]]`  
**Architecture gate:** `scripts/check_architecture_contract.py` and `tests/architecture/test_architecture_contracts.py`

This module ownership inventory treats the repository as a finite-element numerical backend library. The table below maps every current package-root module or package under `src/finite_element_options` to exactly one ownership class. Adding, moving, or deleting a package-root entry must update this document and the architecture contract, `docs/architecture_contract.toml`, in the same PR.

## Ownership classes

| Class | Meaning | Dependency rule |
|---|---|---|
| `core` | FEM mechanisms and native contracts owned by this repo | No UI, plotting, pandas, PyMC, Statsmodels, JAX, Numba, FEniCSx/PETSc eager imports, Haircut, PDP, or production FD dependency |
| `validation` | Fixtures, benchmark evidence, public-synthetic parity artifacts | May depend inward on public core; cannot define product ownership |
| `compatibility` | Time-bounded legacy boundary retained for transition | Must name an extra, removal target, warning/migration path, and must not be imported by stable public API |
| `optional` | Research, IO, calibration, plotting, or optional backend profile | Lazy imports with actionable missing-extra errors |
| `example` | Runnable examples and demos | Imports installed public package; not imported by core |
| `app` | User interface/application shell | Optional only; never imported by core |
| `cli` | Thin command entry point | Delegates inward; does not define numerical semantics |

## Current package-root inventory

| Path | Class | Target owner / treatment | Public surface | Extra / dependency profile | Removal or stabilization target |
|---|---|---|---|---|---|
| `__init__.py` | core | Lightweight package marker; no broad re-exports | Base import only | core | Keep narrow |
| `contracts/` | core | Immutable native FEM/portfolio compatibility contracts, formula bundles and capability records | Stable public contracts | core | Stabilize under #48/#49 |
| `core/` | core | Market/model coefficient helpers and analytical references used by FEM tests | Public but FEM-owned | core | Consolidate with typed contracts under #48 |
| `space/` | core | Meshes, finite-element spaces, weak forms, BCs, solvers and adaptivity | Public FEM mechanics | core | Stabilize under #48 |
| `time_integration/` | core | Theta stepping and time-policy mechanics | Public FEM mechanics | core | Stabilize under #48 |
| `transform.py` | core | Black-Scholes coordinate/time transforms | Public utility | core | Keep only convention-explicit transforms |
| `problems/` | validation | Thin backend-neutral problem fixtures and examples; not product ownership | Transitional public fixtures | core | Move problem identity into typed contracts under #48 |
| `validation/` | validation | Analytical/manufactured/public-synthetic evidence and parity artifacts | Public validation helpers | validation | Feed generated capability matrix under #61 |
| `fdsolver.py` | compatibility | Legacy finite-difference reference route retained only for benchmark/parity transition; production FD ownership belongs to `finite_difference_options` | Not re-exported by base package; emits `DeprecationWarning` on `FDSolver`, `solve_system`, and Greek helper use | fd | Removal version `0.3.0`, removal date `2026-10-31`; remove or replace with a benchmark oracle after FD parity and adapter gates (#49/#50) |
| `data_utils.py` | optional | Experiment snapshot IO and dataframe/xarray utilities | Optional utility | io | Keep outside core import graph |
| `estimation/` | optional | SciPy/PyMC calibration research adapters; synthetic unless evidence says otherwise | Optional research/calibration | calibration | Production Heston Bayesian route remains #54 |
| `jax_greeks.py` | optional | Experimental AD Greek route | Optional research helper | jax | Keep separate from core sensitivity claims |
| `plots.py` | optional | Plotting helpers | Optional visualization | viz | Keep outside core import graph |
| `sidebar.py` | app | Streamlit sidebar/application support | UI helper only | ui | Keep outside core import graph |
| `cli.py` | cli | Thin CLI shell over validated public APIs | Console entry edge | core plus selected extras | Do not encode solver semantics here |
| `examples/` | example | Single installed-package example tree; root `examples/` Python files are forbidden | Example-only | selected extras | Execute against installed package under #61 |

## Issue #50 closure policy

This inventory satisfies the #50 ownership requirement by making duplicate ownership explicit and executable:

- every package-root entry is classified as core, validation, compatibility, optional, example, app, or cli;
- the finite-difference implementation is not a second production owner here: `fdsolver.py` is compatibility-only, behind the `fd` extra, emits a targeted `DeprecationWarning`, and has a removal target tied to FD parity and adapter gates;
- migration example: production finite-difference solves belong in `finite_difference_options`; `finite_element_options.fdsolver` may be used only as a benchmark/parity oracle until removal version `0.3.0` on `2026-10-31`;
- root-level executable examples are consolidated into `finite_element_options.examples` so examples run from the installed package tree;
- UI, plotting, calibration, JAX and IO stacks are optional boundary modules and are forbidden from the FEM core import graph;
- the base package import remains lightweight and does not re-export compatibility, UI, plotting, calibration or research modules;
- architecture tests fail if a new package-root entry appears without a reviewed ownership mapping.

Numerical behavior is intentionally unchanged by this inventory. It is a topology/ownership gate; convergence and production-readiness claims still require the repository-local validation fixtures and benchmark evidence.
