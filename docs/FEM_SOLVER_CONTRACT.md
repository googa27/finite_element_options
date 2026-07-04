# FEM Solver Contract

This repository exposes a conservative public FEM backend contract for downstream parity consumers. The contract is domain-neutral and uses only public-synthetic fixtures; it does not import Pinares, Haircut Engine, PDP, UI, calibration, or private application modules.

## Released contract object

Python consumers can inspect:

```python
from finite_element_options.contracts import DEFAULT_RELEASED_FEM_SOLVER_CONTRACT

payload = DEFAULT_RELEASED_FEM_SOLVER_CONTRACT.to_public_dict()
```

The contract id is `finite-element-options-fem-solver-contract-v0.1` and the backend id is `finite_element_options.fem_backend.v0`.

## Advertised route

The manifest intentionally advertises only the validated route:

- dimension: 1
- mesh family: `line_uniform`
- element family: `lagrange_p2`
- PDE terms: drift, diffusion, reaction
- boundary class: endpoint Dirichlet, including public linear-growth far-field metadata
- exercise style: European
- outputs: value, Delta, Gamma
- time control: theta/Crank-Nicolson with validated increasing grids, nonuniform local steps, new-time Dirichlet enforcement and optional Rannacher startup schedule diagnostics
- linear solver: `scipy_direct` with per invariant theta-system sparse LU factorization reuse

Unsupported solver routes fail closed in the manifest:

- `scipy_banded`: unsupported until banded extraction, residual, and equal-error evidence exists for boundary-eliminated FEM matrices.
- `amg`: unsupported until optional dependency, convergence, tolerance, and equal-error evidence exists.
- `petsc`: unsupported until PETSc profile/platform and parity evidence exists.

## Pinares fixed-price proxy parity

Pinares parity is limited to the public-synthetic fixed-price purchase-option proxy fixtures:

- `tests/fixtures/quant_problem_specs/pinares_fixed_price_proxy.json`
- `tests/fixtures/fem_pinares_fixed_price_proxy_v1/problem_spec.json`
- `tests/fixtures/fem_pinares_fixed_price_proxy_v1/result_export.json`
- `tests/fixtures/fem_pinares_fixed_price_proxy_v1/unsupported_full_deal_problem_spec.json`

The supported benchmark ids are `PINARES-FEM-FIXED-PRICE-PROXY-V0` and `PINARES-QPS-FIXED-PRICE-PROXY-V0`. The fail-closed negative benchmark id is `PINARES-FEM-FAIL-CLOSED-V0`.

Full family contract, ROFR, obstacle/free-boundary, jump/liquidity, HJB/control, legal/tax output, and private-data requests remain unsupported and must be rejected before mesh allocation.

## Verification

Focused contract checks:

```bash
pytest -q --override-ini addopts= tests/test_fem_backend_capabilities.py tests/test_pinares_fem_proxy.py tests/test_solver_cache_benchmark.py
```
