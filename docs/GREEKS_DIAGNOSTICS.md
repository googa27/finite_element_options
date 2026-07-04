# Greeks Diagnostics and JAX Benchmark Policy

**Owner issue:** [finite_element_options #46](https://github.com/googa27/finite_element_options/issues/46)

This repository exposes three distinct Greek paths. They must not be collapsed into a single "JAX Greeks" claim.

## Methods

| Method | Differentiated object | Status |
|---|---|---|
| `analytical_oracle` | Black-Scholes European call price from volatility `sigma` | Canonical scalar oracle used for regular NumPy results |
| `analytical_oracle_limit` | Same analytical object, evaluated through explicit limit semantics | Used for expiry, zero-volatility, zero/near-zero spot, saturated tails, or any input where JAX AD would hit singular expressions |
| `jax_ad_analytical_formula` | JAX implementation of the analytical Black-Scholes formula | Experimental AD of the formula only; it is not AD through FEM assembly or a linear solve |
| `coordinate_aware_np_gradient` | Price grid values over supplied mesh coordinates | Numerical recovery for nonuniform grids; coordinates are passed explicitly to `np.gradient` |

## Required metadata

`compute_greeks_report` returns a `GreekComputationReport`. Each `GreekObservation` records:

- Greek name (`delta`, `vega`);
- method;
- differentiated mathematical object;
- input variable (`spot`, `volatility sigma`);
- units;
- finite/undefined status;
- bump-and-revalue error when a regular central bump is valid.

The report also records requested backend, used backend, dtype, device, JAX 64-bit configuration, and any fallback reason.

## JAX timing policy

`benchmark_greeks` follows JAX asynchronous-dispatch guidance:

1. Convert host inputs to JAX arrays and synchronize (`transfer_seconds`).
2. JIT the AD function and synchronize the first call (`compile_seconds`).
3. Run the compiled function again and synchronize (`warmed_seconds`).

The `memory_bytes` metric preserves the historical benchmark tuple contract as host Python peak memory from `tracemalloc`. It is not an accelerator memory claim. Accelerator memory must remain unreported until a device-specific allocator API is added and validated.

For `backend="auto"`, the historical selection rule is preserved: JAX is selected only when its synchronized total runtime is no more than `1.5x` the NumPy analytical oracle runtime. Otherwise the report records a NumPy fallback reason.

## Boundary policy

Expiry, zero volatility, zero spot and saturated tails use `analytical_oracle_limit`. They must either return finite oracle values or raise the same explicit validation error as the canonical Black-Scholes oracle. The module does not claim AD through FEM until the actual assembly and linear solve path is differentiable and validated.
