# Changelog

## Unreleased

- Reworked `src/estimation` calibration outputs to return structured
  `CalibrationResult` diagnostics. Synthetic fixture calibration now lives behind
  `SyntheticSurfaceCalibrator`; `HestonCalibrator` fails closed until a real
  Heston pricing engine is available, and `StatsmodelsCalibrator` is a
  deprecation shim that delegates to supported SciPy least squares without
  private Statsmodels imports or monkeypatches.
- Introduced an `AdaptiveMesh` utility enabling residual- or gradient-based
  adaptive refinement. Demo and solver now support configurable criteria and
  tests assert element counts increase or decrease accordingly.
- Added `src/problems` package with `OptionPricingProblem` and an explicit
  reduced-form `CreditRiskProblem` for constant-intensity defaultable zero-coupon
  claims.  The credit route now exposes analytical bond/loss/survival/recovery
  outputs and fails closed if routed through a spatial FEM solver.
- Integrated `pytest-benchmark` for performance tracking and configured CI to
  run tests with coverage reporting.
- Added mesh creation tests covering domain extent accuracy and invalid
  dimension handling.
- Introduced an experimental `FenicsSolver` using FEniCSx/UFL with optional
  dependency and benchmarking scaffolding against scikit-fem.
- Added a basic usage example script and documentation for running it.
