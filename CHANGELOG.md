# Changelog

## Unreleased

- Added `src/estimation` module with a generic `Calibrator` base class and a
  sample `HestonCalibrator` demonstrating least-squares calibration on synthetic
  data. Documented expected input formats for future data integration.
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
