# Changelog

## Unreleased

- Added `src/estimation` module with a generic `Calibrator` base class and a
  sample `HestonCalibrator` demonstrating least-squares calibration on synthetic
  data. Documented expected input formats for future data integration.
- Introduced an `AdaptiveMesh` utility enabling residual- or gradient-based
  adaptive refinement. Demo and solver now support configurable criteria and
  tests assert element counts increase or decrease accordingly.
- Added `src/problems` package with `OptionPricingProblem` and
  `CreditRiskProblem` classes bundling dynamics, payoff and boundary defaults,
  plus a Streamlit demo showcasing their instantiation.
- Integrated `pytest-benchmark` for performance tracking and configured CI to
  run tests with coverage reporting.
- Added mesh creation tests covering domain extent accuracy and invalid
  dimension handling.
- Introduced an experimental `FenicsSolver` using FEniCSx/UFL with optional
  dependency and benchmarking scaffolding against scikit-fem.
- Added a basic usage example script and documentation for running it.
