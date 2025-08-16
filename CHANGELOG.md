# Changelog

## Unreleased

- Added `src/estimation` module with a generic `Calibrator` base class and a
  sample `HestonCalibrator` demonstrating least-squares calibration on synthetic
  data. Documented expected input formats for future data integration.
- Introduced an `AdaptiveMesh` utility enabling residual- or gradient-based
  adaptive refinement. Demo and solver now support configurable criteria and
  tests assert element counts increase or decrease accordingly.
