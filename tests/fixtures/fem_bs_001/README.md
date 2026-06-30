# FEM PIELM parity fixture (public)

This directory contains deterministic public fixture files for issue #74.

- `problem_spec.json`: Public problem/spec contract for a European call Black--Scholes
  FEM oracle fixture (`fem-bs-001`).
- `result_export.json`: Executable result export produced by
  `run_public_black_scholes_parity_fixture()` with full mesh/time-grid rows and
  summary diagnostics.

The contract and export are intentionally consumable as plain files and include:

- Weak-form metadata and sign convention (`existing_forward_tau_identity_transform_black_scholes_forms`)
- Typed boundary metadata for `S=0` and `S=S_max`
- Mesh family/element family + mesh/time-step controls
- Deterministic delta/gamma reference policy
- Equal-error comparison policy (`mode: equal_error`)
- Deterministic config hash for fixture provenance