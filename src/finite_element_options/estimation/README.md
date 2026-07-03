# Estimation

Utilities for fitting parameterized pricing surfaces to market data.

## Expected input format

Calibration adapters take a `pandas.DataFrame` with numeric columns:

- `strike`: option strike prices;
- `maturity`: option maturities in years;
- `price`: observed option prices.

## Current calibration status

The production Heston route is fail-closed until a real Heston pricing engine,
parameter constraints, and model-risk diagnostics are implemented. The old toy
polynomial has been moved behind explicitly synthetic names.

Available classes:

- `SyntheticSurfaceCalibrator`: deterministic SciPy `least_squares` fit for a
  documented synthetic fixture only;
- `StatsmodelsCalibrator`: deprecated compatibility shim that warns and delegates
  to the supported SciPy adapter; it does not import private Statsmodels APIs;
- `PyMCCalibrator`: Bayesian PyMC fit for the same synthetic fixture only;
- `HestonCalibrator`: compatibility placeholder that raises `NotImplementedError`
  rather than pretending the synthetic fixture is a Heston model.

All calibration methods return `CalibrationResult`, not a bare parameter vector.
Callers should inspect `success`, `status`, `message`, residual norm, Jacobian
rank/conditioning, bounds, and method before using fitted parameters.

```python
import numpy as np
import pandas as pd

from finite_element_options.estimation import SyntheticSurfaceCalibrator

strikes = np.array([90.0, 100.0, 110.0])
maturities = np.array([0.25, 0.5, 1.0])
true_params = np.array([0.04, 1.0, 0.04, 0.3, -0.7])
prices = SyntheticSurfaceCalibrator.price_formula(strikes, maturities, true_params)
calib = SyntheticSurfaceCalibrator(
    pd.DataFrame({"strike": strikes, "maturity": maturities, "price": prices})
)
result = calib.calibrate(true_params + np.array([0.01, -0.1, 0.02, -0.05, 0.1]))
assert result.success
params = result.parameters
```
