# Estimation

Utilities for calibrating model parameters to market data.

## Expected input format

Calibrators expect three NumPy arrays of identical shape:

- `strikes`: option strike prices
- `maturities`: option maturities in years
- `prices`: observed option prices

These arrays define the option surface used during optimisation.  Example:

The module includes:

- ``HestonCalibrator`` using SciPy's least-squares optimizer
- ``StatsmodelsCalibrator`` relying on statsmodels' ``NonlinearLS``
- ``PyMCCalibrator`` performing Bayesian inference with PyMC

```python
calib = StatsmodelsCalibrator(strikes, maturities, prices)
params = calib.calibrate(initial_guess)
```
