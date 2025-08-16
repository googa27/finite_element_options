# Estimation

Utilities for calibrating model parameters to market data.

## Expected input format

Calibrators expect three NumPy arrays of identical shape:

- `strikes`: option strike prices
- `maturities`: option maturities in years
- `prices`: observed option prices

These arrays define the option surface used during optimisation.  Example:

```python
calib = HestonCalibrator(strikes, maturities, prices)
params = calib.calibrate(initial_guess)
```
