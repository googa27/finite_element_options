# Finite Element Options

Finite Element Options demonstrates pricing of European options under various
stochastic models using a finite element discretisation. The application is
built around [scikit-fem](https://github.com/kinnala/scikit-fem) and provides a
Streamlit-based user interface for interactive exploration.

## Quickstart

Install the dependencies and run a simple Black–Scholes pricing example:

```bash
pip install -r requirements.txt
python - <<'PY'
from src.examples.bs_1d import price_call
grid = price_call()
print(grid[-1])  # option values at maturity
PY
```

The snippet solves a one-dimensional Black–Scholes problem using default
parameters and prints the final price grid.

Mesh creation utilities now return both the mesh and a ``Config`` instance
carrying numerical parameters such as the finite element. Pass this
configuration to ``SpaceSolver`` when constructing spatial discretisations.

Alternatively, run the bundled example script:

```bash
python examples/basic_usage.py
```

This reproduces the same call option pricing workflow in a standalone file.

## Features

- Black-Scholes option pricer with call and put payoffs
- Finite-difference solver on regular grids using ``findiff``
- Numba prototypes for accelerating payoff evaluation
- Greek estimators (Delta, Gamma, Vega) via finite differences
- Experimental JAX-based Greek computation via automatic differentiation
- Configurable mesh generation supporting 1D, 2D and 3D problems
- Central ``Config`` dataclass for numerical parameters
- Sample problems for 1D Black–Scholes and 3D Heston models
- Simple market abstraction with discount factors
- Streamlit UI and plotting helpers
- Example demo for adaptive mesh refinement
- Comprehensive test suite using `pytest`
- Parameter calibration helpers with SciPy, Statsmodels and Bayesian PyMC approaches
- Market data handled via pandas DataFrames and solution snapshots via xarray
- CSV/Parquet serialisation utilities for reproducible experiments
- Experimental FEniCSx solver for Black-Scholes PDE

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Launch the Streamlit application:

```bash
streamlit run main.py
```

Run the adaptive mesh demo:

```bash
python demo_adaptive.py
```

### Example Output

The adaptive mesh demo refines the domain around sharp features. The final
solution after several refinement steps is shown below:

![Adaptive mesh solution showing refined grid](docs/images/adaptive_solution.svg)

## Testing

Execute the unit tests with `pytest`:

```bash
pytest
```

Benchmarks leveraging `pytest-benchmark` can be executed alongside the test
suite. Coverage reports are generated via `pytest-cov`:

```bash
pytest --cov=src
```

## Benchmarking

Solver performance is tracked with `pytest-benchmark`.  Run the dedicated
benchmark suite locally to measure runtime on a fixed mesh:

```bash
pytest tests/test_benchmark_black_scholes.py --benchmark-only
```

Results can be saved for comparison and analysed with `pytest-benchmark compare`.
See [docs/benchmarking.md](docs/benchmarking.md) for details on
interpreting the output and contrasting runs.

### FEniCSx Spike

An experimental `FenicsSolver` mirrors the existing scikit-fem backend using
FEniCSx and UFL. On the Black–Scholes test problem the scikit-fem solver runs
in roughly 2 ms per step on this environment. FEniCSx wheels are unavailable for
Python 3.11, so its performance could not be measured here. The solver is
therefore an optional dependency and currently best treated as a preview for a
potential future migration.

## Continuous Integration

This project uses a GitHub Actions workflow to run the test suite on every
push and pull request, ensuring that the codebase remains reliable.

## Project Structure

```
src/            Core library modules
main.py         Streamlit entry point
requirements.txt  Project dependencies
tests/          Pytest-based test suite
```

## Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md) for development guidelines and prompt templates.

## License

This project is provided as-is without warranty.
