# Finite Element Options

Finite Element Options demonstrates pricing of European options under various
stochastic models using a finite element discretisation.  The application is
built around [scikit-fem](https://github.com/kinnala/scikit-fem) and provides a
Streamlit based user interface for interactive exploration.

## Features

- Black-Scholes option pricer with call and put payoffs
- Finite-difference solver on regular grids using ``findiff``
- Greek estimators (Delta, Gamma, Vega) via finite differences
- Configurable mesh generation supporting 1D, 2D and 3D problems
- Sample problems for 1D Black–Scholes and 3D Heston models
- Simple market abstraction with discount factors
- Streamlit UI and plotting helpers
- Example demo for adaptive mesh refinement
- Comprehensive test suite using `pytest`
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
