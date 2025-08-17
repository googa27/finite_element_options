# Finite Element Options

Finite Element Options demonstrates pricing of European options under various
stochastic models using a finite element discretisation.  The application is
built around [scikit-fem](https://github.com/kinnala/scikit-fem) and provides a
Streamlit based user interface for interactive exploration.

## Features

- Black-Scholes option pricer with call and put payoffs
- Configurable mesh generation supporting 1D, 2D and 3D problems
- Sample problems for 1D Black–Scholes and 3D Heston models
- Simple market abstraction with discount factors
- Streamlit UI and plotting helpers
- Example demo for adaptive mesh refinement
- Comprehensive test suite using `pytest`

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
