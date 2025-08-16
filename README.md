# Finite Element Options

Finite Element Options demonstrates pricing of European options under the
Black-Scholes model using a finite element discretisation.  The application is
built around [scikit-fem](https://github.com/kinnala/scikit-fem) and provides a
Streamlit based user interface for interactive exploration.

## Features

- Black-Scholes option pricer with call and put payoffs
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

## License

This project is provided as-is without warranty.
