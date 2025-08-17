# Benchmarking

This project uses [`pytest-benchmark`](https://pytest-benchmark.readthedocs.io/) to
track solver runtime on fixed meshes.  Benchmarks are executed both locally and
on the continuous integration server.

## Running Benchmarks Locally

Run the benchmark tests in isolation to focus on performance measurements:

```bash
pytest tests/test_benchmark_black_scholes.py --benchmark-only
```

To store results for later comparison, write them to a JSON file:

```bash
pytest tests/test_benchmark_black_scholes.py --benchmark-json=benchmark.json
```

## Interpreting Results

The JSON output records statistics such as mean, median and minimum runtime for
each benchmarked function.  Lower values indicate faster solver performance.
`pytest-benchmark` can compare multiple runs to track regressions:

```bash
pytest-benchmark compare benchmark.json path/to/previous.json
```

CI runs publish the latest `benchmark.json` file as a workflow artifact, making
it easy to download and contrast with local measurements.
