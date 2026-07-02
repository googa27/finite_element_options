"""Basic usage examples for the finite_element_options package."""

from finite_element_options.examples.bs_1d import price_call


def main() -> None:
    """Run a simple Black–Scholes call option pricing example."""
    grid = price_call()
    print(grid[-1])  # option values at maturity


if __name__ == "__main__":
    main()
