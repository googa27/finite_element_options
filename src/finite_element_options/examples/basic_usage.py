"""Basic installed-package usage example for finite_element_options."""

from __future__ import annotations

from finite_element_options.examples.bs_1d import price_call


def main() -> None:
    """Run a simple Black-Scholes call option pricing example."""
    grid = price_call()
    print(grid[-1])


if __name__ == "__main__":
    main()
