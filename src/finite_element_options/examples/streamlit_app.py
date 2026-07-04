"""Installed-package Streamlit application entry point."""

from __future__ import annotations

import streamlit as st
from skfem.visuals.matplotlib import plot as _skfem_plot  # noqa: F401

import finite_element_options.plots as plots
import finite_element_options.sidebar as sidebar
from finite_element_options.space.boundary import DirichletBC
from finite_element_options.space.solver import SpaceSolver
from finite_element_options.time_integration.stepper import ThetaScheme


def main() -> None:
    """Run the Streamlit finite-element option-pricing demo."""
    parameters = sidebar.Sidebar()
    dynamics = parameters.dh
    option = parameters.bsopt
    time_grid = parameters.t

    space = SpaceSolver(
        parameters.mesh,
        dynamics,
        option,
        is_call=parameters.is_call,
        config=parameters.config,
    )
    stepper = ThetaScheme(theta=parameters.lam)

    with st.sidebar:
        st.write(dynamics.cir_message())

    boundary_condition = DirichletBC(parameters.dirichlet_bcs)
    values = stepper.solve(
        time_grid,
        space,
        boundary_condition=boundary_condition,
        is_american=parameters.is_american,
    )

    basis = space.Vh
    with st.sidebar:
        st.write(f"Number of degrees of freedom: {basis.N}")

    terminal_values = values[-1]
    delta = basis.project(basis.interpolate(terminal_values).grad[0])
    gamma = basis.project(basis.interpolate(delta).grad[0])
    variance_vega = basis.project(basis.interpolate(terminal_values).grad[1])

    plot_data = {
        "Option Value": terminal_values,
        "Delta": delta,
        "Gamma": gamma,
        "Variance Vega": variance_vega,
    }
    for title, field_values in plot_data.items():
        plots.plot_2d(basis, field_values, title)


if __name__ == "__main__":
    main()
