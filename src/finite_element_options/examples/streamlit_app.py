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
    if not parameters.can_solve:
        st.warning("This configuration is not supported by the released FEM capability manifest.")
        status = parameters.validated.to_status_dict() if parameters.validated else {}
        st.json(status)
        return
    if not parameters.requires_numerical_solve:
        st.info("Configuration resolved to an explicit analytical limit; no FEM solve allocated.")
        status = parameters.validated.to_status_dict() if parameters.validated else {}
        st.json(status)
        return

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

    if hasattr(dynamics, "cir_message"):
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
        if parameters.validated is not None:
            status = parameters.validated.to_status_dict(
                solver_diagnostics=stepper.last_solve_diagnostics.to_public_dict(),
                domain_diagnostics=stepper.last_domain_diagnostics,
                time_grid_diagnostics=stepper.last_time_grid_diagnostics,
            )
            st.json(status)

    terminal_values = values[-1]
    delta = basis.project(basis.interpolate(terminal_values).grad[0])
    gamma = basis.project(basis.interpolate(delta).grad[0])

    plot_data = {
        "Option Value": terminal_values,
        "Delta": delta,
        "Gamma": gamma,
    }
    if basis.doflocs.shape[0] > 1:
        variance_vega = basis.project(basis.interpolate(terminal_values).grad[1])
        plot_data["Variance Vega"] = variance_vega

    for title, field_values in plot_data.items():
        plots.plot_2d(basis, field_values, title)


if __name__ == "__main__":
    main()
