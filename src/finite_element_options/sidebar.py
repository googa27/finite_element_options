"""Streamlit sidebar widgets for interactive option demos."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from finite_element_options.core.dynamics_black_scholes import (
    DynamicsParametersBlackScholes,
)
from finite_element_options.core.dynamics_heston import DynamicsParametersHeston
from finite_element_options.core.market import Market
from finite_element_options.core.vanilla_bs import EuropeanOptionBs
from finite_element_options.space.mesh import create_mesh
from finite_element_options.ui_config import (
    UiGridSpec,
    UiModelSpec,
    UiSolverOptions,
    ValidatedUiProblem,
    validate_ui_problem,
)


class Sidebar:
    """Collect and validate user input from the Streamlit sidebar."""

    def __init__(self) -> None:
        """Instantiate sidebar widgets without allocating invalid solver resources."""

        self.can_solve = False
        self.requires_numerical_solve = False
        self.validated: ValidatedUiProblem | None = None
        self._make_sidebar()

    @property
    def diagnostics_payload(self) -> list[dict[str, Any]]:
        """Return public diagnostics for Streamlit display."""

        if self.validated is None:
            return []
        return [diag.to_public_dict() for diag in self.validated.diagnostics]

    def _make_sidebar(self) -> None:
        st = _streamlit()
        with st.sidebar:
            st.title("Option Parameters")
            model_choice = st.selectbox(
                "Model route",
                (
                    "Black-Scholes 1D (validated)",
                    "Heston 2D (disabled until capability evidence lands)",
                ),
                index=0,
            )
            model_name: Literal["black_scholes", "heston"] = (
                "black_scholes" if model_choice.startswith("Black") else "heston"
            )
            self.is_call = st.checkbox("call", value=True)
            is_american = st.checkbox(
                "american",
                value=False,
                help="Disabled by capability validation until the LCP route lands.",
            )

            strike = st.slider("Strike", 0.1, 2.0, 0.4)
            maturity = st.slider("Maturity", 0.0, 5.0, 1.0)

            st.title("Market Parameters")
            rate = st.slider("r", -0.1, 0.5, 0.03)
            carry = st.slider("q", 0.0, 1.0, 0.03)
            volatility = st.slider("volatility", 0.0, 0.8, 0.2)

            heston_kwargs: dict[str, float | None] = {
                "kappa": None,
                "long_run_variance": None,
                "vol_of_variance": None,
                "correlation": None,
            }
            if model_name == "heston":
                st.title("Heston Parameters")
                heston_kwargs = {
                    "kappa": st.slider("kappa", 0.0, 2.0, 0.5),
                    "long_run_variance": st.slider("long-run variance", 0.0, 1.0, 0.04),
                    "vol_of_variance": st.slider("vol-of-variance", 0.0, 0.8, 0.2),
                    "correlation": st.slider("correlation", -1.0, 1.0, 0.5),
                }

            st.title("Boundary Conditions")
            boundary_options = ["s_min", "s_max"]
            if model_name == "heston":
                boundary_options.extend(["v_min", "v_max"])
            dirichlet_bcs = tuple(
                st.multiselect(
                    "Dirichlet Boundary Conditions",
                    boundary_options,
                    ["s_min", "s_max"],
                )
            )

            st.title("Discretization Parameters")
            mesh_refine = st.slider("mesh refine", 1, 8, 4)
            time_steps = st.slider("time steps", 2, 1000, 100)
            theta_scheme = st.slider("theta", 0.0, 1.0, 0.5)

        model = UiModelSpec(
            model=model_name,
            strike=strike,
            maturity=maturity,
            rate=rate,
            carry=carry,
            volatility=volatility,
            **heston_kwargs,
        )
        grid = UiGridSpec(mesh_refine=mesh_refine, time_steps=time_steps)
        exercise_style: Literal["american", "european"] = (
            "american" if is_american else "european"
        )
        solver = UiSolverOptions(
            theta=theta_scheme,
            exercise_style=exercise_style,
            dirichlet_boundaries=dirichlet_bcs,
        )
        self.validated = validate_ui_problem(model=model, grid=grid, solver=solver)

        with st.sidebar:
            st.caption(self.validated.work_estimate.summary)
            for diagnostic in self.validated.diagnostics:
                if diagnostic.severity == "error":
                    st.error(f"{diagnostic.field}: {diagnostic.message}")
                elif diagnostic.severity == "warning":
                    st.warning(f"{diagnostic.field}: {diagnostic.message}")
                else:
                    st.info(diagnostic.message)

        self.can_solve = self.validated.can_run
        self.requires_numerical_solve = self.validated.requires_numerical_solve
        self.is_american = is_american
        self.dirichlet_bcs = dirichlet_bcs
        self.lam = theta_scheme
        if not self.can_solve or not self.requires_numerical_solve:
            return

        self.mkt = Market(r=rate)
        dynamics: DynamicsParametersBlackScholes | DynamicsParametersHeston
        if model_name == "black_scholes":
            dynamics = DynamicsParametersBlackScholes(r=rate, q=carry, sig=volatility)
        else:
            assert model.kappa is not None
            assert model.long_run_variance is not None
            assert model.vol_of_variance is not None
            assert model.correlation is not None
            dynamics = DynamicsParametersHeston(
                r=rate,
                q=carry,
                kappa=model.kappa,
                theta=model.long_run_variance,
                sig=model.vol_of_variance,
                rho=model.correlation,
            )
        self.dh = dynamics
        self.bsopt = EuropeanOptionBs(strike, carry, self.mkt)
        self.t = np.linspace(0.0, maturity, time_steps)
        self.mesh, self.config = create_mesh(self.validated.domain_axes, mesh_refine)


def _streamlit():
    """Import Streamlit only for the optional UI extra."""

    try:
        import streamlit as st  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in clean extras CI
        raise ModuleNotFoundError(
            "finite_element_options.sidebar requires the 'ui' extra: "
            "pip install 'finite-element-options[ui]'"
        ) from exc
    return st
