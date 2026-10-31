"""Real installed Streamlit interactions for the maintained UI cohort."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import finite_element_options
from streamlit.testing.v1 import AppTest


def _app() -> AppTest:
    """Use the normal installed application from an isolated consumer."""

    package = Path(finite_element_options.__file__).resolve().parent
    assert sys.flags.isolated
    assert package.is_relative_to(Path(sys.prefix).resolve())
    return AppTest.from_file(str(package / "examples/streamlit_app.py")).run(timeout=60)


def _status(app: AppTest) -> dict:
    """Read the application's public diagnostic record."""

    assert not app.exception
    return json.loads(app.json[-1].value)


def test_default_ui_executes_real_fem_and_renders_all_three_fields() -> None:
    """The installed application emits actual solve diagnostics and three images."""

    app = _app()
    status = _status(app)
    assert status["can_run"] is True
    assert status["requires_numerical_solve"] is True
    assert status["solver_diagnostics"]["solve_count"] == 99
    assert status["solver_diagnostics"]["max_linear_residual_abs"] < 1e-12
    assert status["approximation_status"]["estimated_dofs"] == 33
    assert len(app.get("image")) == 3


def test_ui_expiry_interaction_selects_analytical_limit_without_mesh_solve() -> None:
    """A real widget change reaches the public analytical route."""

    app = _app()
    next(item for item in app.slider if item.label == "Maturity").set_value(0.0)
    app.run(timeout=60)
    status = _status(app)
    assert status["can_run"] is True
    assert status["requires_numerical_solve"] is False
    assert status["solver_diagnostics"] == {}
    assert len(app.get("image")) == 0
    assert any("analytical limit" in item.value for item in app.info)


def test_ui_unsupported_heston_selection_refuses_before_solve_or_render() -> None:
    """Changing the model cannot silently advertise an unsupported FEM route."""

    app = _app()
    app.selectbox[0].select("Heston 2D (disabled until capability evidence lands)")
    app.run(timeout=60)
    status = _status(app)
    assert status["can_run"] is False
    assert status["solver_diagnostics"] == {}
    assert len(app.get("image")) == 0
    assert any("not supported" in item.value for item in app.warning)
