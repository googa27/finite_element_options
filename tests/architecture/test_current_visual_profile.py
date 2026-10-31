"""Current renderer admission preserves the historical publication boundary."""

from __future__ import annotations

from pathlib import Path
import runpy
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "scripts/generate_jax_regime_plot.py"


def test_registered_profiles_keep_the_historical_encoder_identity() -> None:
    """Only the supporting encoder changes in the separately named current route."""

    namespace = runpy.run_path(str(GENERATOR))
    select = namespace["_expected_visual_stack"]
    historical = select("historical", publish_canonical=True)
    current = select("current", publish_canonical=False)
    assert historical == {
        "matplotlib": "3.10.3",
        "numpy": "2.4.6",
        "pillow": "12.3.0",
    }
    assert current == {**historical, "pillow": "13.0.0"}
    with pytest.raises(ValueError, match="unknown visual profile"):
        select("unregistered", publish_canonical=False)
    with pytest.raises(ValueError, match="cannot replace historical"):
        select("current", publish_canonical=True)


def test_current_profile_refuses_historical_publication_before_input_io(
    tmp_path: Path,
) -> None:
    """A new renderer cannot overwrite old evidence even with publication opt-in."""

    result = subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            "--visual-profile",
            "current",
            "--publish-canonical",
            "--input",
            str(tmp_path / "missing-input.json"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "current visual profile cannot replace historical" in result.stderr
    assert not list(tmp_path.iterdir())
