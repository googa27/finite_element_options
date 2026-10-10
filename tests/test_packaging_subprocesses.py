"""Real child-process regressions for the packaging observer (issue168)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from test_packaging_contract import _run

pytestmark = pytest.mark.packaging


@pytest.mark.parametrize("explicit", [False, True])
def test_children_reject_ambient_checkout_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit: bool
) -> None:
    poison = tmp_path / "checkout"
    poison.mkdir()
    (poison / "fem_observer_poison.py").write_text("VALUE = 99\n")
    metadata = poison / "fem_observer_poison-99.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: fem-observer-poison\nVersion: 99.0\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(poison))
    env = dict(os.environ) if explicit else None
    before = dict(env) if env is not None else None
    code = (
        "import importlib.metadata as md, importlib.util, json; "
        "print(json.dumps({'module': "
        "importlib.util.find_spec('fem_observer_poison') is not None, "
        "'distribution': any(d.metadata['Name'] == 'fem-observer-poison' "
        "for d in md.distributions())}))"
    )
    observed = json.loads(_run([sys.executable, "-B", "-c", code], cwd=tmp_path, env=env))
    assert observed == {"module": False, "distribution": False}
    assert env == before
    assert os.environ["PYTHONPATH"] == str(poison)


@pytest.mark.parametrize("explicit", [False, True])
def test_children_ignore_invalid_python_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit: bool
) -> None:
    monkeypatch.setenv("PYTHONHOME", str(tmp_path / "missing-python-home"))
    env = dict(os.environ) if explicit else None
    result = _run(
        [sys.executable, "-B", "-c", "print('actual interpreter started')"],
        cwd=tmp_path,
        env=env,
    )
    assert result.strip() == "actual interpreter started"
    assert os.environ["PYTHONHOME"] == str(tmp_path / "missing-python-home")


def test_children_preserve_private_configuration(tmp_path: Path) -> None:
    values = {
        "PIP_CACHE_DIR": str(tmp_path / "private-cache"),
        "TMPDIR": str(tmp_path / "private-temp"),
        "PIP_INDEX_URL": "https://example.invalid/private-simple",
        "PIP_CERT": str(tmp_path / "private-cert.pem"),
        "SSL_CERT_FILE": str(tmp_path / "private-ca.pem"),
    }
    env = {**os.environ, **values}
    before = dict(env)
    code = (
        "import json, os; print(json.dumps({k: os.environ.get(k) for k in "
        + repr(list(values))
        + "}))"
    )
    assert json.loads(
        _run([sys.executable, "-I", "-B", "-c", code], cwd=tmp_path, env=env)
    ) == values
    assert env == before


def _failure_record(command: list[str], cwd: Path) -> dict:
    with pytest.raises(AssertionError) as failure:
        _run(command, cwd=cwd)
    try:
        record = json.loads(str(failure.value))
    except json.JSONDecodeError:
        pytest.fail("failed child lost structured argv/cwd/exit/output diagnostics")
    assert record["argv"] == command
    assert record["cwd"] == str(cwd)
    return record


def test_failed_children_retain_original_diagnostics(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-I",
        "-B",
        "-c",
        "import sys; print('original-stdout'); "
        "print('original-stderr', file=sys.stderr); sys.exit(7)",
    ]
    record = _failure_record(command, tmp_path)
    assert record["returncode"] == 7
    assert record["stdout"] == "original-stdout\n"
    assert record["stderr"] == "original-stderr\n"


def test_real_ensurepip_refusal_retains_original_diagnostics(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-I",
        "-B",
        "-m",
        "ensurepip",
        "--fem-observer-bootstrap-refusal",
    ]
    record = _failure_record(command, tmp_path)
    assert record["returncode"] == 2
    assert record["stdout"] == ""
    assert "unrecognized arguments: --fem-observer-bootstrap-refusal" in record["stderr"]
