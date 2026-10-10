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
    observed = json.loads(
        _run([sys.executable, "-B", "-c", code], cwd=tmp_path, env=env)
    )
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
    assert (
        json.loads(
            _run([sys.executable, "-I", "-B", "-c", code], cwd=tmp_path, env=env)
        )
        == values
    )
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
    assert (
        "unrecognized arguments: --fem-observer-bootstrap-refusal" in record["stderr"]
    )


@pytest.mark.parametrize("explicit", [False, True])
def test_children_clear_unrelated_virtual_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit: bool
) -> None:
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "unrelated-active-environment"))
    env = dict(os.environ) if explicit else None
    result = _run(
        [sys.executable, "-B", "-c", "import os; print('VIRTUAL_ENV' in os.environ)"],
        cwd=tmp_path,
        env=env,
    )
    assert result.strip() == "False"
    assert os.environ["VIRTUAL_ENV"] == str(tmp_path / "unrelated-active-environment")


def _actual_factory():
    import test_packaging_contract as consumer

    factory = getattr(consumer, "_create_venv", None)
    assert callable(factory), (
        "packaging consumer has no explicit diagnostic venv factory"
    )
    return factory


def test_factory_preserves_actual_ensurepip_failure(tmp_path: Path) -> None:
    target = tmp_path / "failed-bootstrap"
    with pytest.raises(AssertionError) as failure:
        _actual_factory()(
            target,
            cwd=tmp_path,
            ensurepip_args=("--fem-observer-bootstrap-refusal",),
        )
    try:
        record = json.loads(str(failure.value))
    except json.JSONDecodeError:
        pytest.fail("factory did not expose its original ensurepip failure record")
    python = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    assert record["argv"] == [
        str(python),
        "-I",
        "-B",
        "-m",
        "ensurepip",
        "--fem-observer-bootstrap-refusal",
    ]
    assert record["cwd"] == str(tmp_path)
    assert record["returncode"] == 2
    assert record["stdout"] == ""
    assert (
        "unrecognized arguments: --fem-observer-bootstrap-refusal" in record["stderr"]
    )
    assert (target / "pyvenv.cfg").is_file()
    result = _run(
        [
            str(python),
            "-I",
            "-B",
            "-c",
            "import importlib.util; print(importlib.util.find_spec('pip') is None)",
        ],
        cwd=tmp_path,
    )
    assert result.strip() == "True"
    print("actual factory bootstrap refusal:", json.dumps(record, sort_keys=True))


def test_factory_bootstraps_an_isolated_target(tmp_path: Path) -> None:
    target = tmp_path / "successful-bootstrap"
    python = _actual_factory()(target, cwd=tmp_path)
    result = _run(
        [
            str(python),
            "-I",
            "-B",
            "-c",
            "import json, pip, sys; print(json.dumps([sys.prefix, pip.__file__]))",
        ],
        cwd=tmp_path,
    )
    prefix, pip_file = json.loads(result)
    assert Path(prefix).resolve() == target.resolve()
    assert Path(pip_file).resolve().is_relative_to(target.resolve())
    assert "include-system-site-packages = false" in (target / "pyvenv.cfg").read_text()


def test_same_version_checkout_cannot_satisfy_normal_wheel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tomllib

    import test_packaging_contract as consumer

    project = tomllib.loads((consumer.ROOT / "pyproject.toml").read_text())
    version = project["project"]["version"]
    poison = tmp_path / "same-version-checkout"
    poison.mkdir()
    package = poison / "finite_element_options"
    package.mkdir()
    (package / "__init__.py").write_text("POISON_CHECKOUT = True\n")
    metadata = poison / f"finite_element_options-{version}.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: finite-element-options\nVersion: {version}\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(poison))
    consumer.test_installed_wheel_import_contract_has_no_checkout_path_hack(tmp_path)

    wheel = next((tmp_path / "dist").glob("finite_element_options-*.whl"))
    python = (
        tmp_path / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    )
    code = """
import hashlib
import importlib.metadata as md
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

import finite_element_options as fem

wheel = Path(sys.argv[1]).resolve()
target = Path(sys.argv[2]).resolve()
assert sys.flags.isolated
assert Path(sys.prefix).resolve() == target
assert Path(fem.__file__).resolve().is_relative_to(target)
assert not hasattr(fem, "POISON_CHECKOUT")
assert importlib.util.find_spec("src") is None
dist = md.distribution("finite-element-options")
direct = json.loads(dist.read_text("direct_url.json"))
assert direct["url"] == wheel.as_uri()
assert not direct.get("dir_info", {}).get("editable", False)
with zipfile.ZipFile(wheel) as archive:
    members = [n for n in archive.namelist()
               if n.startswith("finite_element_options/") and not n.endswith("/")]
    assert members
    for name in members:
        actual = Path(dist.locate_file(name)).resolve()
        assert actual.is_relative_to(target), name
        assert actual.read_bytes() == archive.read(name), name
print(json.dumps({"normal_members": len(members),
                  "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                  "isolated": True}))
"""
    observed = json.loads(
        _run(
            [str(python), "-I", "-B", "-c", code, str(wheel), str(tmp_path / "venv")],
            cwd=tmp_path,
        )
    )
    assert observed["normal_members"] > 0
    assert observed["isolated"] is True
    print("poisoned same-version consumer normal authority:", json.dumps(observed))


def test_wheel_consumer_excludes_parent_user_site(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import test_packaging_contract as consumer

    monkeypatch.setenv("PYTHONUSERBASE", str(tmp_path / "parent-user-base"))
    user_site = Path(
        _run(
            [
                sys.executable,
                "-B",
                "-c",
                "import site; print(site.getusersitepackages())",
            ],
            cwd=tmp_path,
        ).strip()
    )
    assert user_site.is_relative_to(tmp_path)
    user_site.mkdir(parents=True)
    (user_site / "fem_parent_user_poison.py").write_text("PARENT_ONLY = True\n")
    consumer.test_installed_wheel_import_contract_has_no_checkout_path_hack(tmp_path)
    python = (
        tmp_path / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    )
    result = _run(
        [
            str(python),
            "-B",
            "-c",
            "import importlib.util; "
            "print(importlib.util.find_spec('fem_parent_user_poison') is None)",
        ],
        cwd=tmp_path,
    )
    assert result.strip() == "True"


def test_installed_authority_rejects_foreign_namespace(tmp_path: Path) -> None:
    import test_packaging_contract as consumer

    consumer.test_installed_wheel_import_contract_has_no_checkout_path_hack(tmp_path)
    wheel = next((tmp_path / "dist").glob("finite_element_options-*.whl"))
    python = (
        tmp_path / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    )
    code = """
import importlib.machinery
import sys
import types

name = "finite_element_options.foreign_namespace"
module = types.ModuleType(name)
module.__path__ = [str(Path.cwd() / "foreign-namespace")]
module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
module.__spec__.submodule_search_locations = module.__path__
sys.modules[name] = module
"""
    with pytest.raises(AssertionError) as failure:
        consumer._run_installed(python, wheel, code, cwd=tmp_path)
    record = json.loads(str(failure.value))
    assert record["returncode"] == 1
    assert "finite_element_options.foreign_namespace" in record["stderr"]
