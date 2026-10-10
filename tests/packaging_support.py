"""Checked child execution shared by packaging tests, never runtime code."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from email import message_from_bytes
from pathlib import Path

from packaging.requirements import Requirement


def run_checked(
    command: list[str], *, cwd: Path, env: dict[str, str] | None = None
) -> str:
    """Isolate source/home selection and preserve genuine child failures."""
    child_env = dict(os.environ if env is None else env)
    for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        child_env.pop(name, None)
    result = subprocess.run(
        command,
        cwd=cwd,
        env=child_env,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode:
        raise AssertionError(
            json.dumps(
                {
                    "argv": command,
                    "cwd": str(cwd),
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                },
                sort_keys=True,
            )
        )
    return result.stdout + result.stderr


def create_venv(
    path: Path,
    *,
    cwd: Path,
    ensurepip_args: tuple[str, ...] = ("--upgrade", "--default-pip"),
) -> Path:
    """Create an isolated target and expose its actual pip bootstrap diagnostics."""
    run_checked(
        [sys.executable, "-I", "-B", "-m", "venv", "--without-pip", str(path)],
        cwd=cwd,
    )
    python = path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    run_checked([str(python), "-I", "-B", "-m", "ensurepip", *ensurepip_args], cwd=cwd)
    return python


def core_requirements(wheel: Path) -> list[str]:
    """Read the reviewed wheel's active base requirements, without any extras."""
    with zipfile.ZipFile(wheel) as archive:
        names = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        assert len(names) == 1, names
        message = message_from_bytes(archive.read(names[0]))
    requirements = []
    for value in message.get_all("Requires-Dist", []):
        requirement = Requirement(value)
        if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
            requirements.append(str(requirement))
    assert requirements, "normal wheel declared no core requirements"
    return requirements


_NORMAL_AUTHORITY = """
import hashlib
import importlib.metadata as md
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

def _assert_normal_authority(report):
    wheel = Path(sys.argv[1]).resolve()
    prefix = Path(sys.argv[2]).resolve()
    assert sys.flags.isolated
    assert Path(sys.prefix).resolve() == prefix
    assert importlib.util.find_spec("src") is None
    dist = md.distribution("finite-element-options")
    direct = json.loads(dist.read_text("direct_url.json"))
    assert direct["url"] == wheel.as_uri()
    assert not direct.get("dir_info", {}).get("editable", False)
    with zipfile.ZipFile(wheel) as archive:
        members = [name for name in archive.namelist()
                   if name.startswith("finite_element_options/") and not name.endswith("/")]
        assert members
        for name in members:
            actual = Path(dist.locate_file(name)).resolve()
            assert actual.is_relative_to(prefix), name
            assert actual.read_bytes() == archive.read(name), name
    for name, module in list(sys.modules.items()):
        if name == "finite_element_options" or name.startswith("finite_element_options."):
            origin = getattr(module, "__file__", None)
            if origin is not None:
                assert Path(origin).resolve().is_relative_to(prefix), (name, origin)
            else:
                paths = [Path(p).resolve() for p in getattr(module, "__path__", [])]
                relative = name.replace(".", "/")
                expected = Path(dist.locate_file(relative)).resolve()
                assert paths == [expected] and expected.is_relative_to(prefix), name
                assert any(member.startswith(relative + "/") for member in members), name
                spec = module.__spec__
                assert spec is not None and spec.origin is None, name
                assert spec.submodule_search_locations is not None, name
    if report:
        print(json.dumps({"normal_members": len(members),
                          "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                          "isolated": True}))
_assert_normal_authority(False)
"""


def run_installed(python: Path, wheel: Path, code: str, *, cwd: Path) -> str:
    """Keep the original consumer assertions between full byte/origin checks."""
    return run_checked(
        [
            str(python),
            "-I",
            "-B",
            "-c",
            _NORMAL_AUTHORITY + "\n" + code + "\n_assert_normal_authority(True)\n",
            str(wheel.resolve()),
            str(python.parent.parent.resolve()),
        ],
        cwd=cwd,
    )
