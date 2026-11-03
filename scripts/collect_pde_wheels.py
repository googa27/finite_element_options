"""Collect six auxiliary CPython3.13 wheels; never claim an installed runtime."""

from __future__ import annotations

import argparse
from email import policy
from email.parser import Parser
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from urllib.parse import urlsplit
from urllib.request import urlopen
import zipfile

from packaging.tags import parse_tag, sys_tags
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name, parse_wheel_filename

SPECS = ("meshio", "pyvista", "scooby", "mpi4py", "gmsh", "fenics-basix==0.11.0")
NAMES = {spec.split("==")[0] for spec in SPECS}
MAX_BYTES = 100 * 1024**2


def inspect_wheel(path: Path) -> dict:
    """Inspect distribution metadata without installing or importing a wheel."""
    name, version, _, tags = parse_wheel_filename(path.name)
    if not tags.intersection(sys_tags()):
        raise ValueError(
            f"wheel does not match executing interpreter/platform: {path.name}"
        )
    with zipfile.ZipFile(path) as archive:
        metadata = [
            i for i in archive.infolist() if i.filename.endswith(".dist-info/METADATA")
        ]
        wheel = [
            i for i in archive.infolist() if i.filename.endswith(".dist-info/WHEEL")
        ]
        if len(metadata) != 1 or len(wheel) != 1:
            raise ValueError("exactly one distribution METADATA/WHEEL pair is required")
        if metadata[0].file_size > 131072 or wheel[0].file_size > 131072:
            raise ValueError("wheel metadata exceeds bounded inspection size")
        if (
            metadata[0].filename.rsplit("/", 1)[0]
            != wheel[0].filename.rsplit("/", 1)[0]
        ):
            raise ValueError("wheel metadata directories disagree")
        raw = archive.read(metadata[0])
        # Core metadata is UTF-8 text using email's compat32 grammar. Parse
        # decoded text so valid non-ASCII headers remain JSON-safe Unicode,
        # rather than compat32 BytesParser's email.header.Header wrappers.
        parser = Parser(policy=policy.compat32)
        info = parser.parsestr(raw.decode("utf-8"))
        wheel_info = parser.parsestr(archive.read(wheel[0]).decode("utf-8"))
        if info.defects or wheel_info.defects:
            raise ValueError("malformed wheel metadata headers")
    if canonicalize_name(info["Name"]) != name or info["Version"] != str(version):
        raise ValueError("wheel filename and distribution identity disagree")
    if platform.python_version() not in SpecifierSet(info.get("Requires-Python", "")):
        raise ValueError("wheel Requires-Python excludes the executing interpreter")
    declared_tags = set().union(*(parse_tag(t) for t in wheel_info.get_all("Tag", [])))
    if declared_tags != tags:
        raise ValueError("wheel filename and WHEEL compatibility tags disagree")
    return {
        "name": name,
        "version": str(version),
        "filename": path.name,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "tags": sorted(map(str, tags)),
        "requires_python": info.get("Requires-Python"),
        "license": info.get("License-Expression", info.get("License")),
        "metadata_sha256": hashlib.sha256(raw).hexdigest(),
    }


def validate_wheels(
    directory: Path, receipts: dict, *, max_bytes: int = MAX_BYTES
) -> list[dict]:
    """Require all six exact files and independently published PyPI hashes."""
    files = list(directory.iterdir())
    if len(files) != len(SPECS) or any(
        p.is_symlink() or not p.is_file() or p.suffix != ".whl" for p in files
    ):
        raise ValueError("wheelhouse requires exactly six regular wheel files")
    if sum(p.stat().st_size for p in files) > max_bytes:
        raise ValueError("wheelhouse exceeds total100MiB collection budget")
    rows = [inspect_wheel(p) for p in sorted(files)]
    if {r["name"] for r in rows} != NAMES:
        raise ValueError("wheelhouse is missing or duplicates expected distributions")
    for row in rows:
        name, version = row["name"], row["version"]
        if name == "fenics-basix" and version != "0.11.0":
            raise ValueError("fenics-basix must preserve original0.11.0 requirement")
        release = receipts[(name, version)]
        if (
            canonicalize_name(release["info"]["name"]) != name
            or release["info"]["version"] != version
        ):
            raise ValueError("PyPI release metadata and wheel identity disagree")
        matches = [f for f in release["urls"] if f["filename"] == row["filename"]]
        if len(matches) != 1:
            raise ValueError("wheel is not a unique file in the exact PyPI release")
        published = matches[0]
        url = urlsplit(published["url"])
        if (
            published["packagetype"] != "bdist_wheel"
            or published["yanked"]
            or url.scheme != "https"
            or url.hostname != "files.pythonhosted.org"
            or published["size"] != row["bytes"]
            or published["digests"]["sha256"] != row["sha256"]
        ):
            raise ValueError(
                "wheel artifact differs from non-yanked HTTPS PyPI release file"
            )
        row["download_url"] = published["url"]
        row["release_metadata_url"] = f"https://pypi.org/pypi/{name}/{version}/json"
    return rows


def release_receipt(name: str, version: str) -> dict:
    """Read a bounded primary release record using HTTPS certificate checks."""
    url = f"https://pypi.org/pypi/{name}/{version}/json"
    with urlopen(url, timeout=30) as response:
        if urlsplit(response.url).hostname != "pypi.org":
            raise ValueError("unexpected PyPI metadata redirect")
        raw = response.read(2 * 1024**2 + 1)
    if len(raw) > 2 * 1024**2:
        raise ValueError("PyPI release record exceeds2MiB")
    return json.loads(raw)


def collect(output: Path) -> None:
    """Download binary artifacts, record provenance, and propagate any failure."""
    if sys.version_info[:2] != (3, 13) or platform.python_implementation() != "CPython":
        raise ValueError("auxiliary collection requires actual CPython3.13")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("auxiliary collection requires Linux-x86_64")
    output.mkdir(parents=True, exist_ok=False)
    wheelhouse = output / "wheelhouse"
    wheelhouse.mkdir()
    argv = [
        sys.executable,
        "-I",
        "-m",
        "pip",
        "--isolated",
        "--disable-pip-version-check",
        "download",
        "--only-binary=:all:",
        "--no-deps",
        "--no-cache-dir",
        "--timeout",
        "30",
        "--retries",
        "1",
        "--index-url",
        "https://pypi.org/simple",
        "--dest",
        str(wheelhouse),
        *SPECS,
    ]
    result = subprocess.run(
        argv, capture_output=True, text=True, check=False, timeout=300
    )
    (output / "status.txt").write_text(result.stdout + result.stderr)
    result.check_returncode()
    files = list(wheelhouse.iterdir())
    if len(files) != 6 or sum(p.stat().st_size for p in files) > MAX_BYTES:
        raise ValueError("downloaded collection is incomplete or exceeds100MiB")
    identities = [inspect_wheel(p) for p in files]
    if {r["name"] for r in identities} != NAMES:
        raise ValueError(
            "unexpected downloaded distribution; refusing metadata request"
        )
    receipts = {
        (r["name"], r["version"]): release_receipt(r["name"], r["version"])
        for r in identities
    }
    rows = validate_wheels(wheelhouse, receipts)
    root = Path(__file__).resolve().parents[1]
    source = subprocess.check_output(
        ["git", "rev-parse", "HEAD", "HEAD^{tree}"], cwd=root, text=True
    ).splitlines()
    manifest = {
        "schema": 1,
        "specs": list(SPECS),
        "python": sys.version,
        "platform": platform.platform(),
        "source_commit": source[0],
        "source_tree": source[1],
        "command": argv,
        "total_bytes": sum(r["bytes"] for r in rows),
        "files": rows,
        "limit": "Six no-dependency wheels are auxiliary download seeds, not a complete or installed scientific runtime; no native compatibility claim.",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    (output / "pypi-release-records.json").write_text(
        json.dumps({f"{n}=={v}": d for (n, v), d in receipts.items()}, indent=2) + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        collect(args.output.resolve())
    except (
        ValueError,
        KeyError,
        TypeError,
        OSError,
        subprocess.SubprocessError,
        zipfile.BadZipFile,
    ) as error:
        print(f"wheel collection refused: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
