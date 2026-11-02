"""Offline refusal controls; fabricated wheel metadata is not acquisition proof."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/collect_pde_wheels.py"


class WheelCollectionTests(unittest.TestCase):
    def module(self):
        spec = importlib.util.spec_from_file_location("pde_wheels", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def fixture(self, directory):
        receipts = {}
        for name in ("meshio", "pyvista", "scooby", "mpi4py", "gmsh", "fenics-basix"):
            version = "0.11.0" if name == "fenics-basix" else "1.0.0"
            dist = name.replace("-", "_")
            file = directory / f"{dist}-{version}-py3-none-any.whl"
            with zipfile.ZipFile(file, "w") as archive:
                archive.writestr(
                    f"{dist}-{version}.dist-info/METADATA",
                    f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
                )
                archive.writestr(
                    f"{dist}-{version}.dist-info/WHEEL",
                    "Wheel-Version: 1.0\nTag: py3-none-any\n",
                )
            receipts[(name, version)] = {
                "info": {"name": name, "version": version},
                "urls": [
                    {
                        "filename": file.name,
                        "packagetype": "bdist_wheel",
                        "yanked": False,
                        "url": "https://files.pythonhosted.org/packages/fixture/"
                        + file.name,
                        "size": file.stat().st_size,
                        "digests": {
                            "sha256": hashlib.sha256(file.read_bytes()).hexdigest()
                        },
                    }
                ],
            }
        return receipts

    def test_complete_collection_is_bound_to_each_exact_release_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            receipts = self.fixture(path)
            rows = self.module().validate_wheels(path, receipts)
            self.assertEqual(len(rows), 6)
            self.assertEqual({r["name"] for r in rows}, {k[0] for k in receipts})
            for row in rows:
                self.assertEqual(
                    row["sha256"],
                    hashlib.sha256((path / row["filename"]).read_bytes()).hexdigest(),
                )

    def test_missing_duplicate_and_nonwheel_payloads_are_refused(self):
        for change in ("missing", "duplicate", "nonwheel"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as d:
                path = Path(d)
                receipts = self.fixture(path)
                if change == "missing":
                    next(path.glob("meshio-*.whl")).unlink()
                elif change == "duplicate":
                    source = next(path.glob("meshio-*.whl"))
                    (path / source.name.replace("-py3-", "-py2-")).write_bytes(
                        source.read_bytes()
                    )
                else:
                    (path / "extra.tar.gz").write_bytes(b"not a wheel")
                with self.assertRaises(ValueError):
                    self.module().validate_wheels(path, receipts)

    def test_digest_yanked_version_metadata_and_size_fail_closed(self):
        for change in ("digest", "yanked", "metadata", "version", "size", "origin"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as d:
                path = Path(d)
                receipts = self.fixture(path)
                item = receipts[("meshio", "1.0.0")]["urls"][0]
                if change == "digest":
                    item["digests"]["sha256"] = "0" * 64
                elif change == "yanked":
                    item["yanked"] = True
                elif change == "metadata":
                    receipts[("meshio", "1.0.0")]["info"]["name"] = "other"
                elif change == "version":
                    file = next(path.glob("fenics_basix-*.whl"))
                    file.rename(path / file.name.replace("0.11.0", "0.10.0"))
                elif change == "origin":
                    item["url"] = "http://untrusted.invalid/wheel"
                else:
                    item["size"] += 1
                with self.assertRaises((ValueError, KeyError)):
                    self.module().validate_wheels(path, receipts)

    def test_foreign_tags_and_collection_budget_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            receipts = self.fixture(path)
            file = next(path.glob("meshio-*.whl"))
            file.rename(
                path / file.name.replace("py3-none-any", "cp299-cp299-win_amd64")
            )
            with self.assertRaises(ValueError):
                self.module().validate_wheels(path, receipts)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            receipts = self.fixture(path)
            with self.assertRaises(ValueError):
                self.module().validate_wheels(path, receipts, max_bytes=1)

    def test_failed_download_does_not_become_a_successful_manifest(self):
        import subprocess

        module = self.module()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "collection"
            with (
                mock.patch.object(module.sys, "version_info", (3, 13)),
                mock.patch.object(
                    module.platform, "python_implementation", return_value="CPython"
                ),
                mock.patch.object(module.platform, "system", return_value="Linux"),
                mock.patch.object(module.platform, "machine", return_value="x86_64"),
                mock.patch.object(
                    module.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess(
                        [], 1, "", "download refused"
                    ),
                ),
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    module.collect(path)
            self.assertIn("download refused", (path / "status.txt").read_text())
            self.assertFalse((path / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
