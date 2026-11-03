"""Validated native plot and bounded archive artifact controls."""

import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from native_pde_test_support import NativeEvidenceFixtures


class NativeArtifactTests(NativeEvidenceFixtures, unittest.TestCase):
    """Artifact fixtures test admission and rendering, not native execution."""

    def test_export_preserves_validated_fields_and_renders_both_plots(self):
        import numpy as np
        import matplotlib.image as mpimg

        for backend, png, shape in [
            ("fenicsx", "fenicsx_poisson.png", (750, 900)),
            ("dedalus", "dedalus_heat.png", (600, 1050)),
        ]:
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as d:
                path = self.measurement(
                    Path(d) / "data",
                    phase="original",
                    prefix="/native/source",
                    backend=backend,
                )
                result = self.cli("export", str(path))
                self.assertEqual(result.returncode, 0, result.stderr)
                with np.load(
                    path.parent / "values.npz", allow_pickle=False
                ) as original:
                    with np.load(
                        path.parent / f"{backend}_result.npz", allow_pickle=False
                    ) as exported:
                        for name in ("x", "y", "u", "exact"):
                            np.testing.assert_array_equal(
                                exported[name], original[name]
                            )
                        self.assertEqual(float(exported["rms"]), 0.0)
                pixels = mpimg.imread(path.parent / png)
                self.assertEqual(pixels.shape[:2], shape)
                self.assertGreater(float(np.std(pixels)), 0.05)
                manifest = json.loads(
                    (path.parent / "export-manifest.json").read_text()
                )
                self.assertEqual(
                    manifest["data_sha256"], json.loads(path.read_text())["data_sha256"]
                )
                self.assertEqual(
                    manifest["source_sha256"], self.report()["source_sha256"]
                )
                for item in manifest["exports"]:
                    self.assertEqual(
                        hashlib.sha256(
                            (path.parent / item["name"]).read_bytes()
                        ).hexdigest(),
                        item["sha256"],
                    )

    def test_export_refuses_rehashed_bad_fields_before_writing_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.measurement(
                Path(d) / "data",
                phase="original",
                prefix="/native/source",
                perturbation=0.1,
            )
            result = self.cli("export", str(path))
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertFalse((path.parent / "fenicsx_poisson.png").exists())
            self.assertFalse((path.parent / "fenicsx_result.npz").exists())

    def test_archive_measurement_refuses_oversize_and_binds_runtime_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.measurement(
                root / "original", phase="original", prefix="/native/source"
            )
            (root / "source.txt").write_text("a" * 40 + "\n" + "b" * 40 + "\n")
            (root / "conda-explicit.txt").write_text("@EXPLICIT\n# fixture only\n")
            archive = root / "native-runtime.tar.xz"
            fixture = root / "fixture.txt"
            fixture.write_text("small archive fixture; not a native runtime")
            with tarfile.open(archive, "w:xz", preset=6) as packed:
                packed.add(fixture, arcname="fixture.txt")
            with self.assertRaises(ValueError):
                self.module().archive_manifest(archive, root, max_bytes=1)
            self.assertFalse((root / "archive-manifest.json").exists())
            self.module().archive_manifest(archive, root)
            report = json.loads((root / "archive-manifest.json").read_text())
            self.assertEqual(report["bytes"], archive.stat().st_size)
            self.assertEqual(
                report["sha256"], hashlib.sha256(archive.read_bytes()).hexdigest()
            )
            self.assertEqual(report["source_tree"], "b" * 40)
            self.assertEqual(report["source_sha256"], self.report()["source_sha256"])
            self.assertEqual(
                dict(
                    line.split()[::-1]
                    for line in (root / "SOURCE_SHA256SUMS").read_text().splitlines()
                ),
                report["source_sha256"],
            )
            self.assertIn("format", report)
            self.assertEqual(report["format"], "tar.xz")
            self.assertEqual(report["max_bytes"], 524288000)
            self.assertEqual(report["compression"], {"level": 6, "threads": 2})
            readme = (root / "README-runtime.md").read_text()
            self.assertIn("tar -xJf native-runtime.tar.xz -C runtime", readme)
            self.assertIn("matplotlib-base3.11.2", readme)
            self.assertIn(" export ", readme)
            self.assertIn("scripts/native_pde_artifacts.py", readme)
            self.assertIn("source_sha256", readme)
            self.assertNotIn("native-runtime.tar.gz", readme)

    def test_archive_default_cap_rejects_500_mib_plus_one_without_hashing(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.measurement(
                root / "original", phase="original", prefix="/native/source"
            )
            (root / "source.txt").write_text("a" * 40 + "\n" + "b" * 40 + "\n")
            (root / "conda-explicit.txt").write_text("@EXPLICIT\n# fixture\n")
            archive = root / "native-runtime.tar.xz"
            with archive.open("wb") as stream:
                stream.truncate(524288001)
            for options in ({}, {"max_bytes": 1073741824}):
                with self.subTest(options=options):
                    with self.assertRaisesRegex(ValueError, "500MiB"):
                        self.module().archive_manifest(archive, root, **options)
            self.assertFalse((root / "archive-manifest.json").exists())

    def test_archive_refuses_wrong_format_and_mislabeled_suffix(self):
        for mode, name in [
            ("w:gz", "native-runtime.tar.xz"),
            ("w:xz", "native-runtime.tar.gz"),
        ]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                self.measurement(
                    root / "original", phase="original", prefix="/native/source"
                )
                (root / "source.txt").write_text("a" * 40 + "\n" + "b" * 40 + "\n")
                (root / "conda-explicit.txt").write_text("@EXPLICIT\n# fixture\n")
                archive = root / name
                fixture = root / "fixture.txt"
                fixture.write_text("format fixture")
                with tarfile.open(archive, mode) as packed:
                    packed.add(fixture, arcname="fixture.txt")
                with self.assertRaises(ValueError):
                    self.module().archive_manifest(archive, root)
                self.assertFalse((root / "archive-manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
