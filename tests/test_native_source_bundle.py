"""Fitness and isolated replay controls for the copied native source bundle."""

from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from native_pde_test_support import (
    ARTIFACT_HELPER,
    NativeEvidenceFixtures,
    SCRIPT,
)


ROOT = Path(__file__).resolve().parents[1]


class NativeSourceBundleTests(NativeEvidenceFixtures, unittest.TestCase):
    def test_native_modules_obey_the_unwaived_500_line_limit(self):
        paths = {ROOT / "scripts/validate_native_pde.py"}
        paths.update((ROOT / "scripts").glob("native_pde*.py"))
        paths.update((ROOT / "tests").glob("test_native*.py"))
        paths.update((ROOT / "tests").glob("native_pde*.py"))
        for path in sorted(paths):
            with self.subTest(path=path.relative_to(ROOT).as_posix()):
                self.assertLessEqual(len(path.read_text().splitlines()), 500)

    def test_isolated_copied_bundle_validates_and_exports_from_unrelated_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, caller = root / "bundle", root / "caller"
            bundle.mkdir()
            caller.mkdir()
            for source in (SCRIPT, ARTIFACT_HELPER):
                shutil.copyfile(source, bundle / source.name)
            poison = caller / ARTIFACT_HELPER.name
            poison.write_text("raise RuntimeError('ambient helper imported')\n")
            report = self.measurement(
                root / "data",
                phase="original",
                prefix="/native/source",
                backend="dedalus",
            )
            for command in ("validate", "export"):
                result = subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        "-B",
                        str(bundle / SCRIPT.name),
                        command,
                        str(report),
                    ],
                    cwd=caller,
                    env={**os.environ, "PYTHONPATH": str(caller)},
                    capture_output=True,
                    text=True,
                    timeout=20,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((report.parent / "export-manifest.json").read_text())
            self.assertEqual(manifest["source_sha256"], self.report()["source_sha256"])
            self.assertTrue((report.parent / "dedalus_heat.png").is_file())

    def test_isolated_copy_refuses_missing_or_changed_helper(self):
        for mode in ("missing", "changed"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                shutil.copyfile(SCRIPT, root / SCRIPT.name)
                if mode == "changed":
                    (root / ARTIFACT_HELPER.name).write_bytes(
                        ARTIFACT_HELPER.read_bytes() + b"\n# changed helper source\n"
                    )
                report = self.measurement(
                    root / "data", phase="original", prefix="/native/source"
                )
                result = subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        "-B",
                        str(root / SCRIPT.name),
                        "export",
                        str(report),
                    ],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
                self.assertNotEqual(result.returncode, 0)
                if mode == "changed":
                    self.assertIn("complete source bundle", result.stderr)
                self.assertFalse((report.parent / "fenicsx_poisson.png").exists())
                self.assertFalse((report.parent / "fenicsx_result.npz").exists())

    def test_report_requires_exact_driver_and_helper_hash_mapping(self):
        for mode in ("missing", "helper", "driver", "extra"):
            report = self.report()
            if mode == "missing":
                report["source_sha256"].pop(ARTIFACT_HELPER.name)
            elif mode == "extra":
                report["source_sha256"]["unreviewed.py"] = "a" * 64
            else:
                name = ARTIFACT_HELPER.name if mode == "helper" else SCRIPT.name
                report["source_sha256"][name] = "b" * 64
            with (
                self.subTest(mode=mode),
                self.assertRaisesRegex(ValueError, "complete source bundle"),
            ):
                self.module().validate_report(report)

    def test_solve_report_emits_complete_bundle_identity(self):
        """Stub only the native actor; exercise actual report emission/admission."""
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.measurement(
                root / "fixture", phase="original", prefix=sys.prefix
            )
            report = json.loads(fixture.read_text())
            with np.load(fixture.parent / "values.npz", allow_pickle=False) as data:
                arrays = {name: data[name] for name in ("x", "y", "u", "exact")}
            module = self.module()
            module._poisson = lambda: (report["levels"], arrays, report["origins"])
            module.solve("fenicsx", root / "output", "original", Path(sys.prefix))
            emitted = module.read_report(root / "output/report.json")
            self.assertEqual(emitted["source_sha256"], self.report()["source_sha256"])


if __name__ == "__main__":
    unittest.main()
