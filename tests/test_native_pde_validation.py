"""Behavioral controls for the optional native demonstration evidence boundary.

Literal reports below test admission only; they are never native execution proof.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validate_native_pde.py"


class NativeValidationTests(unittest.TestCase):
    """Reject scientific and relocation failures before declaring acceptance."""

    def cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        """Execute the actual public script, including its argument parser."""
        return subprocess.run(
            [sys.executable, "-B", str(SCRIPT), *args],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )

    def module(self):
        """Load the actual script without importing optional native packages."""
        self.assertTrue(SCRIPT.is_file(), "native validation CLI is missing")
        spec = importlib.util.spec_from_file_location("native_validation", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def report(self, backend="fenicsx", phase="original", prefix="/native/source"):
        """Return hand-derived admission data, never a numerical run receipt."""
        levels = [
            {"resolution": 16, "scale": 0.0625, "error_l2": 0.004},
            {"resolution": 32, "scale": 0.03125, "error_l2": 0.001},
            {"resolution": 64, "scale": 0.015625, "error_l2": 0.00025},
        ]
        for row in levels:
            row.update(ksp_reason=4, iterations=1, boundary_max=0.0)
        if backend == "dedalus":
            levels = [
                {
                    "resolution": n,
                    "scale": dt,
                    "error_l2": err,
                    "iterations": n,
                    "final_time": 0.5,
                    "mean": 0.0,
                    "energy_ratio": 0.9,
                }
                for n, dt, err in [
                    (10, 0.05, 8e-6),
                    (20, 0.025, 2e-6),
                    (40, 0.0125, 5e-7),
                ]
            ]
        return {
            "schema": 1,
            "backend": backend,
            "phase": phase,
            "prefix": prefix,
            "script_sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
            "runtime": {backend: "0.11.0" if backend == "fenicsx" else "3.0.5"},
            "origins": {
                backend: prefix + "/lib/native/__init__.py",
                "numpy": prefix + "/lib/numpy/__init__.py",
            },
            "levels": levels,
        }

    def test_help_without_optional_native_packages(self):
        result = self.cli("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("solve", result.stdout)

    def test_unknown_backend_is_rejected_before_execution(self):
        result = self.cli("solve", "--backend", "invented")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid choice", result.stderr)

    def test_literal_quadratic_error_refinement_is_admitted(self):
        for backend in ["fenicsx", "dedalus"]:
            self.module().validate_report(self.report(backend))

    def test_nonfinite_and_nonconverging_measurements_are_refused(self):
        module = self.module()
        for bad in [float("nan"), float("inf"), 0.0, -1.0, 0.005]:
            report = self.report()
            report["levels"][-1]["error_l2"] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                module.validate_report(report)

    def test_missing_or_duplicate_refinement_levels_are_refused(self):
        module = self.module()
        for rows in [self.report()["levels"][:2], [self.report()["levels"][0]] * 3]:
            report = self.report()
            report["levels"] = rows
            with self.assertRaises(ValueError):
                module.validate_report(report)

    def test_boolean_schema_or_overflowed_refinement_order_is_refused(self):
        module = self.module()
        report = self.report()
        report["schema"] = True
        with self.assertRaises(ValueError):
            module.validate_report(report)
        report = self.report()
        report["levels"][1]["error_l2"] = 1e-320
        report["levels"][2]["error_l2"] = 1e-321
        with self.assertRaises(ValueError):
            module.validate_report(report)

    def test_rehashed_incomplete_measurement_arrays_are_refused(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            report = self.report()
            data = Path(directory) / "values.npz"
            np.savez_compressed(data, x=[0.0], y=[0.0], u=[0.0], exact=[0.0])
            report["data_sha256"] = hashlib.sha256(data.read_bytes()).hexdigest()
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(report))
            result = self.cli("validate", str(path))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("grid", result.stderr)

    def measurement(
        self,
        directory,
        *,
        phase,
        prefix,
        perturbation=0.0,
        reverse=False,
        backend="fenicsx",
        coordinate_offset=(0.0, 0.0),
        exact_bias=0.0,
        ulp_coordinates=False,
    ):
        """Write complete synthetic arrays to exercise real artifact comparison."""
        import numpy as np

        directory.mkdir()
        if backend == "fenicsx":
            x, y = np.meshgrid(np.linspace(0, 1, 65), np.linspace(0, 1, 65))
            x, y = x.ravel() + coordinate_offset[0], y.ravel() + coordinate_offset[1]
            exact = np.sin(np.pi * x) * np.sin(np.pi * y)
            values = exact + perturbation * np.sin(2 * np.pi * x) * np.sin(np.pi * y)
        else:
            x = np.arange(64) * np.pi / 32
            y = np.zeros_like(x)
            exact = np.exp(-0.05) * np.sin(x) + 0.25 * np.exp(-0.2) * np.cos(2 * x)
            values = exact + perturbation * np.sin(2 * x)
        arrays = {"x": x, "y": y, "exact": exact + exact_bias, "u": values}
        if ulp_coordinates:
            for name in ("x", "y"):
                arrays[name][::2] = np.nextafter(arrays[name][::2], np.inf)
                arrays[name][1::2] = np.nextafter(arrays[name][1::2], -np.inf)
        if reverse:
            arrays = {name: value[::-1] for name, value in arrays.items()}
        data = directory / "values.npz"
        np.savez_compressed(data, **arrays)
        report = self.report(backend, phase, prefix)
        report["data_sha256"] = hashlib.sha256(data.read_bytes()).hexdigest()
        path = directory / "report.json"
        path.write_text(json.dumps(report))
        return path

    def test_complete_rehashed_altered_field_is_not_a_relocated_replay(self):
        for backend, perturbation in [("fenicsx", 0.0005), ("dedalus", 1e-6)]:
            with (
                self.subTest(backend=backend),
                tempfile.TemporaryDirectory() as directory,
            ):
                original = self.measurement(
                    Path(directory) / "original",
                    phase="original",
                    prefix="/native/source",
                    backend=backend,
                )
                changed = self.measurement(
                    Path(directory) / "changed",
                    phase="relocated",
                    prefix="/native/consumer",
                    backend=backend,
                    perturbation=perturbation,
                )
                self.assertEqual(self.cli("validate", str(original)).returncode, 0)
                self.assertEqual(self.cli("validate", str(changed)).returncode, 0)
                result = self.cli("compare", str(original), str(changed))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("replay field differs: u", result.stderr)

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
            self.assertIn("format", report)
            self.assertEqual(report["format"], "tar.xz")
            self.assertEqual(report["max_bytes"], 524288000)
            self.assertEqual(report["compression"], {"level": 6, "threads": 2})
            readme = (root / "README-runtime.md").read_text()
            self.assertIn("tar -xJf native-runtime.tar.xz -C runtime", readme)
            self.assertIn("matplotlib-base3.11.2", readme)
            self.assertIn(" export ", readme)
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

    def test_complete_replay_is_coordinate_aligned_for_both_backends(self):
        for backend in ["fenicsx", "dedalus"]:
            with (
                self.subTest(backend=backend),
                tempfile.TemporaryDirectory() as directory,
            ):
                original = self.measurement(
                    Path(directory) / "original",
                    phase="original",
                    prefix="/native/source",
                    backend=backend,
                )
                relocated = self.measurement(
                    Path(directory) / "relocated",
                    phase="relocated",
                    prefix="/native/consumer",
                    backend=backend,
                    reverse=True,
                    perturbation=1e-13,
                )
                result = self.cli("compare", str(original), str(relocated))
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_individually_admitted_coordinate_drift_is_not_a_matching_replay(self):
        for axis in (0, 1):
            with self.subTest(axis=axis), tempfile.TemporaryDirectory() as directory:
                offsets = [(0.0, 0.0), (0.0, 0.0)]
                offsets[0] = tuple(-0.75e-12 if i == axis else 0.0 for i in (0, 1))
                offsets[1] = tuple(0.75e-12 if i == axis else 0.0 for i in (0, 1))
                original = self.measurement(
                    Path(directory) / "original",
                    phase="original",
                    prefix="/native/source",
                    coordinate_offset=offsets[0],
                )
                relocated = self.measurement(
                    Path(directory) / "relocated",
                    phase="relocated",
                    prefix="/native/consumer",
                    coordinate_offset=offsets[1],
                )
                self.assertEqual(self.cli("validate", str(original)).returncode, 0)
                self.assertEqual(self.cli("validate", str(relocated)).returncode, 0)
                result = self.cli("compare", str(original), str(relocated))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    "replay field differs: " + ("x" if axis == 0 else "y"),
                    result.stderr,
                )

    def test_individually_admitted_exact_arrays_must_match_on_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            original = self.measurement(
                Path(directory) / "original",
                phase="original",
                prefix="/native/source",
                exact_bias=-0.75e-12,
            )
            relocated = self.measurement(
                Path(directory) / "relocated",
                phase="relocated",
                prefix="/native/consumer",
                exact_bias=0.75e-12,
            )
            self.assertEqual(self.cli("validate", str(original)).returncode, 0)
            self.assertEqual(self.cli("validate", str(relocated)).returncode, 0)
            result = self.cli("compare", str(original), str(relocated))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("replay field differs: exact", result.stderr)

    def test_metadata_only_pair_cannot_establish_replay(self):
        with self.assertRaises(ValueError):
            self.module().validate_pair(
                self.report(), self.report(phase="relocated", prefix="/native/consumer")
            )

    def test_complete_nextafter_grid_preserves_node_identity_and_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            original = self.measurement(
                Path(directory) / "original", phase="original", prefix="/native/source"
            )
            rounded = self.measurement(
                Path(directory) / "rounded",
                phase="relocated",
                prefix="/native/consumer",
                ulp_coordinates=True,
                reverse=True,
            )
            result = self.cli("validate", str(rounded))
            self.assertEqual(result.returncode, 0, result.stderr)
            replay = self.cli("compare", str(original), str(rounded))
            self.assertEqual(replay.returncode, 0, replay.stderr)

    def test_lattice_admission_rejects_duplicates_missing_nodes_and_off_grid(self):
        import numpy as np

        for mode, diagnostic in [
            ("duplicate", "duplicate"),
            ("missing", "node count"),
            ("off-grid", "off-lattice"),
        ]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                path = self.measurement(
                    Path(directory) / "data", phase="original", prefix="/native/source"
                )
                data = path.parent / "values.npz"
                with np.load(data, allow_pickle=False) as archive:
                    arrays = {name: archive[name] for name in ("x", "y", "exact", "u")}
                if mode == "duplicate":
                    for name in arrays:
                        arrays[name][0] = arrays[name][1]
                    arrays["x"][0] = np.nextafter(arrays["x"][0], np.inf)
                elif mode == "missing":
                    arrays = {name: values[:-1] for name, values in arrays.items()}
                else:
                    arrays["x"][1] += 1e-8
                np.savez_compressed(data, **arrays)
                report = json.loads(path.read_text())
                report["data_sha256"] = hashlib.sha256(data.read_bytes()).hexdigest()
                path.write_text(json.dumps(report))
                result = self.cli("validate", str(path))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(diagnostic, result.stderr)

    def test_failed_ksp_or_nonzero_boundary_is_refused(self):
        module = self.module()
        for field, value in [
            ("ksp_reason", -3),
            ("boundary_max", 0.2),
            ("iterations", 0),
            ("ksp_reason", True),
        ]:
            report = self.report()
            report["levels"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                module.validate_report(report)

    def test_heat_wrong_time_growth_or_mean_is_refused(self):
        module = self.module()
        for field, value in [
            ("final_time", 0.51),
            ("energy_ratio", 1.01),
            ("mean", 0.01),
            ("iterations", 9),
        ]:
            report = self.report("dedalus")
            report["levels"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                module.validate_report(report)

    def test_changed_script_runtime_or_ambient_origin_is_refused(self):
        module = self.module()
        for field, value in [
            ("script_sha256", "b" * 64),
            ("runtime", {"fenicsx": "0.9.0"}),
            ("origins", {"fenicsx": "/ambient/solver.py"}),
        ]:
            report = self.report()
            report[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                module.validate_report(report)

    def test_relocated_pair_requires_different_prefix_and_phase(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as directory:
            original = module.read_report(
                self.measurement(
                    Path(directory) / "original",
                    phase="original",
                    prefix="/native/source",
                )
            )
            relocated = module.read_report(
                self.measurement(
                    Path(directory) / "relocated",
                    phase="relocated",
                    prefix="/native/consumer",
                )
            )
            module.validate_pair(original, relocated)
            bad_phase = copy.deepcopy(relocated)
            bad_phase["phase"] = "original"
            with self.assertRaisesRegex(ValueError, "phases"):
                module.validate_pair(original, bad_phase)
            bad_prefix = copy.deepcopy(relocated)
            bad_prefix["prefix"] = original["prefix"]
            bad_prefix["origins"] = original["origins"]
            with self.assertRaisesRegex(ValueError, "different absolute prefix"):
                module.validate_pair(original, bad_prefix)

    def test_relocated_pair_rejects_a_changed_result_or_backend(self):
        module = self.module()
        original = self.report()
        relocated = self.report(phase="relocated", prefix="/native/consumer")
        relocated["levels"][-1]["error_l2"] *= 1.01
        with self.assertRaises(ValueError):
            module.validate_pair(original, relocated)
        with self.assertRaises(ValueError):
            module.validate_pair(
                original, self.report("dedalus", "relocated", "/native/consumer")
            )

    def test_cli_refuses_untrusted_failed_report(self):
        report = self.report()
        report["levels"][0]["ksp_reason"] = -3
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(report))
            result = self.cli("validate", str(path))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("KSP", result.stderr)

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "untouched"
            marker.write_text("original")
            result = self.cli(
                "solve",
                "--backend",
                "fenicsx",
                "--output",
                directory,
                "--phase",
                "original",
                "--expected-prefix",
                sys.prefix,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("already exists", result.stderr)
            self.assertEqual(marker.read_text(), "original")


if __name__ == "__main__":
    unittest.main()
