"""Behavioral controls for the optional native demonstration evidence boundary.

Literal reports below test admission only; they are never native execution proof.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


from native_pde_test_support import NativeEvidenceFixtures


class NativeValidationTests(NativeEvidenceFixtures, unittest.TestCase):
    """Reject scientific and relocation failures before declaring acceptance."""

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
