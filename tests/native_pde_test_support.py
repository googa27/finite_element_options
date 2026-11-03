"""Synthetic native report/array fixtures, never native execution receipts."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validate_native_pde.py"
ARTIFACT_HELPER = SCRIPT.with_name("native_pde_artifacts.py")


class NativeEvidenceFixtures:
    """Share exact-source fixtures without duplicating discovered test cases."""

    def cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        """Execute the actual public script, including its argument parser."""
        return subprocess.run(
            [sys.executable, "-I", "-B", str(SCRIPT), *args],
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
            "source_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (SCRIPT, ARTIFACT_HELPER)
            },
            "runtime": {backend: "0.11.0" if backend == "fenicsx" else "3.0.5"},
            "origins": {
                backend: prefix + "/lib/native/__init__.py",
                "numpy": prefix + "/lib/numpy/__init__.py",
            },
            "levels": levels,
        }

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
