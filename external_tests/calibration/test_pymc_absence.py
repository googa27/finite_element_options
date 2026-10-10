"""Exercise actual optional-dependency absence in a normal installed wheel."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import unittest
from pathlib import Path

import finite_element_options.estimation as estimation


class PyMCAbsenceTests(unittest.TestCase):
    """The legacy facade must refuse before importing its optional owner."""

    profile = "core"

    def test_actual_profile_and_installed_origin(self) -> None:
        self.assertTrue(sys.flags.isolated)
        self.assertIsNone(importlib.util.find_spec("pymc"))
        self.assertEqual(
            importlib.util.find_spec("pandas") is not None,
            self.profile == "calibration",
        )
        self.assertTrue(
            Path(estimation.__file__).resolve().is_relative_to(Path(sys.prefix))
        )

    def test_legacy_exports_name_extra_without_loading_heston(self) -> None:
        self.assertNotIn("finite_element_options.estimation.heston", sys.modules)
        for name in ("PyMCCalibrator", "sample_pymc_calibration"):
            with self.subTest(export=name):
                with self.assertRaisesRegex(
                    ModuleNotFoundError,
                    r"finite-element-options\[calibration,bayesian\]",
                ) as failure:
                    getattr(estimation, name)
                self.assertIn("pymc", str(failure.exception))
                self.assertNotIn(
                    "finite_element_options.estimation.heston", sys.modules
                )
                self.assertNotIn("pymc", sys.modules)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("core", "calibration"), required=True)
    args = parser.parse_args()
    PyMCAbsenceTests.profile = args.profile
    unittest.main(argv=[sys.argv[0]], verbosity=2)
