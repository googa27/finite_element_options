"""Canonical Pinares publication must preflight every packaged mirror."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from finite_element_options.validation import pinares_fixed_price_proxy as proxy

RELATIVE_FILES = (
    "fem_pinares_fixed_price_proxy_v1/problem_spec.json",
    "fem_pinares_fixed_price_proxy_v1/result_export.json",
    "fem_pinares_fixed_price_proxy_v1/provider_evidence_manifest.json",
    "fem_pinares_fixed_price_proxy_v1/unsupported_full_deal_problem_spec.json",
    "quant_problem_specs/pinares_fixed_price_proxy.json",
)


def maintainer():
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts/export_pinares_fixed_price_proxy_fixture.py"
    )
    spec = importlib.util.spec_from_file_location("pinares_publication_guard", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CanonicalPublicationOwnership(unittest.TestCase):
    def test_last_mirror_refused_before_solve_and_any_checkout_write(self) -> None:
        for kind in (
            "symlink-file",
            "symlink-parent",
            "hardlink",
            "directory",
            "blocked-parent",
        ):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "checkout"
                resource_root = (
                    root
                    / "src/finite_element_options/validation/evidence/reference_data"
                )
                resource_root.mkdir(parents=True)
                target = resource_root / RELATIVE_FILES[-1]
                victim = Path(directory) / "unrelated.py"
                original = b"caller-owned unrelated source bytes\n"
                victim.write_bytes(original)
                if kind == "symlink-parent":
                    outside = Path(directory) / "outside"
                    outside.mkdir()
                    (outside / target.name).write_bytes(original)
                    target.parent.symlink_to(outside, target_is_directory=True)
                elif kind == "blocked-parent":
                    target.parent.write_bytes(original)
                else:
                    target.parent.mkdir()
                    if kind == "symlink-file":
                        target.symlink_to(victim)
                    elif kind == "hardlink":
                        os.link(victim, target)
                    else:
                        target.mkdir()
                checkout = {
                    root / "tests/fixtures" / relative: f"original {index}\n".encode()
                    for index, relative in enumerate(RELATIVE_FILES)
                }
                for path, data in checkout.items():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                module = maintainer()
                with (
                    patch.object(module, "REPO_ROOT", root),
                    patch.object(
                        module,
                        "run_public_pinares_fixed_price_proxy_fixture",
                        side_effect=AssertionError(
                            "mirror preflight did not precede solve"
                        ),
                    ) as solve,
                    self.assertRaisesRegex(ValueError, "mirror"),
                ):
                    module.main(["--publish-canonical"])
                solve.assert_not_called()
                self.assertEqual(victim.read_bytes(), original)
                self.assertEqual(
                    {path: path.read_bytes() for path in checkout}, checkout
                )
                self.assertFalse(
                    (resource_root / "fem_pinares_fixed_price_proxy_v1").exists()
                )
                if kind == "symlink-parent":
                    self.assertEqual((outside / target.name).read_bytes(), original)

    def test_existing_fixture_root_remains_a_public_traversable(self) -> None:
        root = proxy.PINARES_FEM_PROXY_FIXTURE_ROOT
        self.assertTrue(root.is_dir())
        for name, reference in (
            ("problem_spec.json", proxy.PINARES_FEM_PROXY_PROBLEM_SPEC_PATH),
            ("result_export.json", proxy.PINARES_FEM_PROXY_RESULT_EXPORT_PATH),
            (
                "provider_evidence_manifest.json",
                proxy.PINARES_FEM_PROVIDER_EVIDENCE_MANIFEST_PATH,
            ),
            (
                "unsupported_full_deal_problem_spec.json",
                proxy.PINARES_FEM_PROXY_UNSUPPORTED_SPEC_PATH,
            ),
        ):
            with self.subTest(name=name):
                self.assertEqual(
                    root.joinpath(name).read_bytes(), reference.read_bytes()
                )


if __name__ == "__main__":
    unittest.main()
