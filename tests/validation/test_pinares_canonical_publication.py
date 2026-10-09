"""Canonical Pinares publication must preflight every packaged mirror."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

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
    def test_each_mirror_refused_before_solve_and_any_checkout_write(self) -> None:
        for index, relative in enumerate(RELATIVE_FILES):
            for kind in (
                "symlink-file",
                "symlink-parent",
                "hardlink",
                "directory",
                "blocked-parent",
            ):
                with (
                    self.subTest(mirror=index, kind=kind),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    root = Path(directory) / "checkout"
                    resource_root = (
                        root
                        / "src/finite_element_options/validation/evidence/reference_data"
                    )
                    resource_root.mkdir(parents=True)
                    target = resource_root / relative
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
                        root
                        / "tests/fixtures"
                        / relative: f"original {index}\n".encode()
                        for index, relative in enumerate(RELATIVE_FILES)
                    }
                    for path, data in checkout.items():
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(data)
                    original_entries = set(resource_root.rglob("*"))
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
                    self.assertEqual(set(resource_root.rglob("*")), original_entries)
                    if kind == "symlink-parent":
                        self.assertEqual((outside / target.name).read_bytes(), original)

    def test_trusted_root_alias_retains_descendant_symlink_refusal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkout"
            root.mkdir()
            alias = Path(directory) / "trusted-alias"
            alias.symlink_to(root, target_is_directory=True)
            outside = Path(directory) / "outside"
            outside.mkdir()
            victim = outside / "unrelated.py"
            original = b"caller-owned unrelated source bytes\n"
            victim.write_bytes(original)
            module = maintainer()
            resource_root = (
                root / "src/finite_element_options/validation/evidence/reference_data"
            )
            self.assertEqual(
                module._canonical_mirror_destinations(alias),
                tuple(resource_root / relative for relative in RELATIVE_FILES),
            )
            self.assertFalse((root / "src").exists())
            (root / "src").symlink_to(outside, target_is_directory=True)
            with (
                patch.object(module, "REPO_ROOT", alias),
                patch.object(
                    module,
                    "run_public_pinares_fixed_price_proxy_fixture",
                    side_effect=AssertionError(
                        "descendant preflight did not precede solve"
                    ),
                ) as solve,
                self.assertRaisesRegex(ValueError, "mirror"),
            ):
                module.main(["--publish-canonical"])
            solve.assert_not_called()
            self.assertEqual(victim.read_bytes(), original)
            self.assertFalse((root / "tests").exists())


if __name__ == "__main__":
    unittest.main()
