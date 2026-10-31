"""Exact compiler-version admission controls, also runnable from a normal wheel."""

from __future__ import annotations

import hashlib
import json
import unittest
from unittest.mock import patch

from finite_element_options.validation import compiled_weak_form_adapter as adapter
from finite_element_options.validation.compiled_weak_form_golden import (
    packaged_golden_fixture,
)

V0 = "pde_ir_symbolic_compiler.v0"
V1 = "pde_ir_symbolic_compiler.v1"
V0_HASH = "sha256:970088e5dcb16535edfd230bfe992ea7eb68aede901c7b543682b39f1a5ac32e"
V1_HASH = "sha256:b449647e7f8deea870b0e8fbe0cfd4355040a53b9853d69c17443f0b3a6d9cb2"


class CompilerVersionTests(unittest.TestCase):
    """Keep current admission bounded to two independently registered records."""

    def test_original_default_is_unchanged(self) -> None:
        fixture = packaged_golden_fixture()
        self.assertEqual(fixture["compiled_operator"]["compiled_hash"], V0_HASH)
        self.assertTrue(adapter.screen_compiled_weak_form(fixture).accepted)

    def test_registered_versions_screen_and_solve(self) -> None:
        for version, identity in ((V0, V0_HASH), (V1, V1_HASH)):
            with self.subTest(version=version):
                fixture = packaged_golden_fixture(compiler_version=version)
                self.assertTrue(adapter.screen_compiled_weak_form(fixture).accepted)
                result = adapter.solve_compiled_weak_form(fixture)
                self.assertEqual(result["compiled_operator_hash"], identity)
                self.assertEqual(result["status"], "converged")
                for metric, limit in (
                    ("price_absolute_error", 0.002),
                    ("delta_absolute_error", 0.001),
                    ("gamma_absolute_error", 2e-5),
                ):
                    self.assertLessEqual(result["summary"][metric], limit)

    def test_unknown_factory_selection_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "compiler version"):
            packaged_golden_fixture(compiler_version="pde_ir_symbolic_compiler.v999")

    def test_mutations_cannot_change_authority_or_start_numerics(self) -> None:
        for version in (V0, V1):
            for mutation in (
                "unknown_version",
                "mixed_pair",
                "expression",
                "unit",
                "route",
                "private",
                "unknown",
                "boundary",
                "source",
                "rehashed_expression",
                "rehashed_unit",
            ):
                with self.subTest(version=version, mutation=mutation):
                    original = packaged_golden_fixture(compiler_version=version)
                    changed = packaged_golden_fixture(compiler_version=version)
                    compiled = changed["compiled_operator"]
                    if mutation == "unknown_version":
                        compiled["compiler_evidence"]["compiler_version"] = "unknown"
                    elif mutation == "mixed_pair":
                        compiled["compiled_hash"] = (
                            V1_HASH if version == V0 else V0_HASH
                        )
                    elif mutation in ("expression", "rehashed_expression"):
                        compiled["expressions"][0]["normalized"] = "0"
                    elif mutation in ("unit", "rehashed_unit"):
                        compiled["expressions"][0]["declared_result_unit"] = {
                            "currency": "EUR"
                        }
                    elif mutation == "route":
                        changed["fem_route"]["parameters"]["rate"] = 0.15
                    elif mutation == "private":
                        changed["privacy_class"] = "private"
                    elif mutation == "unknown":
                        compiled["unknown"] = True
                    elif mutation == "boundary":
                        changed["fem_route"]["boundary_map"]["upper"]["enforced_as"] = (
                            "natural"
                        )
                    else:
                        changed["pde_ir"]["operator"]["terms"][0]["expression"] = "0"
                    if mutation.startswith("rehashed_"):
                        unhashed = {
                            key: value
                            for key, value in compiled.items()
                            if key != "compiled_hash"
                        }
                        encoded = json.dumps(
                            unhashed,
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        ).encode()
                        compiled["compiled_hash"] = (
                            "sha256:" + hashlib.sha256(encoded).hexdigest()
                        )
                    self.assertFalse(
                        adapter.screen_compiled_weak_form(changed).accepted
                    )
                    with patch.object(
                        adapter, "run_public_black_scholes_parity_fixture"
                    ) as numerical:
                        with self.assertRaises(
                            adapter.CompiledWeakFormUnsupportedError
                        ):
                            adapter.solve_compiled_weak_form(changed)
                        numerical.assert_not_called()
                    self.assertEqual(
                        packaged_golden_fixture(compiler_version=version), original
                    )
                    self.assertTrue(
                        adapter.screen_compiled_weak_form(original).accepted
                    )


if __name__ == "__main__":
    unittest.main()
