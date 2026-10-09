"""Caller-owned decoded fixture data must not change the screening authority."""

from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

from finite_element_options.validation import compiled_weak_form_adapter as adapter
from finite_element_options.validation.compiled_weak_form_golden import (
    packaged_golden_fixture,
)


class PublicGoldenOwnership(unittest.TestCase):
    def setUp(self) -> None:
        # Reset only the legacy buggy cache so original-run cases stay independent.
        clear_legacy_cache = getattr(packaged_golden_fixture, "cache_clear", None)
        if clear_legacy_cache is not None:
            clear_legacy_cache()

    def tearDown(self) -> None:
        self.setUp()

    def test_repeated_factory_calls_have_equal_content_and_distinct_ownership(
        self,
    ) -> None:
        first = packaged_golden_fixture()
        second = packaged_golden_fixture()
        self.assertEqual(first, second)
        self.assertIsNot(first, second)
        self.assertIsNot(first["fem_route"], second["fem_route"])
        self.assertIsNot(
            first["compiled_operator"]["expressions"],
            second["compiled_operator"]["expressions"],
        )

    def test_caller_rate_mutation_cannot_replace_screening_authority(self) -> None:
        original = copy.deepcopy(packaged_golden_fixture())
        caller = packaged_golden_fixture()
        caller["fem_route"]["parameters"]["rate"] = 0.15
        observed = adapter.screen_compiled_weak_form(caller)
        self.assertFalse(observed.accepted)
        self.assertIn(
            "compiled_weak_form.route_exact", [d.code for d in observed.diagnostics]
        )
        self.assertTrue(adapter.screen_compiled_weak_form(original).accepted)
        self.assertEqual(packaged_golden_fixture(), original)
        with patch.object(
            adapter, "run_public_black_scholes_parity_fixture"
        ) as assembly:
            with self.assertRaises(adapter.CompiledWeakFormUnsupportedError):
                adapter.solve_compiled_weak_form(caller)
            assembly.assert_not_called()

    def test_caller_nested_expression_mutation_cannot_replace_expected_subobject(
        self,
    ) -> None:
        original = copy.deepcopy(packaged_golden_fixture())
        caller = packaged_golden_fixture()
        caller["compiled_operator"]["expressions"][0]["declared_result_unit"] = {
            "dimension": "altered"
        }
        observed = adapter.screen_compiled_weak_form(caller)
        self.assertFalse(observed.accepted)
        self.assertIn(
            "compiled_weak_form.compiled_operator_exact",
            [d.code for d in observed.diagnostics],
        )
        self.assertTrue(adapter.screen_compiled_weak_form(original).accepted)
        self.assertEqual(packaged_golden_fixture(), original)

    def test_caller_unknown_nested_field_does_not_extend_authority_schema(self) -> None:
        original = copy.deepcopy(packaged_golden_fixture())
        caller = packaged_golden_fixture()
        caller["fem_route"]["parameters"]["unregistered_parameter"] = 7
        observed = adapter.screen_compiled_weak_form(caller)
        self.assertFalse(observed.accepted)
        self.assertIn(
            "compiled_weak_form.unknown_field", [d.code for d in observed.diagnostics]
        )
        self.assertTrue(adapter.screen_compiled_weak_form(original).accepted)
        self.assertEqual(packaged_golden_fixture(), original)

    def test_independent_copy_negative_and_untouched_original_positive(self) -> None:
        original = copy.deepcopy(packaged_golden_fixture())
        changed = copy.deepcopy(original)
        changed["fem_route"]["parameters"]["rate"] = 0.15
        self.assertFalse(adapter.screen_compiled_weak_form(changed).accepted)
        self.assertTrue(adapter.screen_compiled_weak_form(original).accepted)


if __name__ == "__main__":
    unittest.main()
