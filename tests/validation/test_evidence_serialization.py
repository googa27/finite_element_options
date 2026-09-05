"""Regression tests for shared canonical evidence serialization."""

from __future__ import annotations

import numpy as np

from finite_element_options.examples.regime_switching_quanto.adoption import evidence_io
from finite_element_options.validation.evidence.serialization import (
    canonical_json,
    canonical_json_sha256,
    json_safe,
)


def test_shared_serialization_handles_numpy_bool_and_preserves_compatibility_exports() -> None:
    """Promoting evidence I/O must preserve adoption artifact semantics."""

    payload = {"flag": np.bool_(True), "value": np.float64(1.25), "array": np.array([1, 2])}
    assert json_safe(payload) == {"flag": True, "value": 1.25, "array": [1, 2]}
    assert evidence_io.canonical_json(payload) == canonical_json(payload)
    assert evidence_io.canonical_json_sha256(payload) == canonical_json_sha256(payload)
