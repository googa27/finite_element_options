"""Regression tests for shared canonical evidence serialization."""

from __future__ import annotations

import numpy as np
import pytest

from finite_element_options.contracts import evidence_serialization as serialization_impl
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


@pytest.mark.parametrize(
    ("direct_url", "expected"),
    [
        (None, "unknown"),
        ('{"url":"file:///tmp/package.whl","archive_info":{}}', "wheel"),
        ('{"url":"file:///tmp/package.tar.gz","archive_info":{}}', "archive"),
        ('{"url":"file:///tmp/source","dir_info":{}}', "directory"),
        ('{"url":"file:///tmp/source","dir_info":{"editable":true}}', "editable"),
        ('{"url":"https://example.test/repo","vcs_info":{"vcs":"git"}}', "vcs"),
        ("{}", "direct-url"),
        ("not-json", "unknown"),
    ],
)
def test_distribution_install_mode_rejects_non_wheel_direct_urls(
    monkeypatch: pytest.MonkeyPatch,
    direct_url: str | None,
    expected: str,
) -> None:
    """Source trees, sdists, and VCS installs cannot satisfy wheel evidence."""

    class FakeDistribution:
        def read_text(self, filename: str) -> str | None:
            assert filename == "direct_url.json"
            return direct_url

    monkeypatch.setattr(serialization_impl, "distribution", lambda _: FakeDistribution())
    assert serialization_impl.distribution_install_mode("finite-element-options") == expected
