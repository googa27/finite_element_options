"""Capability matrix documentation and README example contract tests."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

from finite_element_options.contracts.capability_matrix import (
    DEFAULT_CAPABILITY_RECORDS,
    CapabilityStatus,
)

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts" / "generate_capability_docs.py"
spec = importlib.util.spec_from_file_location("generate_capability_docs", GENERATOR)
assert spec is not None and spec.loader is not None
generate_capability_docs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generate_capability_docs)
README_END_MARKER = generate_capability_docs.README_END_MARKER
README_START_MARKER = generate_capability_docs.README_START_MARKER
render_capability_matrix_markdown = (
    generate_capability_docs.render_capability_matrix_markdown
)
render_readme_capability_snapshot = (
    generate_capability_docs.render_readme_capability_snapshot
)


def test_capability_registry_has_unique_evidenced_maturity_records() -> None:
    """Every advertised capability should carry maturity and evidence."""

    ids = [record.capability_id for record in DEFAULT_CAPABILITY_RECORDS]
    assert len(ids) == len(set(ids))

    for record in DEFAULT_CAPABILITY_RECORDS:
        assert record.status in CapabilityStatus
        assert record.evidence_ids, record.capability_id
        assert record.evidence_scope, record.capability_id
        assert record.limitations, record.capability_id
        if record.optional_extra is not None:
            assert record.absence_behavior, record.capability_id
        if record.status in {CapabilityStatus.VALIDATED, CapabilityStatus.PRODUCTION}:
            assert record.benchmark_ids or record.reference_ids, record.capability_id


def test_capability_registry_evidence_paths_exist() -> None:
    """Generated evidence references should point to committed files or dirs."""

    missing: list[str] = []
    for record in DEFAULT_CAPABILITY_RECORDS:
        for evidence_id in record.evidence_ids:
            path_text = evidence_id.split("#", 1)[0]
            if path_text.startswith(("http://", "https://")):
                continue
            if not (ROOT / path_text).exists():
                missing.append(f"{record.capability_id}: {evidence_id}")
    assert not missing


def test_generated_capability_matrix_doc_is_current() -> None:
    """The committed capability matrix should be exactly generated."""

    expected = render_capability_matrix_markdown(DEFAULT_CAPABILITY_RECORDS)
    with open("docs/CAPABILITY_MATRIX.md", encoding="utf-8") as handle:
        actual = handle.read()
    assert actual == expected


def test_readme_capability_snapshot_is_generated_and_current() -> None:
    """README capability claims should come from the same registry."""

    with open("README.md", encoding="utf-8") as handle:
        readme = handle.read()

    assert README_START_MARKER in readme
    assert README_END_MARKER in readme
    start = readme.index(README_START_MARKER) + len(README_START_MARKER)
    end = readme.index(README_END_MARKER)
    actual = readme[start:end].strip()
    expected = render_readme_capability_snapshot(DEFAULT_CAPABILITY_RECORDS).strip()
    assert actual == expected


def test_readme_code_examples_execute_from_installed_package_context() -> None:
    """README Python and shell examples should be executable documentation."""

    subprocess.run(
        [sys.executable, "scripts/check_readme_examples.py", "README.md"],
        check=True,
    )


def test_capability_doc_check_script_passes() -> None:
    """The docs staleness check used by CI should pass locally."""

    subprocess.run(
        [sys.executable, "scripts/generate_capability_docs.py", "--check"],
        check=True,
    )
