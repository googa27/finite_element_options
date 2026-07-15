#!/usr/bin/env python3
"""Self-check the additive AI, hierarchy, and workflow governance gate."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path
from types import ModuleType

SCRIPT = Path(__file__).with_name("check_ai_hierarchy_policy.py")
WORKFLOW_SCRIPT = Path(__file__).with_name("workflow_policy_checks.py")


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_checker() -> ModuleType:
    load_module("workflow_policy_checks", WORKFLOW_SCRIPT)
    return load_module("ai_hierarchy_policy", SCRIPT)


def write_workflow(
    root: Path, *, pinned: bool = True, persistent: bool = False
) -> None:
    workflow = root / ".github" / "workflows" / "ci.yml"
    workflow.parent.mkdir(parents=True, exist_ok=True)
    ref = "34e114876b0b11c390a56381ad16ebd13914f8d5" if pinned else "v4"
    persistence = "true" if persistent else "false"
    workflow.write_text(
        "name: CI\n"
        "on: [push]\n"
        "permissions:\n"
        "  contents: read\n"
        "jobs:\n"
        "  test:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        f"      - uses: actions/checkout@{ref}\n"
        "        with:\n"
        f"          persist-credentials: {persistence}\n",
        encoding="utf-8",
    )


def main() -> int:
    checker = load_checker()
    assert checker.branch_review_metrics([20], 3, 0.95)[0]
    assert checker.branch_review_metrics([19, 1], 3, 0.95)[0]
    assert not checker.branch_review_metrics([10, 10], 3, 0.95)[0]
    assert checker.branch_review_metrics([18, 1, 1], 3, 0.95)[0]
    assert not checker.branch_review_metrics([7, 7, 6], 3, 0.95)[0]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        checker.ROOT = root
        facade = root / "src" / "example" / "compat"
        facade.mkdir(parents=True)
        (facade / "__init__.py").write_text(
            "from elsewhere import Public as Public\n", encoding="utf-8"
        )
        assert checker.marker_only_package(facade)

        write_workflow(root)
        errors: list[str] = []
        checker.validate_workflows(checker.ROOT, errors, {})
        assert not errors, errors

        write_workflow(root, pinned=False)
        errors = []
        checker.validate_workflows(checker.ROOT, errors, {})
        assert any("not SHA-pinned" in error for error in errors)

        write_workflow(root, persistent=True)
        errors = []
        checker.validate_workflows(checker.ROOT, errors, {})
        assert any("credentials persist" in error for error in errors)

        workflow = root / ".github" / "workflows" / "ci.yml"
        workflow.write_text(
            "name: CI\non: [push]\npermissions:\n  contents: read\njobs:\n"
            "  test:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      - uses: actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5\n"
            "      - uses: actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5\n"
            "        with:\n          persist-credentials: false\n",
            encoding="utf-8",
        )
        errors = []
        checker.validate_workflows(checker.ROOT, errors, {})
        credential_errors = [
            error for error in errors if "credentials persist" in error
        ]
        assert len(credential_errors) == 1, credential_errors

        workflow.write_text(
            workflow.read_text(encoding="utf-8").replace(
                "permissions:\n  contents: read", "permissions: write-all"
            ),
            encoding="utf-8",
        )
        errors = []
        checker.validate_workflows(checker.ROOT, errors, {})
        assert any("forbidden broad write permissions" in error for error in errors)

        write_workflow(root)
        workflow.write_text(
            workflow.read_text(encoding="utf-8").replace(
                "contents: read", "contents: write"
            ),
            encoding="utf-8",
        )
        errors = []
        checker.validate_workflows(
            checker.ROOT, errors, {".github/workflows/ci.yml": {"contents"}}
        )
        assert not errors, errors

    print("AI/hierarchy policy self-tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
