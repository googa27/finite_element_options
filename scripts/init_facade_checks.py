"""AST checks for keeping package initializers as low-side-effect facades."""

from __future__ import annotations

import ast
from pathlib import Path

ALLOWED_INIT_FUNCTIONS = {"__getattr__", "__dir__"}


def _type_checking_guard(node: ast.If) -> bool:
    test = node.test
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute)
        and isinstance(test.value, ast.Name)
        and test.value.id == "typing"
        and test.attr == "TYPE_CHECKING"
    )


def _assignment_targets(
    node: ast.Assign | ast.AnnAssign | ast.AugAssign,
) -> list[ast.expr]:
    if isinstance(node, ast.Assign):
        return list(node.targets)
    return [node.target]


def _metadata_assignment(
    node: ast.Assign | ast.AnnAssign | ast.AugAssign,
) -> bool:
    allowed_names = {"__all__", "__version__"}
    targets = _assignment_targets(node)
    return bool(targets) and all(
        (isinstance(target, ast.Name) and target.id in allowed_names)
        or (isinstance(target, ast.Attribute) and target.attr == "__module__")
        for target in targets
    )


def _static_metadata_assignment(node: ast.Assign | ast.AnnAssign) -> bool:
    return _metadata_assignment(node) and not any(
        isinstance(child, ast.Call) for child in ast.walk(node.value)
    )


def _optional_import_try(node: ast.Try) -> bool:
    body_is_facade = all(
        isinstance(item, (ast.Import, ast.ImportFrom))
        or (
            isinstance(item, (ast.Assign, ast.AnnAssign))
            and _static_metadata_assignment(item)
        )
        for item in node.body
    )
    handlers_are_fallbacks = all(
        all(
            isinstance(item, (ast.Assign, ast.AnnAssign))
            and (
                _static_metadata_assignment(item)
                or (isinstance(item.value, ast.Constant) and item.value.value is None)
            )
            for item in handler.body
        )
        for handler in node.handlers
    )
    return (
        not node.orelse
        and not node.finalbody
        and body_is_facade
        and handlers_are_fallbacks
    )


def init_implementation(path: Path) -> list[str]:
    """Return top-level runtime constructs forbidden in a facade initializer."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    findings: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            findings.append(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name not in ALLOWED_INIT_FUNCTIONS:
                findings.append(node.name)
        elif isinstance(node, ast.If):
            if not _type_checking_guard(node):
                findings.append("top-level-if")
        elif isinstance(node, ast.Try):
            if not _optional_import_try(node):
                findings.append("Try")
        elif isinstance(node, (ast.For, ast.While, ast.With, ast.Match)):
            findings.append(type(node).__name__)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            value = getattr(node, "value", None)
            if value is not None and any(
                isinstance(child, ast.Call) for child in ast.walk(value)
            ):
                findings.append("assignment-call")
        elif isinstance(node, ast.Expr):
            if not (
                isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                findings.append("expression")
        elif not isinstance(node, (ast.Import, ast.ImportFrom, ast.Delete, ast.Pass)):
            findings.append(type(node).__name__)
    return findings
