"""GitHub Actions supply-chain policy checks used by portfolio governance."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any

REQUIRED_WORKFLOW_POLICY = {
    "action_pinning": "full_length_commit_sha",
    "token_permissions": "least_privilege_explicit",
    "checkout_credentials": "non_persistent",
    "security_audit": "zizmor_medium_or_higher_zero_findings",
    "pinning_tool": "pinact",
}
MIN_WORKFLOW_EVIDENCE_SOURCES = 2
USES_PATTERN = re.compile(r"^\s*(?:-\s*)?uses:\s*['\"]?([^\s'\"]+)")
FULL_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
TOP_LEVEL_PERMISSIONS_PATTERN = re.compile(r"(?m)^permissions\s*:")
PERMISSIONS_LINE_PATTERN = re.compile(r"^(\s*)permissions\s*:\s*([^#]*?)\s*(?:#.*)?$")
PERMISSION_SCOPE_PATTERN = re.compile(
    r"^\s*([a-z][a-z0-9-]*)\s*:\s*['\"]?write['\"]?\s*(?:#.*)?$"
)
CHECKOUT_PATTERN = re.compile(r"actions/checkout@[0-9a-f]{40}")


def validate_workflow_policy(
    contract: dict[str, Any], errors: list[str]
) -> dict[str, set[str]]:
    governance = contract.get("governance")
    policy = governance.get("github_actions") if isinstance(governance, dict) else None
    prefix = "governance.github_actions"
    if not isinstance(policy, dict):
        policy = contract.get("github_actions")
        prefix = "github_actions"
    if not isinstance(policy, dict):
        errors.append(f"{prefix} must be an object")
        return {}
    for key, expected in REQUIRED_WORKFLOW_POLICY.items():
        if policy.get(key) != expected:
            errors.append(f"{prefix}.{key} must be {expected!r}")
    evidence = policy.get("evidence")
    if not isinstance(evidence, list) or len(evidence) < MIN_WORKFLOW_EVIDENCE_SOURCES:
        errors.append(f"{prefix}.evidence must contain at least two sources")
    return _write_exceptions(policy, prefix, errors)


def _write_exceptions(
    policy: dict[str, Any], prefix: str, errors: list[str]
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    raw_exceptions = policy.get("write_permission_exceptions")
    if not isinstance(raw_exceptions, list):
        errors.append(f"{prefix}.write_permission_exceptions must be a list")
        return result
    for index, item in enumerate(raw_exceptions):
        item_prefix = f"{prefix}.write_permission_exceptions[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{item_prefix} must be an object")
            continue
        path = item.get("path")
        scopes = item.get("scopes")
        if not isinstance(path, str) or not path.startswith(".github/workflows/"):
            errors.append(f"{item_prefix}.path must name an exact workflow")
            continue
        if (
            not isinstance(scopes, list)
            or not scopes
            or not all(isinstance(scope, str) and scope for scope in scopes)
        ):
            errors.append(f"{item_prefix}.scopes must be a non-empty string list")
            continue
        _validate_exception_metadata(item, item_prefix, errors)
        if path in result:
            errors.append(f"duplicate workflow write-permission exception: {path}")
        result[path] = set(scopes)
    return result


def _validate_exception_metadata(
    item: dict[str, Any], prefix: str, errors: list[str]
) -> None:
    for field in ("reason", "owner"):
        value = item.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{prefix}.{field} is required")
    try:
        review_date = date.fromisoformat(str(item.get("review_by")))
    except ValueError:
        errors.append(f"{prefix}.review_by must be an ISO date")
    else:
        if review_date < date.today():
            errors.append(f"{prefix} expired on {review_date.isoformat()}")


def workflow_write_scopes(lines: list[str]) -> set[str]:
    scopes: set[str] = set()
    for index, line in enumerate(lines):
        match = PERMISSIONS_LINE_PATTERN.match(line)
        if not match:
            continue
        base_indent = len(match.group(1))
        scalar = match.group(2).strip().strip("'\"")
        if scalar:
            if "write" in scalar.lower():
                scopes.add("*")
            continue
        for candidate in lines[index + 1 :]:
            stripped = candidate.strip()
            if not stripped:
                continue
            indent = len(candidate) - len(candidate.lstrip())
            if indent <= base_indent:
                break
            scope_match = PERMISSION_SCOPE_PATTERN.match(candidate)
            if scope_match:
                scopes.add(scope_match.group(1))
    return scopes


def validate_workflows(
    root: Path, errors: list[str], write_exceptions: dict[str, set[str]]
) -> None:
    workflows = root / ".github" / "workflows"
    seen_write_paths: set[str] = set()
    for path in sorted((*workflows.glob("*.yml"), *workflows.glob("*.yaml"))):
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(root).as_posix()
        if not TOP_LEVEL_PERMISSIONS_PATTERN.search(text):
            errors.append(f"workflow missing explicit top-level permissions: {rel}")
        lines = text.splitlines()
        _validate_write_scopes(rel, lines, write_exceptions, seen_write_paths, errors)
        _validate_action_references(rel, lines, errors)
    for path in sorted(set(write_exceptions) - seen_write_paths):
        errors.append(f"stale workflow write-permission exception: {path}")


def _validate_write_scopes(
    rel: str,
    lines: list[str],
    exceptions: dict[str, set[str]],
    seen: set[str],
    errors: list[str],
) -> None:
    scopes = workflow_write_scopes(lines)
    if "*" in scopes:
        errors.append(f"workflow uses forbidden broad write permissions: {rel}")
    elif scopes:
        seen.add(rel)
        expected = exceptions.get(rel)
        if expected != scopes:
            errors.append(
                f"workflow write scopes need an exact contract exception: {rel}: "
                f"actual={sorted(scopes)}, expected={sorted(expected or set())}"
            )


def _validate_action_references(rel: str, lines: list[str], errors: list[str]) -> None:
    for index, line in enumerate(lines):
        match = USES_PATTERN.match(line)
        if not match:
            continue
        action = match.group(1).rstrip("'\"")
        if action.startswith("./"):
            continue
        ref = action.rsplit("@", 1)[-1]
        if not FULL_SHA_PATTERN.fullmatch(ref):
            errors.append(
                f"workflow action is not SHA-pinned: {rel}:{index + 1}: {action}"
            )
        if CHECKOUT_PATTERN.search(action) and not checkout_is_nonpersistent(
            lines, index
        ):
            errors.append(
                f"checkout credentials persist without an exact policy exception: "
                f"{rel}:{index + 1}"
            )


def checkout_is_nonpersistent(lines: list[str], uses_index: int) -> bool:
    uses_line = lines[uses_index]
    uses_indent = len(uses_line) - len(uses_line.lstrip())
    step_indent = (
        uses_indent if uses_line.lstrip().startswith("- ") else uses_indent - 2
    )
    for line in lines[uses_index + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip())
        if indent < step_indent or (
            indent == step_indent and stripped.startswith("- ")
        ):
            break
        if re.fullmatch(r"persist-credentials:\s*false", stripped):
            return True
    return False
