"""GitHub Actions supply-chain policy checks used by portfolio governance."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any, Iterator

import yaml

REQUIRED_WORKFLOW_POLICY = {
    "action_pinning": "full_length_commit_sha",
    "token_permissions": "read_only_top_level_with_exact_job_write_exceptions",
    "checkout_credentials": "non_persistent",
    "security_audit": "zizmor_medium_or_higher_zero_findings",
    "pinning_tool": "pinact",
}
REQUIRED_EXCEPTION_FIELDS = {"path", "scopes", "reason", "owner", "review_by"}
MIN_WORKFLOW_EVIDENCE_SOURCES = 2
FULL_SHA_PATTERN = re.compile(r"[0-9a-fA-F]{40}")
DIGEST_PATTERN = re.compile(r"sha256:[0-9a-fA-F]{64}")


def _iso_date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def validate_workflow_policy(
    contract: dict[str, Any], errors: list[str]
) -> dict[str, set[str]]:
    governance = contract.get("governance")
    policy = governance.get("github_actions") if isinstance(governance, dict) else None
    if not isinstance(policy, dict):
        policy = contract.get("github_actions")
    if not isinstance(policy, dict):
        errors.append("missing github_actions governance policy")
        return {}
    for key, expected in REQUIRED_WORKFLOW_POLICY.items():
        if policy.get(key) != expected:
            errors.append(f"github_actions.{key} must be {expected!r}")
    evidence = policy.get("evidence")
    if not isinstance(evidence, list) or len(evidence) < MIN_WORKFLOW_EVIDENCE_SOURCES:
        errors.append("github_actions.evidence must contain at least two sources")
    else:
        for index, item in enumerate(evidence):
            source = item.get("source") if isinstance(item, dict) else None
            finding = item.get("finding") if isinstance(item, dict) else None
            if not isinstance(source, str) or not source.startswith("https://"):
                errors.append(f"github_actions.evidence[{index}] needs an HTTPS source")
            if not isinstance(finding, str) or not finding.strip():
                errors.append(f"github_actions.evidence[{index}] needs a finding")

    raw_exceptions = policy.get("write_permission_exceptions")
    if not isinstance(raw_exceptions, list):
        errors.append("github_actions.write_permission_exceptions must be a list")
        return {}
    parsed: dict[str, set[str]] = {}
    for index, item in enumerate(raw_exceptions):
        label = f"write_permission_exceptions[{index}]"
        if not isinstance(item, dict) or not REQUIRED_EXCEPTION_FIELDS.issubset(item):
            errors.append(f"{label} missing path/scopes/reason/owner/review_by")
            continue
        path = str(item["path"])
        scopes = item["scopes"]
        review_by = _iso_date(item["review_by"])
        if not re.fullmatch(r"\.github/workflows/[^/]+\.ya?ml", path):
            errors.append(f"{label}.path must name an exact workflow YAML file: {path}")
            continue
        if path in parsed:
            errors.append(f"duplicate write permission exception: {path}")
            continue
        if (
            not isinstance(scopes, list)
            or not scopes
            or not all(
                isinstance(scope, str) and bool(scope) and scope == scope.lower()
                for scope in scopes
            )
        ):
            errors.append(
                f"{label} ({path}): scopes must be non-empty lowercase strings"
            )
            continue
        if not str(item["reason"]).strip() or not str(item["owner"]).strip():
            errors.append(f"{label} ({path}): reason and owner are required")
        if review_by is None:
            errors.append(f"{label} ({path}): invalid review_by date")
        elif review_by < date.today():
            errors.append(f"{label} ({path}): review_by is expired")
        parsed[path] = set(scopes)
    return parsed


def _workflow_document(path: Path, errors: list[str]) -> dict[str, Any] | None:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        errors.append(f"{path}: cannot read workflow: {exc}")
        return None
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        errors.append(f"{path}: invalid workflow YAML: {exc}")
        return None
    if not isinstance(document, dict):
        errors.append(f"{path}: workflow root must be a mapping")
        return None
    return document


def _permission_write_scopes(value: Any, label: str, errors: list[str]) -> set[str]:
    if isinstance(value, str):
        access = value.lower()
        if access == "write-all":
            return {"*"}
        if access == "read-all":
            return set()
        errors.append(f"{label}: unsupported scalar permissions value {value!r}")
        return set()
    if not isinstance(value, dict):
        errors.append(f"{label}: permissions must be a mapping or read-all")
        return set()
    scopes: set[str] = set()
    for scope, raw_access in value.items():
        scope_name = str(scope)
        if scope_name != scope_name.lower():
            errors.append(
                f"{label}: permission scope must be lowercase: {scope_name!r}"
            )
            continue
        if not isinstance(raw_access, str):
            errors.append(f"{label}: invalid {scope_name} permission {raw_access!r}")
            continue
        access = raw_access.lower()
        if access not in {"read", "write", "none"}:
            errors.append(f"{label}: invalid {scope_name} permission {raw_access!r}")
        if access == "write":
            scopes.add(scope_name)
    return scopes


def _uses_entries(
    document: dict[str, Any], label: str, errors: list[str]
) -> Iterator[tuple[str, dict[str, Any] | None]]:
    jobs = document.get("jobs")
    if not isinstance(jobs, dict):
        errors.append(f"{label}: jobs must be a mapping")
        return
    for job_name, raw_job in jobs.items():
        if not isinstance(raw_job, dict):
            errors.append(f"{label}: job {job_name!r} must be a mapping")
            continue
        job_uses = raw_job.get("uses")
        if job_uses is not None:
            if isinstance(job_uses, str):
                yield job_uses, None
            else:
                errors.append(f"{label}: job {job_name!r} uses must be a string")
        steps = raw_job.get("steps", [])
        if not isinstance(steps, list):
            errors.append(f"{label}: job {job_name!r} steps must be a list")
            continue
        for index, step in enumerate(steps):
            if not isinstance(step, dict):
                errors.append(
                    f"{label}: job {job_name!r} step {index} must be a mapping"
                )
                continue
            uses = step.get("uses")
            if uses is None:
                continue
            if isinstance(uses, str):
                yield uses, step
            else:
                errors.append(
                    f"{label}: job {job_name!r} step {index} uses must be a string"
                )


def _local_action_file(root: Path, uses: str) -> Path | None:
    candidate = (root / uses).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    for name in ("action.yml", "action.yaml"):
        path = candidate / name
        if path.is_file():
            return path
    return None


def _validate_local_action(
    root: Path,
    uses: str,
    label: str,
    errors: list[str],
    visited: set[Path],
) -> None:
    action_path = _local_action_file(root, uses)
    if action_path is None:
        errors.append(
            f"{label}: local action metadata is missing or escapes root: {uses}"
        )
        return
    action_path = action_path.resolve()
    if action_path in visited:
        return
    visited.add(action_path)
    document = _workflow_document(action_path, errors)
    if document is None:
        return
    runs = document.get("runs")
    if not isinstance(runs, dict) or runs.get("using") != "composite":
        return
    steps = runs.get("steps")
    if not isinstance(steps, list):
        errors.append(f"{action_path}: composite runs.steps must be a list")
        return
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            errors.append(f"{action_path}: composite step {index} must be a mapping")
            continue
        nested = step.get("uses")
        if nested is None:
            continue
        if not isinstance(nested, str):
            errors.append(
                f"{action_path}: composite step {index} uses must be a string"
            )
            continue
        _validate_action_reference(
            root,
            nested,
            step,
            f"{action_path}: composite step {index}",
            errors,
            visited,
        )


def _validate_action_reference(
    root: Path,
    uses: str,
    step: dict[str, Any] | None,
    label: str,
    errors: list[str],
    visited: set[Path],
) -> None:
    if uses.startswith("./"):
        _validate_local_action(root, uses, label, errors, visited)
        return
    if uses.startswith("docker://"):
        if not DIGEST_PATTERN.fullmatch(uses.rsplit("@", 1)[-1]):
            errors.append(f"{label}: docker action is not digest-pinned: {uses}")
        return
    action, separator, reference = uses.rpartition("@")
    if not separator or not FULL_SHA_PATTERN.fullmatch(reference):
        errors.append(f"{label}: action is not SHA-pinned: {uses}")
        return
    if action != "actions/checkout":
        return
    if step is None:
        errors.append(
            f"{label}: actions/checkout is invalid as a job-level reusable workflow"
        )
        return
    inputs = step.get("with")
    persisted = inputs.get("persist-credentials") if isinstance(inputs, dict) else None
    if not (persisted is False or str(persisted).lower() == "false"):
        errors.append(f"{label}: checkout credentials persist for action {uses}")


def validate_workflows(
    root: Path,
    errors: list[str],
    write_exceptions: dict[str, set[str]],
) -> None:
    workflows = root / ".github" / "workflows"
    workflow_paths = sorted((*workflows.glob("*.yml"), *workflows.glob("*.yaml")))
    if not workflow_paths:
        errors.append(".github/workflows must contain at least one workflow YAML file")
        return
    seen_write_paths: set[str] = set()
    for path in workflow_paths:
        rel = path.relative_to(root).as_posix()
        document = _workflow_document(path, errors)
        if document is None:
            continue
        if "permissions" not in document:
            errors.append(f"{rel}: missing explicit top-level permissions")
            top_scopes: set[str] = set()
        else:
            top_scopes = _permission_write_scopes(
                document.get("permissions"), f"{rel}: top-level permissions", errors
            )
            if top_scopes:
                errors.append(
                    f"{rel}: top-level write scopes are forbidden; move writes to the exact mutating job: {sorted(top_scopes)}"
                )
        write_scopes = set(top_scopes)
        jobs = document.get("jobs")
        if isinstance(jobs, dict):
            for job_name, raw_job in jobs.items():
                if isinstance(raw_job, dict) and "permissions" in raw_job:
                    write_scopes.update(
                        _permission_write_scopes(
                            raw_job["permissions"], f"{rel}: job {job_name!r}", errors
                        )
                    )
        visited_local_actions: set[Path] = set()
        for uses, step in _uses_entries(document, rel, errors):
            _validate_action_reference(
                root, uses, step, rel, errors, visited_local_actions
            )

        if "*" in write_scopes:
            errors.append(f"{rel}: forbidden broad write permissions")
        elif write_scopes:
            seen_write_paths.add(rel)
            expected = write_exceptions.get(rel)
            if expected != write_scopes:
                errors.append(
                    f"{rel}: write scopes {sorted(write_scopes)} require an exact "
                    "write_permission_exceptions entry"
                )
    for path in sorted(set(write_exceptions) - seen_write_paths):
        errors.append(f"stale write permission exception: {path}")
