"""GitHub Actions supply-chain policy checks used by portfolio governance."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any, Iterator

import yaml

REQUIRED_WORKFLOW_POLICY = {
    "action_pinning": "full_length_commit_sha",
    "token_permissions": "least_privilege_explicit",
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

    raw_exceptions = policy.get("write_permission_exceptions")
    if not isinstance(raw_exceptions, list):
        errors.append("github_actions.write_permission_exceptions must be a list")
        return {}
    parsed: dict[str, set[str]] = {}
    for item in raw_exceptions:
        if not isinstance(item, dict) or not REQUIRED_EXCEPTION_FIELDS.issubset(item):
            errors.append(
                "write permission exception missing path/scopes/reason/owner/review_by"
            )
            continue
        path = str(item["path"])
        scopes = item["scopes"]
        review_by = _iso_date(item["review_by"])
        if path in parsed:
            errors.append(f"duplicate write permission exception: {path}")
        if (
            not isinstance(scopes, list)
            or not scopes
            or not all(isinstance(scope, str) and scope for scope in scopes)
        ):
            errors.append(
                f"write permission exception {path}: scopes must be non-empty"
            )
            continue
        if review_by is None:
            errors.append(f"write permission exception {path}: invalid review_by date")
        elif review_by < date.today():
            errors.append(f"write permission exception {path}: review_by is expired")
        parsed[path] = set(scopes)
    return parsed


def _workflow_document(path: Path, errors: list[str]) -> dict[str, Any] | None:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
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
        access = str(raw_access).lower()
        if access not in {"read", "write", "none"}:
            errors.append(f"{label}: invalid {scope} permission {raw_access!r}")
        if access == "write":
            scopes.add(str(scope))
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


def _validate_action_reference(
    uses: str, step: dict[str, Any] | None, label: str, errors: list[str]
) -> None:
    if uses.startswith("./"):
        return
    if uses.startswith("docker://"):
        if not DIGEST_PATTERN.fullmatch(uses.rsplit("@", 1)[-1]):
            errors.append(f"{label}: docker action is not digest-pinned: {uses}")
        return
    action, separator, reference = uses.rpartition("@")
    if not separator or not FULL_SHA_PATTERN.fullmatch(reference):
        errors.append(f"{label}: action is not SHA-pinned: {uses}")
        return
    if action != "actions/checkout" or step is None:
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
    seen_write_paths: set[str] = set()
    for path in sorted((*workflows.glob("*.yml"), *workflows.glob("*.yaml"))):
        rel = path.relative_to(root).as_posix()
        document = _workflow_document(path, errors)
        if document is None:
            continue
        if "permissions" not in document:
            errors.append(f"{rel}: missing explicit top-level permissions")
            top_scopes: set[str] = set()
        else:
            top_scopes = _permission_write_scopes(
                document["permissions"], f"{rel}: top-level", errors
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
        for uses, step in _uses_entries(document, rel, errors):
            _validate_action_reference(uses, step, rel, errors)

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
