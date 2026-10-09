#!/usr/bin/env python3
"""Enforce the reviewed Actions runtime and hosted-runner contract (issue173)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_OVERRIDES = {
    "ACTIONS_ALLOW_USE_UNSECURE_NODE_VERSION",
    "FORCE_JAVASCRIPT_ACTIONS_TO_NODE24",
}


def _scalars(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [
            item
            for key, child in value.items()
            for item in [str(key), *_scalars(child)]
        ]
    if isinstance(value, list):
        return [item for child in value for item in _scalars(child)]
    return [str(value)]


def validate_runtime_contract(
    policy: dict[str, Any], workflows: dict[str, Any]
) -> list[str]:
    """Validate parsed workflows against the canonical reviewed selection."""
    errors: list[str] = []
    actions = policy["actions"]
    for action, selected in actions.items():
        if selected["runtime"] != "node24" or not re.fullmatch(
            r"[0-9a-f]{40}", selected["sha"]
        ):
            errors.append(f"invalid canonical Node24 action selection: {action}")
    if set(workflows) != set(policy["workflows"]):
        errors.append("workflow inventory differs from the reviewed runtime contract")
    seen: set[str] = set()
    for path, workflow in workflows.items():
        if not isinstance(workflow, dict):
            errors.append(f"{path}: workflow must be a mapping")
            continue
        if workflow.get("permissions") != {"contents": "read"}:
            errors.append(f"{path}: workflow permissions must be read-only contents")
        if any(
            token in scalar
            for scalar in _scalars(workflow)
            for token in FORBIDDEN_OVERRIDES
        ):
            errors.append(f"{path}: runtime override is forbidden")
        jobs = workflow.get("jobs")
        if not isinstance(jobs, dict) or not jobs:
            errors.append(f"{path}: jobs must be a nonempty mapping")
            continue
        for name, job in jobs.items():
            label = f"{path}:{name}"
            if not isinstance(job, dict):
                errors.append(f"{label}: job must be a mapping")
                continue
            if job.get("runs-on") != policy["runner_label"]:
                errors.append(f"{label}: runner must be {policy['runner_label']}")
            timeout = job.get("timeout-minutes")
            if (
                isinstance(timeout, bool)
                or not isinstance(timeout, int)
                or not 1 <= timeout <= policy["maximum_job_minutes"]
            ):
                errors.append(f"{label}: explicit bounded timeout is required")
            if "permissions" in job and job["permissions"] != {"contents": "read"}:
                errors.append(f"{label}: job permission escalation is forbidden")
            if "if" in job or "continue-on-error" in job:
                errors.append(f"{label}: job gates must not be conditional or waived")
            steps = job.get("steps")
            if not isinstance(steps, list):
                errors.append(f"{label}: steps must be a list")
                continue
            for step in steps:
                if not isinstance(step, dict):
                    errors.append(f"{label}: step must be a mapping")
                    continue
                if "if" in step or "continue-on-error" in step:
                    errors.append(
                        f"{label}: step gates must not be conditional or waived"
                    )
                if "uses" not in step:
                    continue
                action, separator, ref = str(step["uses"]).partition("@")
                if (
                    not separator
                    or action not in actions
                    or ref != actions[action]["sha"]
                ):
                    errors.append(f"{label}: unreviewed action {step['uses']}")
                    continue
                seen.add(action)
                inputs = step.get("with", {})
                if not isinstance(inputs, dict):
                    errors.append(f"{label}: action inputs must be a mapping")
                elif action == "actions/checkout":
                    if inputs.get("persist-credentials") is not False:
                        errors.append(f"{label}: checkout credentials must not persist")
                elif action == "actions/setup-python":
                    if "pip-install" in inputs:
                        errors.append(
                            f"{label}: removed pip-install input is forbidden"
                        )
                    if not inputs.get("python-version"):
                        errors.append(f"{label}: explicit Python selection is required")
                elif action == "actions/upload-artifact":
                    if inputs.get("archive", True) is not True:
                        errors.append(
                            f"{label}: artifact ZIP semantics must be retained"
                        )
                    if inputs.get("if-no-files-found") != "error":
                        errors.append(f"{label}: missing artifact must fail")
    policy_path = ".github/workflows/ai-hierarchy-policy.yml"
    policy_workflow = workflows.get(policy_path, {})
    policy_steps = policy_workflow.get("jobs", {}).get("policy", {}).get("steps", [])
    required = (
        "python3 scripts/check_actions_runtime_contract.py\n"
        "python3 scripts/selftest_actions_runtime_contract.py\n"
    )
    if sum(step.get("run") == required for step in policy_steps) != 1:
        errors.append(
            "runtime policy must execute its checker and self-test exactly once"
        )
    for action in sorted(set(actions) - seen):
        errors.append(f"missing reviewed action: {action}")
    return errors


def load_contract(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read canonical JSON and every actual YAML workflow using PyYAML."""
    architecture = json.loads((root / "docs/ARCHITECTURE.yaml").read_text())
    policy = architecture["governance"]["github_actions"]["runtime_contract"]
    workflows = {
        path.relative_to(root).as_posix(): yaml.safe_load(path.read_text())
        for path in sorted((root / ".github/workflows").iterdir())
        if path.suffix in {".yml", ".yaml"}
    }
    return policy, workflows


def main() -> int:
    """Fail the workflow gate on runtime, identity or runner drift."""
    errors = validate_runtime_contract(*load_contract(ROOT))
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        return 1
    print("Actions runtime contract passed: reviewed Node24 pins / ubuntu-24.04")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
