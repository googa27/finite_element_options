#!/usr/bin/env python3
"""Validate evidence-backed AI-assistance and semantic source-hierarchy policy."""

from __future__ import annotations

import ast
import json
import math
import sys
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "docs" / "ARCHITECTURE.yaml"
ALLOWED_INIT_FUNCTIONS = {"__getattr__", "__dir__"}
IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "build",
    "dist",
}
METADATA_NAMES = {
    "__init__.py",
    "README.md",
    "ARCHITECTURE.md",
    "ARCHITECTURE.yaml",
    "py.typed",
    ".gitkeep",
}
REQUIRED_AI_POLICY = {
    "output_trust": "untrusted_until_human_review_and_executable_verification",
    "human_accountability": True,
    "change_scope": "small_reviewable_single_purpose_slices",
    "agent_generated_tests": "not_sufficient_as_sole_oracle",
    "dependency_changes": "human_approval_and_existence_maintenance_license_security_verification",
    "high_risk_review": "human_required",
    "least_privilege": "workspace_scoped_network_and_secret_access_requires_approval",
    "provenance": "record_agent_assistance_and_verification_evidence",
    "measurement": "objective_metrics_not_self_report",
}
REQUIRED_AI_METRICS = {
    "lead_time",
    "review_time",
    "ci_failure_rate",
    "revert_rate",
    "defect_escape_rate",
    "code_churn",
}
REQUIRED_HIERARCHY_POLICY = {
    "principle": "semantic_cohesion_and_low_coupling_not_equal_branch_size",
    "empty_runtime_directories": "forbidden_without_exact_exception",
    "init_modules": "facade_only_no_domain_implementation",
    "concentration": "review_trigger_not_rebalance_mandate",
    "minimum_branches": 3,
    "minimum_descendant_modules": 20,
    "include_direct_modules_as_branch": True,
    "new_or_worsened_unclassified_imbalance": "forbidden",
}
REQUIRED_STRUCTURAL_ROLES = {
    "namespace_package",
    "compatibility_facade",
    "generated_mount",
    "plugin_namespace",
    "adapter_namespace",
    "test_mirror",
    "package_data",
    "monorepo_boundary",
}
MIN_EVIDENCE_SOURCES = 4
THREE_BRANCHES = 3
FIVE_BRANCHES = 5
SEVEN_BRANCHES = 7
THREE_BRANCH_MAX_SHARE = 0.85
MID_BRANCH_MAX_SHARE = 0.80
SIX_SEVEN_MAX_SHARE = 0.70
MANY_BRANCH_MAX_SHARE = 0.65
THREE_BRANCH_MAX_EFFECTIVE = 2.25
MID_BRANCH_MAX_EFFECTIVE = 2.50
THREE_BRANCH_MAX_EFFECTIVE_FRACTION = 0.60
MID_BRANCH_MAX_EFFECTIVE_FRACTION = 0.50
MANY_BRANCH_MAX_EFFECTIVE_FRACTION = 0.45


class ContractShapeError(TypeError):
    """Architecture contract root is not a mapping."""

    def __init__(self) -> None:
        super().__init__("architecture contract root must be an object")


def load_contract() -> dict[str, Any]:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractShapeError
    return payload


def ignored(path: Path) -> bool:
    return any(part in IGNORED_DIRS or part.startswith(".") for part in path.parts)


def exception_map(
    contract: dict[str, Any], errors: list[str]
) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for index, item in enumerate(contract.get("exceptions", [])):
        if not isinstance(item, dict):
            continue
        rule = str(item.get("rule", ""))
        path = str(item.get("path", ""))
        if rule not in {
            "empty_runtime_directory",
            "hierarchy_imbalance",
            "init_module_implementation",
        }:
            continue
        if item.get("structural_role") not in REQUIRED_STRUCTURAL_ROLES:
            errors.append(f"exceptions[{index}] has invalid structural_role for {rule}")
        try:
            review_by = date.fromisoformat(str(item.get("review_by", "")))
        except ValueError:
            errors.append(
                f"exceptions[{index}].review_by must be an ISO date for {rule}"
            )
        else:
            if review_by < date.today():
                errors.append(
                    f"exceptions[{index}] hierarchy review expired on {review_by}"
                )
        if not str(item.get("evidence", "")).strip():
            errors.append(f"exceptions[{index}].evidence is required for {rule}")
        result[(rule, path)] = item
    return result


def validate_evidence(prefix: str, evidence: Any, errors: list[str]) -> None:
    if not isinstance(evidence, list) or len(evidence) < MIN_EVIDENCE_SOURCES:
        errors.append(f"{prefix}.evidence must contain at least four sources")
        return
    for index, item in enumerate(evidence):
        source = item.get("source") if isinstance(item, dict) else None
        finding = item.get("finding") if isinstance(item, dict) else None
        if not isinstance(source, str) or not source.startswith("https://"):
            errors.append(f"{prefix}.evidence[{index}] needs an HTTPS source")
        if not isinstance(finding, str) or not finding.strip():
            errors.append(f"{prefix}.evidence[{index}] needs a finding")


def validate_policy_shape(
    contract: dict[str, Any], errors: list[str]
) -> dict[str, Any] | None:
    governance = contract.get("governance")
    ai = (
        governance.get("ai_assisted_development")
        if isinstance(governance, dict)
        else None
    )
    if not isinstance(ai, dict):
        errors.append("governance.ai_assisted_development must be an object")
    else:
        for key, expected in REQUIRED_AI_POLICY.items():
            if ai.get(key) != expected:
                errors.append(
                    f"governance.ai_assisted_development.{key} must be {expected!r}"
                )
        metrics = ai.get("metrics")
        if not isinstance(metrics, list) or set(metrics) != REQUIRED_AI_METRICS:
            errors.append(
                "governance.ai_assisted_development.metrics has the wrong set"
            )
        validate_evidence(
            "governance.ai_assisted_development", ai.get("evidence"), errors
        )
    layout = contract.get("source_layout")
    hierarchy = layout.get("hierarchy_policy") if isinstance(layout, dict) else None
    if not isinstance(hierarchy, dict):
        errors.append("source_layout.hierarchy_policy must be an object")
        return None
    for key, expected in REQUIRED_HIERARCHY_POLICY.items():
        if hierarchy.get(key) != expected:
            errors.append(f"source_layout.hierarchy_policy.{key} must be {expected!r}")
    roles = hierarchy.get("structural_role_exclusions")
    if not isinstance(roles, list) or set(roles) != REQUIRED_STRUCTURAL_ROLES:
        errors.append(
            "source_layout.hierarchy_policy.structural_role_exclusions has the wrong set"
        )
    validate_evidence(
        "source_layout.hierarchy_policy", hierarchy.get("evidence"), errors
    )
    return hierarchy


def init_implementation(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) or (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name not in ALLOWED_INIT_FUNCTIONS
        ):
            names.append(node.name)
    return names


def marker_only_package(path: Path) -> bool:
    init_path = path / "__init__.py"
    if not init_path.is_file():
        return False
    for candidate in path.rglob("*"):
        if (
            ignored(candidate.relative_to(ROOT))
            or not candidate.is_file()
            or candidate == init_path
        ):
            continue
        if candidate.name not in METADATA_NAMES:
            return False
    tree = ast.parse(init_path.read_text(encoding="utf-8"), filename=str(init_path))
    for node in tree.body:
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.Pass)):
            continue
        return False
    return True


def runtime_branches(current: Path) -> list[tuple[str, int]]:
    branches: list[tuple[str, int]] = []
    direct = sum(
        path.suffix == ".py" and path.name != "__init__.py"
        for path in current.iterdir()
        if path.is_file()
    )
    if direct:
        branches.append(("__direct_modules__", direct))
    for child in sorted(path for path in current.iterdir() if path.is_dir()):
        if ignored(child.relative_to(ROOT)):
            continue
        count = sum(
            candidate.name != "__init__.py" and not ignored(candidate.relative_to(ROOT))
            for candidate in child.rglob("*.py")
        )
        if count:
            branches.append((child.name, count))
    return branches


def concentration(branch_counts: list[int]) -> tuple[bool, float, float]:
    count = len(branch_counts)
    total = sum(branch_counts)
    shares = [value / total for value in branch_counts]
    largest = max(shares)
    effective = math.exp(-sum(share * math.log(share) for share in shares))
    fraction = effective / count
    if count == THREE_BRANCHES:
        triggered = largest >= THREE_BRANCH_MAX_SHARE and (
            effective <= THREE_BRANCH_MAX_EFFECTIVE
            or fraction <= THREE_BRANCH_MAX_EFFECTIVE_FRACTION
        )
    elif count <= FIVE_BRANCHES:
        triggered = largest >= MID_BRANCH_MAX_SHARE and (
            effective <= MID_BRANCH_MAX_EFFECTIVE
            or fraction <= MID_BRANCH_MAX_EFFECTIVE_FRACTION
        )
    elif count <= SEVEN_BRANCHES:
        triggered = (
            largest >= SIX_SEVEN_MAX_SHARE
            and fraction <= MANY_BRANCH_MAX_EFFECTIVE_FRACTION
        )
    else:
        triggered = (
            largest >= MANY_BRANCH_MAX_SHARE
            and fraction <= MANY_BRANCH_MAX_EFFECTIVE_FRACTION
        )
    return triggered, largest, effective


def validate_directory(
    current: Path,
    exceptions: dict[tuple[str, str], dict[str, Any]],
    minimum_branches: int,
    minimum_modules: int,
    errors: list[str],
) -> None:
    rel = current.relative_to(ROOT).as_posix()
    if (
        marker_only_package(current)
        and ("empty_runtime_directory", rel) not in exceptions
    ):
        errors.append(f"empty_runtime_directory violation at {rel}")
    init_path = current / "__init__.py"
    if init_path.is_file():
        names = init_implementation(init_path)
        init_rel = init_path.relative_to(ROOT).as_posix()
        if names and ("init_module_implementation", init_rel) not in exceptions:
            errors.append(
                f"init_module_implementation violation at {init_rel}: {sorted(names)}"
            )
    branches = runtime_branches(current)
    counts = [value for _, value in branches]
    if len(counts) < minimum_branches or sum(counts) < minimum_modules:
        return
    triggered, largest, effective = concentration(counts)
    if triggered and ("hierarchy_imbalance", rel) not in exceptions:
        errors.append(
            f"hierarchy_imbalance review required at {rel}: branches={branches}, "
            f"largest_share={largest:.3f}, effective_branches={effective:.2f}"
        )


def validate(contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    hierarchy = validate_policy_shape(contract, errors)
    exceptions = exception_map(contract, errors)
    layout = contract.get("source_layout")
    if (
        hierarchy is None
        or not isinstance(layout, dict)
        or not layout.get("python_rules_applicable", True)
    ):
        return errors
    minimum_branches = int(hierarchy["minimum_branches"])
    minimum_modules = int(hierarchy["minimum_descendant_modules"])
    for rel_root in layout.get("python_source_roots", []):
        source_root = ROOT / rel_root
        if not source_root.is_dir():
            continue
        directories = [
            source_root,
            *sorted(path for path in source_root.rglob("*") if path.is_dir()),
        ]
        for current in directories:
            if not ignored(current.relative_to(ROOT)):
                validate_directory(
                    current, exceptions, minimum_branches, minimum_modules, errors
                )
    return errors


def write_line(message: str) -> None:
    sys.stdout.write(f"{message}\n")


def main() -> int:
    try:
        errors = validate(load_contract())
    except Exception as exc:
        write_line(f"AI/hierarchy policy FAILED\n- {type(exc).__name__}: {exc}")
        return 1
    if errors:
        write_line("AI/hierarchy policy FAILED")
        for error in errors:
            write_line(f"- {error}")
        return 1
    write_line("AI/hierarchy policy OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
