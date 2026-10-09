#!/usr/bin/env python3
"""Exercise real parsed-workflow mutations against the issue173 runtime gate."""

from __future__ import annotations

import copy
import unittest

from check_actions_runtime_contract import (
    ROOT,
    load_contract,
    validate_runtime_contract,
)


class RuntimeContractTests(unittest.TestCase):
    """Reject unsafe or unreviewed changes without replacing the validator."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.policy, cls.workflows = load_contract(ROOT)

    def test_reviewed_repository_workflows_pass(self) -> None:
        self.assertEqual(validate_runtime_contract(self.policy, self.workflows), [])

    def test_workflow_inventory_is_exact(self) -> None:
        for path in self.workflows:
            with self.subTest(path=path):
                mutated = copy.deepcopy(self.workflows)
                del mutated[path]
                self.assertTrue(validate_runtime_contract(self.policy, mutated))
        mutated = copy.deepcopy(self.workflows)
        mutated[".github/workflows/unreviewed.yml"] = copy.deepcopy(
            next(iter(mutated.values()))
        )
        self.assertTrue(validate_runtime_contract(self.policy, mutated))

    def test_every_job_rejects_runner_timeout_and_permission_drift(self) -> None:
        for path, workflow in self.workflows.items():
            for name in workflow["jobs"]:
                for field, value in (
                    ("runs-on", "ubuntu-latest"),
                    ("runs-on", "ubuntu-26.04"),
                    ("runs-on", ["self-hosted"]),
                    ("timeout-minutes", None),
                    ("timeout-minutes", True),
                    ("timeout-minutes", 0),
                    ("timeout-minutes", self.policy["maximum_job_minutes"] + 1),
                    ("permissions", {"contents": "write"}),
                ):
                    with self.subTest(path=path, job=name, field=field, value=value):
                        mutated = copy.deepcopy(self.workflows)
                        mutated[path]["jobs"][name][field] = value
                        self.assertTrue(validate_runtime_contract(self.policy, mutated))

    def test_each_action_rejects_mutable_stale_and_unknown_refs(self) -> None:
        for path, workflow in self.workflows.items():
            for name, job in workflow["jobs"].items():
                for index, step in enumerate(job["steps"]):
                    if "uses" not in step:
                        continue
                    action = step["uses"].split("@")[0]
                    for ref in (f"{action}@v7", f"{action}@{'0' * 40}", "./local"):
                        with self.subTest(path=path, job=name, index=index, ref=ref):
                            mutated = copy.deepcopy(self.workflows)
                            mutated[path]["jobs"][name]["steps"][index]["uses"] = ref
                            self.assertTrue(
                                validate_runtime_contract(self.policy, mutated)
                            )

    def test_each_action_retains_sensitive_input_contracts(self) -> None:
        replacements = {
            "actions/checkout": (
                ("persist-credentials", None),
                ("persist-credentials", True),
                ("persist-credentials", "false"),
            ),
            "actions/setup-python": (
                ("python-version", None),
                ("pip-install", "pytest"),
            ),
            "actions/upload-artifact": (
                ("archive", False),
                ("if-no-files-found", "warn"),
            ),
        }
        for path, workflow in self.workflows.items():
            for name, job in workflow["jobs"].items():
                for index, step in enumerate(job["steps"]):
                    if "uses" not in step:
                        continue
                    for field, value in replacements[step["uses"].split("@")[0]]:
                        with self.subTest(
                            path=path, job=name, index=index, field=field
                        ):
                            mutated = copy.deepcopy(self.workflows)
                            target = mutated[path]["jobs"][name]["steps"][index]
                            target.setdefault("with", {})[field] = value
                            self.assertTrue(
                                validate_runtime_contract(self.policy, mutated)
                            )

    def test_runtime_override_cannot_hide_old_runtime(self) -> None:
        for path in self.workflows:
            for variable in (
                "ACTIONS_ALLOW_USE_UNSECURE_NODE_VERSION",
                "FORCE_JAVASCRIPT_ACTIONS_TO_NODE24",
            ):
                with self.subTest(path=path, variable=variable):
                    mutated = copy.deepcopy(self.workflows)
                    mutated[path]["env"] = {variable: "true"}
                    self.assertTrue(validate_runtime_contract(self.policy, mutated))

    def test_runtime_gate_cannot_be_omitted_or_suppressed(self) -> None:
        path = ".github/workflows/ai-hierarchy-policy.yml"
        for replacement in ("", "# disabled", "echo runtime gate", "true"):
            with self.subTest(replacement=replacement):
                mutated = copy.deepcopy(self.workflows)
                for step in mutated[path]["jobs"]["policy"]["steps"]:
                    if "scripts/check_actions_runtime_contract.py" in step.get(
                        "run", ""
                    ):
                        step["run"] = replacement
                self.assertTrue(validate_runtime_contract(self.policy, mutated))
        for field, value in (("if", False), ("continue-on-error", True)):
            with self.subTest(field=field):
                mutated = copy.deepcopy(self.workflows)
                mutated[path]["jobs"]["policy"][field] = value
                self.assertTrue(validate_runtime_contract(self.policy, mutated))
                mutated = copy.deepcopy(self.workflows)
                mutated[path]["jobs"]["policy"]["steps"][0][field] = value
                self.assertTrue(validate_runtime_contract(self.policy, mutated))

    def test_top_level_permissions_cannot_expand(self) -> None:
        for path in self.workflows:
            with self.subTest(path=path):
                mutated = copy.deepcopy(self.workflows)
                mutated[path]["permissions"] = "write-all"
                self.assertTrue(validate_runtime_contract(self.policy, mutated))


if __name__ == "__main__":
    unittest.main(verbosity=2)
