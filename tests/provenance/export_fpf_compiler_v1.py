"""Export public compiler.v1 provenance from an isolated normal FPF wheel.

This is a test-only producer observer, not FEM admission or scientific evidence.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import importlib.metadata as metadata
import importlib.util
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

PRODUCER_HEAD = "968780defc687cd9296badf980811459e7c14b28"
PRODUCER_TREE = "95edd8dd36e1b0a10d2b0414c87f1215e52ad936"
SOURCE_IR = "sha256:5ab53779a5e322284a6cb18b22302c119f22bc740659aedf1c07823529d68a47"
COMPILED_V1 = "sha256:b449647e7f8deea870b0e8fbe0cfd4355040a53b9853d69c17443f0b3a6d9cb2"
HISTORY_BYTES = "2071b2f2a1651cf1e4f97b09f05e6b6a09c6af9a2e6758b1fc6b53bdc15757a9"


def encoded(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def identity(value: object) -> object:
    if isinstance(value, dict):
        return {key: identity(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [identity(item) for item in value]
    return value


def git(source: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source), *arguments], text=True
    ).strip()


def main() -> None:
    assert sys.flags.isolated and not sys.flags.optimize
    source, wheel, output = (Path(item).resolve() for item in sys.argv[1:])
    prefix = Path(sys.prefix).resolve()
    assert not Path.cwd().resolve().is_relative_to(source)
    assert all(
        not Path(item).resolve().is_relative_to(source)
        for item in sys.path
        if item
    )
    assert importlib.util.find_spec("src") is None
    assert git(source, "rev-parse", "HEAD") == PRODUCER_HEAD
    assert git(source, "rev-parse", "HEAD^{tree}") == PRODUCER_TREE
    assert not git(source, "status", "--porcelain")
    distribution = metadata.distribution("financial_problem_formulations")
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    assert direct_url["url"] == wheel.as_uri()
    assert not direct_url.get("dir_info", {}).get("editable", False)
    with zipfile.ZipFile(wheel) as archive:
        names = [item.filename for item in archive.infolist() if not item.is_dir()]
        assert len(names) == len(set(names))
        members = {name: archive.read(name) for name in names}
    package_members = {
        name: value
        for name, value in members.items()
        if name.startswith("financial_problem_formulations/")
    }
    assert package_members
    record_names = [name for name in members if name.endswith(".dist-info/RECORD")]
    assert len(record_names) == 1
    rows = list(csv.reader(io.StringIO(members[record_names[0]].decode())))
    assert len(rows) == len(members)
    assert {row[0] for row in rows} == set(members)
    for name, digest, size in rows:
        value = members[name]
        if name == record_names[0]:
            assert digest == size == ""
        else:
            actual = base64.urlsafe_b64encode(hashlib.sha256(value).digest()).rstrip(b"=").decode()
            assert digest == "sha256=" + actual
            assert int(size) == len(value)

    def verify_installed() -> None:
        for name, value in package_members.items():
            installed = Path(distribution.locate_file(name)).resolve()
            assert installed.is_relative_to(prefix)
            assert installed.read_bytes() == value
            assert (source / "src" / name).read_bytes() == value
        for name, module in tuple(sys.modules.items()):
            if name == "financial_problem_formulations" or name.startswith(
                "financial_problem_formulations."
            ):
                assert module.__file__
                assert Path(module.__file__).resolve().is_relative_to(prefix)

    verify_installed()
    from financial_problem_formulations.algebra.pde_ir.compiler import compile_pde_ir_json

    source_path = source / "tests/fixtures/pde_ir/black_scholes_call_v0.json"
    source_bytes = source_path.read_bytes()
    source_payload = json.loads(source_bytes)
    assert source_payload["canonical_hash"] == SOURCE_IR
    source_identity = {
        key: value for key, value in source_payload.items() if key != "canonical_hash"
    }
    assert "sha256:" + hashlib.sha256(encoded(identity(source_identity))).hexdigest() == SOURCE_IR
    current = compile_pde_ir_json(source_path).to_dict()
    assert current == compile_pde_ir_json(source_path).to_dict()
    assert current["accepted"] is True
    compiled = current["compiled_operator"]
    assert compiled["compiled_hash"] == COMPILED_V1
    assert compiled["source_ir_canonical_hash"] == SOURCE_IR
    assert compiled["compiler_evidence"]["compiler_version"] == "pde_ir_symbolic_compiler.v1"
    assert compiled["compiler_evidence"]["grammar_version"] == "restricted_math_ast.v0"
    unhashed = {key: value for key, value in compiled.items() if key != "compiled_hash"}
    assert "sha256:" + hashlib.sha256(encoded(identity(unhashed))).hexdigest() == COMPILED_V1
    history_path = source / "tests/fixtures/compiler_history/v0/black_scholes_call_v0.json"
    history_bytes = history_path.read_bytes()
    assert hashlib.sha256(history_bytes).hexdigest() == HISTORY_BYTES
    old = json.loads(history_bytes)["compiled_operator"]
    assert old["compiled_hash"] == "sha256:970088e5dcb16535edfd230bfe992ea7eb68aede901c7b543682b39f1a5ac32e"
    excluded = {"compiled_hash", "compiler_evidence", "expressions"}
    assert {key: value for key, value in old.items() if key not in excluded} == {
        key: value for key, value in compiled.items() if key not in excluded
    }
    evidence = dict(old["compiler_evidence"])
    evidence["compiler_version"] = "pde_ir_symbolic_compiler.v1"
    assert evidence == compiled["compiler_evidence"]
    previous = {item["path"]: item for item in old["expressions"]}
    emitted = {item["path"]: item for item in compiled["expressions"]}
    assert previous.keys() == emitted.keys()
    changes = []
    for path, item in emitted.items():
        prior = previous[path]
        fields = sorted(key for key in item if item[key] != prior[key])
        assert set(item) == set(prior)
        assert fields in ([], ["expression_hash", "normalized"])
        if fields:
            changes.append(
                {
                    "path": path,
                    "old_normalized": prior["normalized"],
                    "new_normalized": item["normalized"],
                    "old_expression_hash": prior["expression_hash"],
                    "new_expression_hash": item["expression_hash"],
                }
            )
    assert len(changes) == 2
    verify_installed()
    assert not git(source, "status", "--porcelain")
    result_bytes = encoded(current) + b"\n"
    provenance = {
        "privacy": "public-synthetic",
        "scope": "normal installed producer only; no FEM admission/solve, full release or science claim",
        "producer_repository": "googa27/financial_problem_formulations",
        "producer_head": PRODUCER_HEAD,
        "producer_tree": PRODUCER_TREE,
        "distribution_version": distribution.version,
        "python": sys.version,
        "wheel_name": wheel.name,
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "wheel_members": len(members),
        "record_rows": len(rows),
        "package_members": len(package_members),
        "source_input_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_ir_canonical_hash": SOURCE_IR,
        "compiled_hash": COMPILED_V1,
        "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "historical_result_sha256": HISTORY_BYTES,
        "expression_changes": changes,
        "source_free_execution": True,
        "installed_source_wheel_pre_post_identity": True,
        "member_sha256": {
            name: hashlib.sha256(value).hexdigest()
            for name, value in sorted(package_members.items())
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "black_scholes_call_compiler_v1.json").write_bytes(result_bytes)
    (output / "black_scholes_call_source_ir.json").write_bytes(source_bytes)
    (output / "PROVENANCE.json").write_bytes(encoded(provenance) + b"\n")
    print(json.dumps(provenance, sort_keys=True))


if __name__ == "__main__":
    main()
