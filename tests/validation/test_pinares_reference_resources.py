"""Public Pinares reference bytes must be readable by ordinary wheel consumers."""

from __future__ import annotations

from hashlib import sha256
from contextlib import redirect_stderr, redirect_stdout
import io
import importlib.metadata as metadata
import importlib.util
import json
from importlib.resources import as_file
import os
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from finite_element_options.validation import pinares_fixed_price_proxy as proxy

# Literal digests of the five original reviewed checkout snapshots at e946f38.
# These expectations do not depend on the package's resource or hash helpers.
REFERENCE_DIGESTS = {
    "PINARES_FEM_PROXY_PROBLEM_SPEC_PATH": "f7e48be4f88c572f6c4f11c0a3fdb12741dfaeebfe71a2da1762202096e6a092",
    "PINARES_FEM_PROXY_RESULT_EXPORT_PATH": "984e01a10d17693400b269a1fd935851abda34e253caeb2b36fd217d037698c1",
    "PINARES_FEM_PROVIDER_EVIDENCE_MANIFEST_PATH": "96c5d506eb3540e19a95a2679cd2c1645276ef936560d95e64f54ad7f87463ba",
    "PINARES_FEM_PROXY_UNSUPPORTED_SPEC_PATH": "02bd2bd3440dd14a76ab22b2732c6f673e239a014846da2bc918ba032bfbc1d3",
    "PINARES_QPS_FIXTURE_PATH": "f7e48be4f88c572f6c4f11c0a3fdb12741dfaeebfe71a2da1762202096e6a092",
}


class PinaresReferenceResources(unittest.TestCase):
    def test_existing_fixture_root_remains_a_public_traversable(self) -> None:
        root = proxy.PINARES_FEM_PROXY_FIXTURE_ROOT
        self.assertTrue(root.is_dir())
        for name, reference in (
            ("problem_spec.json", proxy.PINARES_FEM_PROXY_PROBLEM_SPEC_PATH),
            ("result_export.json", proxy.PINARES_FEM_PROXY_RESULT_EXPORT_PATH),
            (
                "provider_evidence_manifest.json",
                proxy.PINARES_FEM_PROVIDER_EVIDENCE_MANIFEST_PATH,
            ),
            (
                "unsupported_full_deal_problem_spec.json",
                proxy.PINARES_FEM_PROXY_UNSUPPORTED_SPEC_PATH,
            ),
        ):
            with self.subTest(name=name):
                self.assertEqual(
                    root.joinpath(name).read_bytes(), reference.read_bytes()
                )


    def test_original_public_reference_bytes_read_from_unrelated_directory(
        self,
    ) -> None:
        if sys.flags.isolated:
            self.assertTrue(
                Path(proxy.__file__).resolve().is_relative_to(Path(sys.prefix))
            )
            self.assertIsNone(importlib.util.find_spec("src"))
            distribution = metadata.distribution("finite-element-options")
            direct = json.loads(distribution.read_text("direct_url.json"))
            self.assertFalse(direct.get("dir_info", {}).get("editable", False))

        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as unrelated:
            try:
                os.chdir(unrelated)
                for name, expected in REFERENCE_DIGESTS.items():
                    with self.subTest(reference=name):
                        resource = getattr(proxy, name)
                        self.assertTrue(
                            resource.is_file(),
                            f"installed public reference is missing: {name}: {resource}",
                        )
                        contents = resource.read_bytes()
                        self.assertEqual(sha256(contents).hexdigest(), expected)
                        self.assertIsInstance(json.loads(contents), dict)
            finally:
                os.chdir(original_cwd)

    def test_writers_require_explicit_destinations_before_generation(self) -> None:
        def unexpected(*args, **kwargs):
            raise AssertionError("payload generation preceded destination validation")

        writers = (
            proxy.write_public_pinares_fixed_price_problem_spec,
            proxy.write_public_pinares_fixed_price_result_export,
            proxy.write_public_pinares_provider_evidence_manifest,
            proxy.write_public_pinares_unsupported_problem_spec,
            proxy.write_public_pinares_quant_problem_spec,
        )
        with (
            patch.object(proxy, "public_pinares_fixed_price_problem_spec", unexpected),
            patch.object(
                proxy, "run_public_pinares_fixed_price_proxy_fixture", unexpected
            ),
            patch.object(
                proxy, "public_pinares_full_deal_unsupported_problem_spec", unexpected
            ),
        ):
            for writer in writers:
                with self.subTest(writer=writer.__name__):
                    with self.assertRaises(ValueError):
                        writer()

    def test_refresh_requires_destination_before_numerical_work(self) -> None:
        def unexpected(*args, **kwargs):
            raise AssertionError("numerical work preceded destination validation")

        with patch.object(proxy, "_run_row", unexpected):
            with self.assertRaises(ValueError):
                proxy.run_public_pinares_fixed_price_proxy_fixture(refresh_exports=True)

    def test_package_destinations_are_refused_before_generation(self) -> None:
        def unexpected(*args, **kwargs):
            raise AssertionError("generation preceded package destination refusal")

        writers = (
            proxy.write_public_pinares_fixed_price_problem_spec,
            proxy.write_public_pinares_fixed_price_result_export,
            proxy.write_public_pinares_provider_evidence_manifest,
            proxy.write_public_pinares_unsupported_problem_spec,
            proxy.write_public_pinares_quant_problem_spec,
        )
        destination = Path(proxy.__file__).parent / "forbidden-output.json"
        with (
            patch.object(proxy, "public_pinares_fixed_price_problem_spec", unexpected),
            patch.object(
                proxy, "run_public_pinares_fixed_price_proxy_fixture", unexpected
            ),
            patch.object(
                proxy, "public_pinares_full_deal_unsupported_problem_spec", unexpected
            ),
        ):
            for writer in writers:
                with self.subTest(writer=writer.__name__):
                    with self.assertRaises(ValueError):
                        writer(destination)
        self.assertFalse(destination.exists())

    def test_unused_export_directory_refused_before_numerical_work(self) -> None:
        def unexpected(*args, **kwargs):
            raise AssertionError("numerical work preceded unused destination refusal")

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "must-not-be-created"
            with patch.object(proxy, "_run_row", unexpected):
                try:
                    with self.assertRaises(ValueError):
                        proxy.run_public_pinares_fixed_price_proxy_fixture(
                            export_directory=destination
                        )
                except TypeError as error:
                    self.fail(f"explicit export_directory contract is missing: {error}")
            self.assertFalse(destination.exists())

    def test_maintainer_requires_output_policy_before_solve(self) -> None:
        script_path = (
            Path(__file__).resolve().parents[2]
            / "scripts/export_pinares_fixed_price_proxy_fixture.py"
        )
        spec = importlib.util.spec_from_file_location("pinares_maintainer", script_path)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        def unexpected(*args, **kwargs):
            raise AssertionError("maintainer started solving without output policy")

        with patch.object(
            module, "run_public_pinares_fixed_price_proxy_fixture", unexpected
        ):
            try:
                with (
                    redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as refusal,
                ):
                    module.main([])
            except TypeError as error:
                self.fail(f"maintainer output-policy parser is missing: {error}")
            self.assertEqual(refusal.exception.code, 2)

    def test_reference_hardlink_aliases_refused_before_generation(self) -> None:
        def unexpected(*args, **kwargs):
            raise AssertionError(
                "generation preceded immutable reference alias refusal"
            )

        before = {name: getattr(proxy, name).read_bytes() for name in REFERENCE_DIGESTS}
        with tempfile.TemporaryDirectory() as directory:
            for name in REFERENCE_DIGESTS:
                with self.subTest(reference=name):
                    reference = getattr(proxy, name)
                    with as_file(reference) as path:
                        alias = Path(directory) / (name + ".json")
                        os.link(path, alias)
                        with patch.object(
                            proxy, "public_pinares_fixed_price_problem_spec", unexpected
                        ):
                            with self.assertRaises(ValueError):
                                proxy.write_public_pinares_fixed_price_problem_spec(
                                    alias
                                )
        self.assertEqual(
            before, {name: getattr(proxy, name).read_bytes() for name in before}
        )

    def test_real_generated_bundle_preserves_bytes_and_relocates(self) -> None:
        before = {name: getattr(proxy, name).read_bytes() for name in REFERENCE_DIGESTS}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bundle"
            report = proxy.run_public_pinares_fixed_price_proxy_fixture(
                refresh_exports=True, export_directory=root
            )
            self.assertTrue(report.converged)
            self.assertLessEqual(report.price_absolute_error_uf, 1.0)
            self.assertLessEqual(report.delta_absolute_error, 1e-3)
            self.assertLessEqual(report.gamma_absolute_error, 5e-6)
            moved = Path(directory) / "relocated"
            root.rename(moved)
            manifest = json.loads(
                (
                    moved
                    / "tests/fixtures/fem_pinares_fixed_price_proxy_v1/provider_evidence_manifest.json"
                ).read_bytes()
            )
            expected = dict(
                zip(
                    (
                        "problem_spec",
                        "result_export",
                        "provider_evidence_manifest",
                        "unsupported_problem_spec",
                        "quant_problem_spec",
                    ),
                    REFERENCE_DIGESTS,
                )
            )
            for key, name in expected.items():
                data = (moved / manifest["fixture_refs"][key]).read_bytes()
                self.assertEqual(sha256(data).hexdigest(), REFERENCE_DIGESTS[name])
            result = moved / manifest["fixture_refs"]["result_export"]
            result.write_bytes(b"caller-owned result retained")
            proxy.write_public_pinares_fixed_price_result_export(result)
            self.assertEqual(result.read_bytes(), b"caller-owned result retained")
            proxy.write_public_pinares_fixed_price_result_export(
                result, report=report, refresh=True
            )
            self.assertEqual(
                result.read_bytes(), before["PINARES_FEM_PROXY_RESULT_EXPORT_PATH"]
            )
        self.assertEqual(
            before, {name: getattr(proxy, name).read_bytes() for name in before}
        )

    def test_fifth_refresh_alias_is_refused_before_solve_or_other_writes(self) -> None:
        def unexpected(*args, **kwargs):
            raise AssertionError("partial refresh admission started numerical work")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            last = (
                root
                / "tests/fixtures/quant_problem_specs/pinares_fixed_price_proxy.json"
            )
            last.parent.mkdir(parents=True)
            with as_file(proxy.PINARES_QPS_FIXTURE_PATH) as reference:
                os.link(reference, last)
                with patch.object(proxy, "_run_row", unexpected):
                    with self.assertRaises(ValueError):
                        proxy.run_public_pinares_fixed_price_proxy_fixture(
                            refresh_exports=True, export_directory=root
                        )
                self.assertFalse(
                    (root / "tests/fixtures/fem_pinares_fixed_price_proxy_v1").exists()
                )
                self.assertEqual(
                    sha256(reference.read_bytes()).hexdigest(),
                    REFERENCE_DIGESTS["PINARES_QPS_FIXTURE_PATH"],
                )

    def test_distinct_refresh_outputs_must_not_alias_each_other(self) -> None:
        def unexpected(*args, **kwargs):
            raise AssertionError("aliased refresh outputs reached numerical work")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "tests/fixtures/fem_pinares_fixed_price_proxy_v1"
            fixture.mkdir(parents=True)
            spec = fixture / "problem_spec.json"
            result = fixture / "result_export.json"
            spec.write_bytes(b"caller-owned shared inode")
            os.link(spec, result)
            with patch.object(proxy, "_run_row", unexpected):
                with self.assertRaises(ValueError):
                    proxy.run_public_pinares_fixed_price_proxy_fixture(
                        refresh_exports=True, export_directory=root
                    )
            self.assertEqual(spec.read_bytes(), b"caller-owned shared inode")
            self.assertEqual(result.read_bytes(), b"caller-owned shared inode")
            self.assertFalse((fixture / "provider_evidence_manifest.json").exists())

    def test_invalid_last_output_parent_refused_before_numerical_work(self) -> None:
        def unexpected(*args, **kwargs):
            raise AssertionError("invalid refresh parent reached numerical work")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blocked = root / "tests/fixtures/quant_problem_specs"
            blocked.parent.mkdir(parents=True)
            blocked.write_bytes(b"caller-owned ancestor file")
            with patch.object(proxy, "_run_row", unexpected):
                with self.assertRaises(ValueError):
                    proxy.run_public_pinares_fixed_price_proxy_fixture(
                        refresh_exports=True, export_directory=root
                    )
            self.assertEqual(blocked.read_bytes(), b"caller-owned ancestor file")
            self.assertFalse(
                (root / "tests/fixtures/fem_pinares_fixed_price_proxy_v1").exists()
            )

    def test_real_maintainer_cli_exports_only_caller_bundle(self) -> None:
        script = (
            Path(__file__).resolve().parents[2]
            / "scripts/export_pinares_fixed_price_proxy_fixture.py"
        )
        before = {name: getattr(proxy, name).read_bytes() for name in REFERENCE_DIGESTS}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "independent"
            result = subprocess.run(
                [sys.executable, "-I", "-B", str(script), "--output-dir", str(root)],
                cwd=directory,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            receipt = json.loads(result.stdout)
            self.assertTrue(receipt["converged"])
            self.assertEqual(len(receipt["generated"]), 5)
            self.assertFalse((root / "src").exists())
            for item in receipt["generated"]:
                path = Path(item["path"])
                self.assertTrue(path.is_relative_to(root))
                self.assertEqual(sha256(path.read_bytes()).hexdigest(), item["sha256"])
                self.assertIn(item["sha256"], REFERENCE_DIGESTS.values())
        self.assertEqual(
            before, {name: getattr(proxy, name).read_bytes() for name in before}
        )

    def test_explicit_maintainer_publication_updates_both_temporary_mirrors(
        self,
    ) -> None:
        script = (
            Path(__file__).resolve().parents[2]
            / "scripts/export_pinares_fixed_price_proxy_fixture.py"
        )
        spec = importlib.util.spec_from_file_location(
            "pinares_maintainer_publication", script
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        before = {name: getattr(proxy, name).read_bytes() for name in REFERENCE_DIGESTS}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = io.StringIO()
            with patch.object(module, "REPO_ROOT", root), redirect_stdout(output):
                module.main(["--publish-canonical"])
            receipt = json.loads(output.getvalue())
            self.assertTrue(receipt["converged"])
            self.assertEqual(len(receipt["generated"]), 10)
            for relative, digest in (
                (
                    "fem_pinares_fixed_price_proxy_v1/problem_spec.json",
                    REFERENCE_DIGESTS["PINARES_FEM_PROXY_PROBLEM_SPEC_PATH"],
                ),
                (
                    "fem_pinares_fixed_price_proxy_v1/result_export.json",
                    REFERENCE_DIGESTS["PINARES_FEM_PROXY_RESULT_EXPORT_PATH"],
                ),
                (
                    "fem_pinares_fixed_price_proxy_v1/provider_evidence_manifest.json",
                    REFERENCE_DIGESTS["PINARES_FEM_PROVIDER_EVIDENCE_MANIFEST_PATH"],
                ),
                (
                    "fem_pinares_fixed_price_proxy_v1/unsupported_full_deal_problem_spec.json",
                    REFERENCE_DIGESTS["PINARES_FEM_PROXY_UNSUPPORTED_SPEC_PATH"],
                ),
                (
                    "quant_problem_specs/pinares_fixed_price_proxy.json",
                    REFERENCE_DIGESTS["PINARES_QPS_FIXTURE_PATH"],
                ),
            ):
                checkout = root / "tests/fixtures" / relative
                packaged = (
                    root
                    / "src/finite_element_options/validation/evidence/reference_data"
                    / relative
                )
                self.assertEqual(checkout.read_bytes(), packaged.read_bytes())
                self.assertEqual(sha256(packaged.read_bytes()).hexdigest(), digest)
        self.assertEqual(
            before, {name: getattr(proxy, name).read_bytes() for name in before}
        )


if __name__ == "__main__":
    unittest.main()
