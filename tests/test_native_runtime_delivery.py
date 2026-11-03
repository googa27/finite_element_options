"""Keep the native visualization archive handoff fail closed.

These inspect executable shell commands and rejection controls. They do not
replace real hosted packing, relocation, rendering or complete binary readback.
"""

from pathlib import Path
import re
import shlex
import subprocess
import textwrap
import unittest


WORKFLOW = (
    Path(__file__).resolve().parents[1] / ".github/workflows/chatgpt-pde-supplement.yml"
)


def check_delivery_contract(source):
    """Reject missing visualization, unsafe ordering and compression drift."""
    job = source.split("\n  relocated:\n", 1)[1].split("\n  wheels:\n", 1)[0]
    block = job.split("name: Execute producer then pack", 1)[1]
    script = textwrap.dedent(
        block.split("run: |\n", 1)[1].split("\n      - name:", 1)[0]
    )
    syntax = subprocess.run(
        ["bash", "-n"], input=script, text=True, capture_output=True, check=False
    )
    assert syntax.returncode == 0, syntax.stderr
    commands = [shlex.split(line) for line in script.splitlines()]
    create = next(words for words in commands if words[:2] == ["conda", "create"])
    assert "matplotlib-base=3.11.2" in create, "portable plots need Matplotlib"
    pack = next(
        words
        for words in commands
        if words and words[0].endswith("/conda-pack") and "--prefix" in words
    )
    assert pack[pack.index("--format") + 1] == "tar.xz"
    assert pack[pack.index("--compress-level") + 1] == "6"
    assert pack[pack.index("--n-threads") + 1] == "2"
    assert pack[pack.index("--output") + 1] == "$RUNNER_TEMP/native-runtime.tar.xz"
    assert not any(word.startswith("--ignore-") for word in pack)
    original_export = (
        '"$NATIVE_PREFIX/bin/python" -I scripts/validate_native_pde.py export '
        '"$EVIDENCE/original/report.json"'
    )
    relocated_export = (
        '"$CONSUMER_PREFIX/bin/python" -I validate_native_pde.py export '
        '"$EVIDENCE/relocated/report.json"'
    )
    compare = '"$CONSUMER_PREFIX/bin/python" -I validate_native_pde.py compare '
    move = (
        'mv "$RUNNER_TEMP/native-runtime.tar.xz" '
        '"$RUNTIME_EXPORT/native-runtime.tar.xz"'
    )
    order = [
        original_export,
        '"$NATIVE_PREFIX/bin/conda-pack" --prefix',
        "archive-info",
        'test ! -e "$NATIVE_PREFIX"',
        'tar -xJf "$RUNNER_TEMP/native-runtime.tar.xz"',
        'env -u PYTHONPATH -u PYTHONHOME "$CONSUMER_PREFIX/bin/python" -E -s '
        '"$CONSUMER_PREFIX/bin/conda-unpack"',
        relocated_export,
        compare,
        move,
        "sha256sum --check SHA256SUMS",
    ]
    positions = [script.index(command) for command in order]
    assert positions == sorted(positions), "export/replay/delivery order changed"
    assert 'mv "$NATIVE_PREFIX" "$RUNNER_TEMP/native-producer-disabled"' in script
    assert 'cd "$CONSUMER_INPUT"' in script
    assert 'export MPLCONFIGDIR="$RUNNER_TEMP/producer-mpl"' in script
    assert 'export MPLCONFIGDIR="$RUNNER_TEMP/consumer-mpl"' in script
    for name in ("validate_native_pde.py", "native_pde_artifacts.py"):
        for operation in ("cp", "cmp"):
            command = f'{operation} scripts/{name} "$CONSUMER_INPUT/{name}"'
            assert command in script, "the entire exact source bundle is required"
            assert script.index(command) < script.index(relocated_export)
    upload = job.split("name: Deliver the exact tested native archive", 1)[1]
    assert "compression-level: 0" in upload
    assert "retention-days: 1" in upload
    assert "if-no-files-found: error" in upload
    assert not re.search(r"^\s*(if|continue-on-error):", upload, re.MULTILINE)
    assert "native-runtime.tar.gz" not in job
    return script


class NativeRuntimeDeliveryTests(unittest.TestCase):
    """Pin the concrete handoff, then prove omissions are refused."""

    def test_actual_workflow_proves_visualization_before_delivering_xz(self):
        check_delivery_contract(WORKFLOW.read_text())

    def test_missing_plotting_or_changed_pack_compression_is_refused(self):
        source = WORKFLOW.read_text()
        for before, after in [
            (" matplotlib-base=3.11.2", ""),
            ("--format tar.xz", "--format tar.gz"),
            ("--compress-level 6", "--compress-level 9"),
            ("--n-threads 2", "--n-threads -1"),
        ]:
            with self.subTest(mutation=after):
                self.assertIn(before, source)
                with self.assertRaises((AssertionError, ValueError)):
                    check_delivery_contract(source.replace(before, after))

    def test_missing_exports_or_producer_isolation_are_refused(self):
        source = WORKFLOW.read_text()
        for command in [
            '"$NATIVE_PREFIX/bin/python" -I scripts/validate_native_pde.py export',
            '"$CONSUMER_PREFIX/bin/python" -I validate_native_pde.py export',
            'test ! -e "$NATIVE_PREFIX"',
            'mv "$NATIVE_PREFIX" "$RUNNER_TEMP/native-producer-disabled"',
            'env -u PYTHONPATH -u PYTHONHOME "$CONSUMER_PREFIX/bin/python" -E -s',
            "cp scripts/native_pde_artifacts.py",
            "cmp scripts/native_pde_artifacts.py",
        ]:
            with self.subTest(command=command):
                self.assertIn(command, source)
                altered = "\n".join(
                    line for line in source.splitlines() if command not in line
                )
                with self.assertRaises((AssertionError, ValueError)):
                    check_delivery_contract(altered)

    def test_delivery_before_replay_or_upload_suppression_is_refused(self):
        source = WORKFLOW.read_text()
        move = (
            'mv "$RUNNER_TEMP/native-runtime.tar.xz" '
            '"$RUNTIME_EXPORT/native-runtime.tar.xz"'
        )
        self.assertIn(move, source)
        altered = source.replace(move, "")
        altered = altered.replace(
            '          cd "$CONSUMER_INPUT"',
            "          " + move + '\n          cd "$CONSUMER_INPUT"',
        )
        with self.assertRaisesRegex(AssertionError, "order"):
            check_delivery_contract(altered)
        for before, after in [
            ("compression-level: 0", "compression-level: 6"),
            ("retention-days: 1", "retention-days: 90"),
            ("if-no-files-found: error", "if-no-files-found: ignore"),
        ]:
            with self.subTest(mutation=after), self.assertRaises(AssertionError):
                check_delivery_contract(source.replace(before, after))
        marker = "name: Deliver the exact tested native archive within the measured cap"
        for addition in ["if: always()", "continue-on-error: true"]:
            with self.subTest(addition=addition), self.assertRaises(AssertionError):
                check_delivery_contract(
                    source.replace(marker, marker + "\n        " + addition)
                )


if __name__ == "__main__":
    unittest.main()
