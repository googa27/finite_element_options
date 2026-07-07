#!/usr/bin/env python3
"""Execute README Python and shell examples from an installed-package context."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

_EXECUTABLE_LANGUAGES = {"python", "py", "bash", "sh"}
_IGNORED_LANGUAGES = {"text", "txt", "mermaid", "math", "sympy-check"}
_FENCE_RE = re.compile(r"^```(?P<lang>[A-Za-z0-9_-]*)\s*$")
_FORBIDDEN_SHELL_PREFIXES = ("pip ", "python -m pip ", "pytest ", "streamlit ")


@dataclass(frozen=True, slots=True)
class CodeBlock:
    """A fenced README code block."""

    language: str
    code: str
    start_line: int


def iter_code_blocks(path: Path) -> Iterator[CodeBlock]:
    """Yield fenced code blocks from a Markdown file."""

    in_block = False
    language = ""
    start_line = 0
    body: list[str] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = _FENCE_RE.match(raw_line)
        if match and not in_block:
            in_block = True
            language = match.group("lang").lower()
            start_line = line_number
            body = []
            continue
        if raw_line == "```" and in_block:
            yield CodeBlock(language=language, code="\n".join(body), start_line=start_line)
            in_block = False
            language = ""
            body = []
            continue
        if in_block:
            body.append(raw_line)
    if in_block:
        raise ValueError(f"unclosed fenced code block starting at line {start_line}")


def _clean_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONWARNINGS"] = env.get("PYTHONWARNINGS", "default")
    python_dir = str(Path(sys.executable).parent)
    env["PATH"] = python_dir + os.pathsep + env.get("PATH", "")
    return env


def _run_python(block: CodeBlock, cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", block.code],
        cwd=cwd,
        env=_clean_environment(),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _validate_shell(block: CodeBlock) -> None:
    for line in block.code.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith(_FORBIDDEN_SHELL_PREFIXES):
            raise ValueError(
                f"README shell block at line {block.start_line} contains "
                f"environment-mutating or repo-only command: {stripped!r}"
            )


def _run_shell(block: CodeBlock, cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    _validate_shell(block)
    return subprocess.run(
        ["bash", "-euo", "pipefail", "-c", block.code],
        cwd=cwd,
        env=_clean_environment(),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def check_readme_examples(path: Path, *, timeout: int = 60) -> list[str]:
    """Execute supported README code examples and return failures."""

    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="feo-readme-examples-") as tmp:
        cwd = Path(tmp)
        for block in iter_code_blocks(path):
            language = block.language
            if language in _IGNORED_LANGUAGES:
                continue
            if language not in _EXECUTABLE_LANGUAGES:
                failures.append(
                    f"line {block.start_line}: unsupported or unlabeled code block "
                    f"language {language!r}; use an executable language or text"
                )
                continue
            try:
                result = (
                    _run_python(block, cwd, timeout)
                    if language in {"python", "py"}
                    else _run_shell(block, cwd, timeout)
                )
            except Exception as exc:  # noqa: BLE001 - report exact README block failure.
                failures.append(f"line {block.start_line}: {exc}")
                continue
            if result.returncode != 0:
                failures.append(
                    f"line {block.start_line}: example failed with "
                    f"exit {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
                )
    return failures


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("readme", type=Path)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args(argv)

    failures = check_readme_examples(args.readme, timeout=args.timeout)
    if failures:
        print("README example contract failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print(f"README examples passed: {args.readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
