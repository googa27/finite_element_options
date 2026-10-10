# Packaging subprocess isolation implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Root implements and performs a dedicated, honestly labelled authoring-root review; the user's explicit root-alone authorization prohibits delegation.

**Goal:** Satisfy issue168 by isolating wheel installation/probes and retaining real bootstrap failure diagnostics.

**Architecture:** Test-only packaging utilities own child environment preparation and checked execution. Both existing normal-wheel consumers use the same boundary, with isolated environments and byte/origin checks. Runtime APIs, numerical bodies, package members and optional-profile contracts stay unchanged.

**Tech Stack:** Python3.11/3.12, stdlib subprocess/venv/ensurepip/importlib.metadata/zipfile, existing packaging and pytest dependencies, existing hosted CI.

**Spec:** https://github.com/googa27/finite_element_options/issues/168 ; parent43 ; canonical AGENTS.md, docs/PRD.md and docs/ARCHITECTURE.md.

## Global constraints

- Preserve ordinary wheel installation with --no-deps in the first consumer.
- Do not force reinstall, ignore installed distributions, auto-retry, suppress failures or widen skips.
- Preserve required private cache/temp/index/certificate configuration while removing PYTHONPATH, PYTHONHOME and VIRTUAL_ENV selection.
- Preserve original numerical algorithms, coefficients/tolerances, fixture/compiler identities, dependencies, action identities, permissions and scientific maturity.
- Run all existing packaging/architecture/profile gates; prove reviewed normal wheel bytes and origins on3.11/3.12 and actual resulting default.
- Storage ruling: no local source/.git/environment/cache/tmpfs/artifact writes below the installed floor. Use native Git commits/PRs and hosted runners; the native issue/PR ledger replaces local .superpowers scratch under this hold. This is not a normal disk-wrapper admission or SSD recovery.
- Workflow ruling: explicit user autonomous decisions/root-alone authorization supersedes skill approval/delegation menus. Review is authoring-root, never fabricated human or different-agent approval. Ordinary integration remains contingent on current enforced gates.

## Review focus

1. Both implicit and explicit child environments may contain poisoned source metadata: exercise both.
2. A private cache, temporary root, index or certificate may be required: preserve exact caller values and never mutate the caller mapping.
3. A real child may fail with distinct stdout/stderr: retain argv, cwd and its actual nonzero exit without retries.
4. Nested venv bootstrap can hide ensurepip output: expose the actual bootstrap subprocess and exercise its real failure path.
5. Same-version metadata or parent editable installation can satisfy pip incorrectly: use isolated target environments, an unrelated working directory and independent wheel-member/normal-origin assertions.

### Task1: Checked child boundary

**Files:** Create tests/test_packaging_subprocesses.py and tests/packaging_support.py; modify tests/test_packaging_contract.py and .github/workflows/ci.yml.

**Interfaces:** Preserve _run(command: list[str], *, cwd: Path, env: dict[str,str] | None = None) -> str. A shared test-only run_checked has the same signature and returns concatenated captured streams on success.

- [ ] Add real implicit/explicit PYTHONPATH poisoning, invalid PYTHONHOME, preserved private configuration and literal child-exit/ensurepip diagnostic tests.
- [ ] Run python -m pytest -q -o addopts='' tests/test_packaging_subprocesses.py on both package-job minors. Expected RED: source poison remains visible, invalid home breaks startup, and existing failures lack structured context; the configuration positive remains valid.
- [ ] Implement copied child environment minus the three Python selectors and structured failure JSON with argv/cwd/returncode/stdout/stderr. Preserve original successful output and all existing callers.
- [ ] Repeat the exact tests on both minors; expected all controls pass with the original failing children still genuinely nonzero.
- [ ] Commit the minimal boundary repair with native RED/GREEN receipts.

### Task2: Real factory and normal-wheel authority

**Files:** Extend the helper and tests, modify both existing installed-wheel consumers.

**Interfaces:** create_venv(path: Path, *, cwd: Path, ensurepip_args: tuple[str,...] = ("--upgrade","--default-pip")) -> Path creates an isolated venv without pip, then runs its actual interpreter's ensurepip explicitly. core_requirements(wheel: Path) -> list[str] uses standard wheel METADATA and packaging.Requirement marker evaluation for core requirements.

- [ ] Add the real same-version poisoned consumer test and an intentional factory bootstrap failure before factory/consumer changes. Missing factory is an explicitly labelled missing-interface RED, not a reproduced filesystem failure.
- [ ] Use a real invalid ensurepip option to verify the distinct original diagnostic, actual argv/cwd/exit2 and preserved partial environment with no installed pip or retry.
- [ ] Create isolated targets. Install first consumer's actual core requirements separately, then install the reviewed wheel once with --no-deps; retain the second consumer's ordinary dependency installation.
- [ ] Execute original import/optional-refusal scripts with isolated Python from an unrelated directory. Verify all wheel package bytes, normal direct_url and sys.prefix origins before/after; independently poison same-version metadata and reject checkout authority.
- [ ] Run complete packaging/architecture and both declared minor jobs. Expected all original controls and new behavioral controls pass; preserve original source/installed evidence boundaries.

### Task3: Governance, dedicated review and integration

**Files:** Update AGENTS.md, docs/PRD.md, docs/ARCHITECTURE.md and additive docs/ARCHITECTURE.yaml tests.packaging_observer ownership. Keep current architecture limits and exceptions unchanged.

- [ ] Record exact helper/control ownership and3.11/3.12 commands; no prose-grep tests.
- [ ] Run all original CI/source/profile/numerical gates and inspect complete current automated findings, annotations and logs.
- [ ] Perform a dedicated authoring-root review of every exact changed patch; concrete dispositions for all findings.
- [ ] Refresh actual head/base/rules and ordinarily merge only when acceptance is proved. Repeat actual resulting-default source/normal-installed/profile acceptance before closing168 or setting Done.
- [ ] Preserve useful histories and retire only the completed owned branch after ancestry/default/current-dependency proof. Keep every broader portfolio/scientific obligation ACTIVE.

## Execution ledger

- Pre-flight: Task2 consumes Task1's exact checked-command interface; Task3 documents and verifies both. Interfaces agree.
- Plan self-review: all issue168 acceptance requirements map to Tasks1-3; no numerical or scientific promotion is included.
- Initial base: master4f18785af418d241242baf013fb053dba78fcf32.
- No repair, new installed acceptance or merge is established by this plan.
