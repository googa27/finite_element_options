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

- Task1 RED: exact6ad8e276, hosted CI38010993449 package3.11 job114090707003 6failed/1passed0.29s nativeexit1 raw217285B/SHA5d8e0b9d1381d7211c23a4775889ec064c78fda27858682ead420cd4e4377367; package3.12 job114090707124 6failed/1passed0.42s nativeexit1 raw216034B/SHAef6058368ea6d29235962525dd06ae741b2d7d61dc8fb29cd9682c738d1d4f74. Both source-poison cases observed module/distributionTrue; both invalid-home cases failed real startup; both diagnostic controls lacked context. Private configuration positive passed. Actual bootstrap direct refusal is separate from the pending factory path.
- Task1 repair: only copied-env PYTHONPATH/PYTHONHOME isolation and actual child diagnostic context. VIRTUAL_ENV selection and factory/normal-wheel authority remain Task2; no Task1 GREEN or owner completion claimed yet.

- Task1 focused GREEN: immutable10052d6e, CI38011189423 package3.11 job114091335215 7passed0.32s/normal157/control2+5+14;3.12 job114091335261 7passed0.31s/normal157/control2+5+14. Both actualsuccess, rawlogsSHA3feaff0a3e039810ad5920238418f9d3e361d272f272f23f4ab61331cffc81ea/c00fb47989c323ad7ccb78dc92088fa11e415f228d7caab7e2f5767a6d121882. Actual checkouta612b30c parent/tree binding independently verified against10052. Fullsource remains pending/formatting issue, not fullbranch acceptance.
- Original6ad8 source job114090707102 actually failed only new-test formatting (Ruff lint passed); rawlog71090B/SHAe2771600c36196492dfac6c48b62e981f143834ad5ad2c0d754cd05e187aec95. Test formatting corrected in this successor; no original failure rewritten.
- Task2 tests-first: VIRTUAL_ENV leak controls, missing explicit factory controls and same-version metadata/normal-origin consumer control added before factory/consumer repair. Package observer intentionally receives the ordinary editable core install to exercise the same parent-distribution trap as the source suite; normal release target remains isolated and independent. No extra declaration or dependency pin changed. Factory RED is labelled missing-interface; real deliberate ensurepip failure is required after implementation.

- Task2 initial RED d3900044: actual3.11/3.12 each4failed8passed, jobs114092132246/114092132259 logsSHA0c344e02d42b95dc17b7826f78e3f8140b753f1ef92bf932fe28e7437fc7290c/30206b7d4184c782a375082ca1088f8fda6962401a9997e6ad19a1e5dbe4c44b. Only two actual VIRTUAL_ENV leaks and two explicitly missing-factory outcomes failed. Same-version normal-wheel control already passed after Task1: do not claim a reproduced parent-editable origin defect.
- Ruling: exercise actual user-site leakage through the existing consumer before changing its system-site policy. New input is caller-owned PYTHONUSERBASE plus one private synthetic marker, never a shared environment edit. A non-isolated target subprocess must not import that parent marker. Added before consumer/factory implementation.
- Original d390 source formatting failure retained (job114092132168/logSHAe8d084f5fb7dd9e90e79db3128aa6a877f625776056d639ab640516abeeb6335). Required pinned Ruff check now prints the actual diff on failure and still exits nonzero; no source rewriting or gate bypass occurs.

- Task2 behavioral RED275b4b9c: CI38011719804 actual package3.11 job114092999126 5failed8passed12.75s/log198516B/SHAf825c15845cf8bfbd838723392fde7efc1166598ebd5511b7fbeef019ce54ff9;3.12 job114092999087 5failed8passed11.22s/log197262B/SHAe317ac66eab250a55e4845ae878d042cde9e1a3ddc07897d4329c49138336bfa. Both actual target environments expose the parent-user marker. Two VIRTUAL_ENV selectors leak; two factory interfaces are still explicitly absent. Same-version normal-wheel control remains passing, not a new origin failure.
- Pinned Ruff0.12.12 actual source-job114092999189 diff/log73204B/SHA3905018ebfa1c3a0845ed33f649030afa29f88bafa497c4928324dbc6fc0d76f supplied the two exact formatting corrections; no unpinned formatter or local write was used.
- Task2 repair now adds the real isolated venv/explicit ensurepip factory, removes VIRTUAL_ENV, installs actual core METADATA requirements before the first single no-dependencies wheel, and surrounds both unchanged original consumer scripts with normal byte/origin checks. New execution pending: no factory GREEN, full-source/installed/default/owner acceptance is inferred by this commit.
- Technical basis: Python venv default system-site isolation and explicit --without-pip/ensurepip, https://docs.python.org/3.12/library/venv.html ; isolated interpreter excludes user site/source injection, https://docs.python.org/3.11/using/cmdline.html .

- Actual fed38f07 source job114094563100 failed only the new helper's pinned Ruff0.12.12 list-comprehension layout; exact native diff/log72269B/SHAfd090b75a6d760554c84d03b780aab5f39c774c98607a15385136a47bea90800. This successor applies that exact layout, with no behavioral change or gate weakening. The real package jobs were still running at that capture; no result inferred.
- Publication observer2149542 genuinely exited1 after the successful ordinary ref update because its immediate PR-head read was stale. Later independent native ref, PR head, commit parent275b4b9c and tree8fc4a6bb prove actual publishedfed38f07; master4f18785 stayed unchanged. No successful tree/commit/ref mutation was repeated, and the writer's finally released all three locks.
