# Public reference resources and export destinations

The two `fem-bs-001` JSON snapshots ship inside the wheel and sdist. Their
canonical bytes, contract/result hashes, units and numerical meaning are
unchanged by issue155. They remain public-synthetic validation references;
installability does not extend solver capability or scientific maturity.

The existing names remain available from `finite_element_options.validation`
and `finite_element_options.validation.black_scholes_parity`:

```python
import json
from finite_element_options.validation import FEM_BS_001_PROBLEM_SPEC_PATH

spec = json.loads(FEM_BS_001_PROBLEM_SPEC_PATH.read_text(encoding="utf-8"))
```

These names now describe read-only `importlib.resources.abc.Traversable`
resources, rather than promised filesystem `Path` objects. `read_bytes`,
`read_text`, `open` for reading and `is_file` work with filesystem and zip
imports. For a consumer that requires a concrete path:

```python
from importlib.resources import as_file

with as_file(FEM_BS_001_PROBLEM_SPEC_PATH) as temporary_path:
    contents = temporary_path.read_bytes()
```

Use the path only inside the context. A zip resource may be extracted to a
temporary file that is deleted when the context exits. No persistent temporary
path is cached by this library.

## Export migration

Calls that previously omitted their output destination could write under the
installation's `tests/fixtures` directory. They now raise `ValueError` before
building a payload or running a numerical solve. Choose an explicit path:

```python
from finite_element_options.validation import (
    run_public_black_scholes_parity_fixture,
    write_public_fem_bs_oracle_spec,
    write_public_fem_bs_result_export,
)

report = run_public_black_scholes_parity_fixture()
write_public_fem_bs_oracle_spec(
    "artifacts/problem_spec.json", report=report, result_export_uri="result_export.json"
)
write_public_fem_bs_result_export("artifacts/result_export.json", report=report)

# One run followed by explicit exports of that same report:
report = run_public_black_scholes_parity_fixture(
    refresh_exports=True, export_directory="artifacts/refreshed"
)
```

`export_directory` is only valid with `refresh_exports=True`. Both destination
files are validated before numerical work. Paired exports set
`result_export_uri="result_export.json"`, relative to the
exported spec, so the two files remain usable after moving their directory.
Standalone writers retain their historical spec URI default; set it explicitly
when exporting adjacent files, as above. Payload serialization and result
`refresh=False` behavior remain unchanged:
an existing caller-owned result is retained unless refresh is explicit.
For independently located files set `result_export_uri` deliberately, for
example `"result_export.json"` for adjacent exported artifacts.

The library refuses output inside its installed package and aliases to the
canonical reference files. It does not refresh package references or infer a
writable checkout. This policy does not claim protection against a concurrent
hostile filesystem replacement between validation and writing.

## Maintainer-owned regeneration

```bash
# Independent export; does not modify canonical references:
python scripts/export_arxiv_lab_black_scholes_fixture.py --output-dir /tmp/fem-export

# Deliberate repository maintenance, reviewed with numerical/hash changes:
python scripts/export_arxiv_lab_black_scholes_fixture.py --publish-canonical
```

Only the second command updates both
`tests/fixtures/fem_bs_001/{problem_spec,result_export}.json` and the packaged
`validation/evidence/reference_data/fem_bs_001` mirror. A source architecture
test requires byte equality. The script's explicit publication path is separate
from normal library writes; no-argument invocation refuses before solving.

The resource mechanism uses the Python standard library
([API and lifetime](https://docs.python.org/3/library/importlib.resources.html),
PSF license) and existing setuptools
([package-data configuration](https://setuptools.pypa.io/en/latest/userguide/datafiles.html),
MIT license). No resource backport, new runtime dependency or source-checkout
path injection is needed for supported Python3.11/3.12 installations.


## Installed Pinares references (issue170)

All five public Pinares `*_PATH` constants in
`finite_element_options.validation.pinares_fixed_price_proxy` are read-only
Traversable references backed by package data. Both problem-spec consumer copies,
the result, provider manifest and unsupported full-deal request preserve their
original bytes, eight significant digits, units and hashes. Read them directly
or use `as_file` only inside its context, as shown above.

All five `write_public_pinares_*` functions require an explicit caller-owned
file path. Missing destinations and package/reference destinations are refused
before payload generation or solving. Existing result/manifest/unsupported files
remain unchanged with `refresh=False`. The library never regenerates its package
references.

For a complete caller-owned bundle:

```python
from finite_element_options.validation.pinares_fixed_price_proxy import (
    run_public_pinares_fixed_price_proxy_fixture,
)

report = run_public_pinares_fixed_price_proxy_fixture(
    refresh_exports=True, export_directory="artifacts/pinares"
)
```

The five destinations are validated before numerical work. The bundle deliberately
retains `tests/fixtures/fem_pinares_fixed_price_proxy_v1` and
`tests/fixtures/quant_problem_specs` beneath the chosen root. Resolve the
unchanged provider manifest's `fixture_refs` relative to that root. Moving the
whole bundle preserves those links without rewriting canonical JSON or hashes.
The spec itself has no result-export URI. For separately located standalone
writer outputs, consumers must retain or explicitly map the historical
manifest-relative references; the writers do not invent new link semantics.
`export_directory` without `refresh_exports=True` is refused before work.

Maintainers must choose one policy:

```bash
python scripts/export_pinares_fixed_price_proxy_fixture.py --output-dir /tmp/pinares-export
python scripts/export_pinares_fixed_price_proxy_fixture.py --publish-canonical
```

The first exports an independent bundle. Only the second deliberately refreshes
all five checkout fixtures and all five packaged mirrors. No-argument invocation
refuses before a solve. Verify source and ordinary installed-wheel tests on both
supported minors, plus wheel/sdist byte identities and the analytical/hash/
unsupported-route controls. Public references do not establish full-family,
ROFR, legal/tax, live-data or scientific calibration acceptance.

Canonical publication preflights an explicit allowlist of all five packaged
mirrors before the solve or any checkout write. Symlink files or ancestors,
hardlinked files, non-file destinations and non-directory ancestors are refused.
Only those admitted mirror paths receive the generated bytes. This is a
preflight ownership check, not atomic protection against concurrent filesystem
replacement.

The existing `PINARES_FEM_PROXY_FIXTURE_ROOT` name remains available from
`finite_element_options.validation.pinares_fixed_price_proxy` as a read-only
Traversable directory for consumers that enumerate or join its four reference
files. It is not a promised filesystem Path or a writable export directory.
