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
write_public_fem_bs_oracle_spec("artifacts/problem_spec.json", report=report)
write_public_fem_bs_result_export("artifacts/result_export.json", report=report)

# One run followed by explicit exports of that same report:
report = run_public_black_scholes_parity_fixture(
    refresh_exports=True, export_directory="artifacts/refreshed"
)
```

`export_directory` is only valid with `refresh_exports=True`. Both destination
files are validated before numerical work. The existing spec URI default,
payload serialization and result `refresh=False` behavior remain unchanged:
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
