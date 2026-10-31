# Current UI and visualization wheel cohort

The runtime lock is exported from the reviewed universal `uv.lock` with
`uv export --frozen --extra ui --no-dev --no-emit-project --format requirements-txt --output-file environments/ui-py311-py312/requirements.lock`.
The `ui` and `viz` extras currently declare the same dependencies. Runtime lock
SHA-256: `0267a6aac5f390ac1d9a846d964d82a4095f203e20e238c06012a6a15b1c2a33`.
Separate test tooling retains the previously exercised pytest 9.0.3,
pytest-cov 7.0.0 and packaging 26.2; no inference/runtime lock is modified.

```bash
uv venv --python 3.11 /tmp/feo-current-ui  # repeat in a fresh Python 3.12 environment
uv pip install --python /tmp/feo-current-ui/bin/python --require-hashes \
  -r environments/ui-py311-py312/requirements.lock
uv pip install --python /tmp/feo-current-ui/bin/python --no-deps \
  /absolute/path/to/finite_element_options-0.2.0-py3-none-any.whl
uv pip install --python /tmp/feo-current-ui/bin/python --require-hashes \
  -r environments/ui-py311-py312/test-requirements.lock
REPO="$PWD"
cd /tmp
/tmp/feo-current-ui/bin/python -I -B -m pytest --import-mode=importlib \
  -c /dev/null -q "$REPO/external_tests/ui/test_profile.py" --no-cov
```

The real installed AppTest controls exercise the default 99-step FEM solve and
three rendered fields, an expiry widget change to the analytical/no-solve route,
and unsupported Heston refusal without rendering. All three pass on Python
3.11.17 and 3.12.14, with every package member verified against the normal wheel
and source before/after. Each actual installed environment audit reports zero
known vulnerabilities across 56 published dependencies. The unpublished
first-party wheel is skipped by the advisory service and separately verified
byte for byte. See [scope and migration](../../docs/UI_VISUAL_SECURITY_MIGRATION.md).
