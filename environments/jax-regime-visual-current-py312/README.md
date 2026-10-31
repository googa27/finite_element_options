# Current deterministic JAX-regime renderer

Issue [184](https://github.com/googa27/finite_element_options/issues/184) separates
the maintained renderer from its immutable historical Pillow 12.3 reference.
This profile fixes Matplotlib 3.10.3, NumPy 2.4.6 and Pillow 13.0.0; all other
renderer versions/hashes match the historical lock. Its lock SHA-256 is
`1a0939035ba0eb7927ffec7d7b1fb958700316769eb22f91c74d6609d30ebb4f`.

```bash
uv venv --python 3.12 /tmp/feo-current-visual
uv pip install --python /tmp/feo-current-visual/bin/python --require-hashes \
  -r environments/jax-regime-visual-current-py312/requirements.lock
MPLCONFIGDIR=/tmp/feo-current-mpl /tmp/feo-current-visual/bin/python -I -B \
  scripts/generate_jax_regime_plot.py --visual-profile current --qa-layout \
  --output /tmp/feo-current-visual-output/jax_regime_study_2026-09-07_current_2026-10-31.png
(cd /tmp/feo-current-visual-output && \
  sha256sum -c "$OLDPWD/docs/images/jax_regime_study_2026-09-07_current_2026-10-31.sha256")
```

The explicit current profile requires the exact registered stack and refuses
`--publish-canonical` before input reads or plotting imports. Historical/default
selection and historical output alias protections are unchanged. Current CI
replays this secure profile and separately checks stored historical bytes.

The real current replay uses the committed, already public derived research
report, SHA-256 `2faf09c5316d59ebdeec31e26c85483c87cee5f92be786f563923d6e11a9a854`.
It performs no inference, acquisition or new market calibration. Original plot
construction and numeric configuration remain exact. Layout QA passes; all
4800x2700 RGBA pixel bytes and complete PNG/PDF bytes equal the historical
figures. New filenames and the [migration receipt](../../docs/evidence/ui_visual_security_2026-10-31.json)
identify the actual current environment even though its image hashes coincide.

The actual installed renderer audit reports zero known vulnerabilities across
11 published dependencies. This is a bounded cohort result, not a claim that
every historical/research environment is secure.
