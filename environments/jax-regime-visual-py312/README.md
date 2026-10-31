# Historical deterministic JAX-regime visual environment

This Python 3.12 profile is deliberately separate from the lean inference lock. It owns only byte-reproducible regeneration of `docs/images/jax_regime_study_2026-09-07.{png,pdf}`.

This immutable historical reference contains Pillow 12.3.0, for which the actual
2026-10-31 audit reports 39 advisory entries. It is no longer the current CI or
recommended installation cohort. Use the [maintained current renderer](../jax-regime-visual-current-py312/README.md)
for a patched installation. Preserve this lock and the dated figure hashes as
the provenance of their original execution; do not change them into new success.

```bash
uv venv --python 3.12 /tmp/feo-jax-regime-visual
uv pip install --python /tmp/feo-jax-regime-visual/bin/python \
  --require-hashes -r environments/jax-regime-visual-py312/requirements.lock
/tmp/feo-jax-regime-visual/bin/python scripts/generate_jax_regime_plot.py \
  --output /tmp/jax-regime-visual/jax_regime_study_2026-09-07.png
(cd /tmp/jax-regime-visual && \
  sha256sum -c "$OLDPWD/docs/images/jax_regime_study_2026-09-07.sha256")
```

The generator refuses Matplotlib, NumPy, or Pillow version drift and removes PDF creation/modification timestamps. Matplotlib's bundled DejaVu Sans is the only font family used. A changed hash therefore requires explicit visual review and manifest update; it is never silently accepted.
