# Optional profile security lock repair

Base: published PR183 `acc77c552178c25b09ea58c51102815bdbbeaefe`.

Scope: the two executed optional-profile audit failures. Preserve sampler versions,
synthetic data, seeds, draws, chains, diagnostics, numerical tolerances, historical
evidence and all audit refusals.

1. Verify primary release metadata/API/license for filelock4.2.1, Pillow13.0.0,
   setuptools84.1.0, certifi2026.10.19 and msgpack1.2.3. Use the retained native
   audit failures as the security RED evidence.
2. Update only those necessary transitive lock records and their artifact hashes.
   Compile the two isolated Bayesian locks with explicit security constraints;
   a universal uv-lock attempt also upgraded unrelated Streamlit and was retained
   then rejected. Keep that unrelated universal/UI graph unchanged. Regenerate
   the JAX-regime CI lock while preserving its unaffected package versions/hashes.
3. Keep the historical Bayesian artifact and lock identity intact. Bind current
   replay to the new exact lock and permit only its explicit historical/current
   reference identities. Test refusal of an unregistered reference. Preserve the
   JAX scientific runtime/test/visual locks and historical study outputs.
4. Create fresh isolated Python3.12 environments, install locks with hashes and a
   normal reviewed wheel, verify complete package origins, run unchanged PyMC,
   NumPyro, DYNAMAX/Diffrax tests and declared profile replay, then actual audit,
   pip consistency and SBOM commands. Retain exact command/output receipts.
5. Update current lock fingerprints and migration metadata, run relevant static,
   architecture, lint/type/build gates, freeze the patch and obtain independent
   review. Root owns publication/native reruns and resulting-default acceptance.

Review controls: no audit ignores or dependency omissions; no scientific-package
upgrades; no relabeling of dated historical evidence; no package-origin overlays;
only exact historical/current replay lock identities; no inference maturity claim.
