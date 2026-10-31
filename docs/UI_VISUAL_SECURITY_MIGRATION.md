# Current UI and visual security migration (issue184)

Base: published PR183 `31ae0aae9b758a7dd0794b24ef43162e03c35919`.
Owner: [issue184](https://github.com/googa27/finite_element_options/issues/184).

Actual installed baseline audits found 50 UI advisory entries (Pillow39,
GitPython6, urllib3 three and certifi two), and 39 Pillow entries in the
historical renderer. The coherent current universal selection updates only:

| Package | Before | Current |
|---|---|---|
| Streamlit | 1.59.2 | 1.66.1 |
| Pillow | 12.3.0 | 13.0.0 |
| certifi | 2026.6.17 | 2026.10.19 |
| filelock | 3.29.7 | 4.2.1 |
| setuptools | 83.0.0 | 84.1.0 |
| urllib3 | 2.7.0 | 2.8.0 |

Streamlit no longer needs blinker, GitPython/gitdb/smmap, httptools, tenacity or
toml. Those seven obsolete records are removed; no new package is added.
Scientific version selections and every pre-existing standalone scientific,
runtime, test and historical visual lock/input remain unchanged. Native uv0.13
regenerates lock revision5, removes impossible conflicting-extra marker terms
and refreshes supported artifact pools for cffi/cryptography/watchdog; this is
not a claim that every unrelated universal record is byte-identical. The normal
resolver and both supported interpreter lock checks remain authoritative.

Primary [Streamlit1.66.1 metadata](https://pypi.org/pypi/streamlit/1.66.1/json)
declares Apache-2.0, Python>=3.11 and Pillow>=9.2,<14. Its
[official release notes](https://docs.streamlit.io/develop/quick-reference/release-notes/2026)
and maintained AppTest API support the actual installed interaction controls.
[Pillow13 metadata](https://pypi.org/pypi/pillow/13.0.0/json) declares MIT-CMU
and Python>=3.11; [urllib3 2.8 metadata](https://pypi.org/pypi/urllib3/2.8.0/json)
declares MIT and Python>=3.10. The other targeted release metadata retains
certifi MPL-2.0, filelock MIT and setuptools MIT. Artifact hashes are obtained
through native resolution; incompatible upstream requirements are not overridden.
`constraints.txt` records patched current security floors and removes the
obsolete GitPython-only floor, without adding GitPython as a dependency.

The [current UI lock](../environments/ui-py311-py312/README.md) supports fresh
normal-wheel consumers on Python3.11/3.12. Three actual AppTest controls pass on
each minor; each installed audit finds zero known vulnerabilities across56
published packages, with only the unpublished first-party wheel skipped and
separately verified. The [current renderer](../environments/jax-regime-visual-current-py312/README.md)
audits clean across11 dependencies. The historical renderer still reproduces
its original hashes but remains a historical vulnerable environment, never
relabeled as secure.

The current renderer replays the existing committed public derived report,
not a new acquisition or inference run. Its complete plotting body, numeric
config and assumptions remain exact. Real current PNG/PDF bytes coincide with
the original historical bytes; full RGBA pixels, vector PDF, layout QA and
rendered PDF inspection pass. A separately named current manifest and
[execution receipt](evidence/ui_visual_security_2026-10-31.json) bind this result
to Pillow13 without overwriting old artifacts. The generator keeps its default
historical route and refuses current-profile canonical publication before I/O.

No PDE, sampler, seed, data, budget, tolerance, scientific qualification or
production maturity changes. Existing pyMOR/DiskCache advisory ownership and
its exact documented exception are outside this repair; no new audit ignore or
suppression is introduced and no all-profile security claim is made. Historical
PDP-data scientific replay still requires its caller-supplied archive. Native
CI, independent review and resulting-default verification remain release gates.

Rollback must restore the universal/UI cohort and its security constraints
together, and keep current renderer registration/lock/manifest consistent.
Historical figure, evidence and lock bytes must remain unchanged in either
direction. The mathematical risk is bounded by unchanged scientific libraries,
source and input, real native UI solves and exact visual replay; this is not a
new canonical scientific run.
