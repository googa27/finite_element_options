# AGENTS.md — Finite Element Options

## Mission
- Run the PDE solver regression suite whenever code or data affecting numerical solvers changes, and block merges on failures until resolved.
- Keep `docs/` and in-repo Markdown guides up to date after modelling or API changes; prefer succinct, technically precise updates over verbose prose.
- Surface discrepancies between documented processes and actual repository state so maintainers can keep human- and agent-facing docs aligned.

## Project Snapshot
Finite Element Options provides finite-element and related numerical tooling to price and hedge derivative contracts. Core goals:
- Solve transformed forward parabolic PDEs (e.g. Black–Scholes via `tau = T - t`) with finite elements and supporting numerical schemes.
- Compute Greeks and risk measures, optionally via JAX acceleration.
- Support market data ingestion, calibration workflows, plotting, and Streamlit-based exploration.
- Maintain extensibility for new products (options on futures, exotic payoffs) together with robust testing.

## Repository Map
- `src/`: library code (solvers, spaces, problem definitions, calibration, plotting, CLI helpers).
- `tests/`: pytest suite (unit, integration, benchmarks, property-based checks). PDE solvers live in `tests/test_black_scholes_1d.py`, `tests/test_fenics_solver.py`, `tests/test_fd_black_scholes.py`, and related files.
- `examples/`: runnable scenarios such as `basic_usage.py` showing end-to-end pricing pipelines.
- `docs/`: living documentation (`benchmarking.md`, `ROADMAP.md`, figures). Update alongside code changes.
- `main.py` / `demo_adaptive.py`: Streamlit app entry point and adaptive mesh showcase.
- `.github/workflows/ci.yml`: CI pipeline (Python 3.11, `pytest --cov`, solver benchmarks, `pydocstyle`).

## Environment Setup
1. Create a fresh virtual environment: `python -m venv .venv`
2. Activate it: `source .venv/bin/activate` (Unix) or `.venv\Scripts\Activate.ps1` (Windows PowerShell).
3. Install runtime/test dependencies: `pip install --upgrade pip && pip install -r requirements.txt`
4. Install the project in editable mode for local imports: `pip install -e .`
5. Tooling (recommended for agents & contributors): `pip install black ruff mypy pre-commit`
6. Optional FEniCS backend: uncomment `fenics-dolfinx` in `requirements.txt` and install if those solvers are exercised.

## Core Commands
- Unit/integration tests (parallel): `pytest -n auto`
- PDE solver regression focus: `pytest -n auto tests/test_black_scholes_1d.py tests/test_fd_black_scholes.py tests/test_fenics_solver.py`
- Coverage report: `pytest --cov=src --cov-report=term-missing`
- Benchmarks (matches CI): `pytest tests/test_benchmark_black_scholes.py --benchmark-json=benchmark.json`
- Linting & formatting: `black src tests`, `ruff check src tests`, `mypy src tests`
- Streamlit UI: `streamlit run main.py`
- Example pipeline: `python examples/basic_usage.py`

## Workflow Checklist (for every meaningful change)
1. Sync with the latest `main` and branch before edits.
2. Activate the project virtual environment and confirm dependencies are current.
3. Implement changes following the design principles below; prefer composition, keep responsibilities narrow, and add docstrings where public APIs change.
4. Update or add tests (unit, integration, end-to-end) covering new behaviour; extend PDE solver coverage when altering numerical code or meshes.
5. Run targeted PDE solver tests first, then the full suite with coverage. Record any failure details in `.gemini_project`.
6. Format and lint (`black`, `ruff`, `mypy`, `pydocstyle` as needed). If fixes are required, apply them before continuing.
7. Update documentation (`docs/`, `README.md`, changelog) to reflect behaviour, performance, or interface changes.
8. Confirm CI parity by mirroring workflow steps locally when possible.
9. Craft Conventional Commit messages (e.g., `feat(solver): add theta scheme`), and prefer small, reviewable diffs.

## Code Quality & Style Expectations
- Follow OOP and SOLID principles where they improve clarity; favour composition over inheritance.
- Employ standard design patterns (Strategy for pricers, Factory for problem setups, etc.) when they simplify extensibility.
- Adhere to Python style guidance per PEP 8, except where this document specifies explicit overrides (e.g., 120-character lines).
- Use type hints everywhere; keep `mypy` clean and avoid `Any` unless unavoidable.
- Provide docstrings for all public modules, classes, functions, and methods, documenting inputs, outputs, and side-effects.
- Use custom exceptions for domain-specific error states (e.g., invalid payoff configuration, mesh issues).
- Maintain 120-character line length and auto-format with Black; resolve Ruff (`E`, `F`, `B`) warnings promptly.
- Keep `src/` importable (no implicit relative imports) so tests and consumers work under both editable installs and packaged builds.

## Testing Strategy
- Unit tests live alongside core modules; add fixtures targeting deterministic PDE solutions where closed forms exist (Black–Scholes analytic pricing for validation).
- Integration tests ensure solver pipelines work end-to-end (mesh creation, PDE solve, Greeks, calibration).
- End-to-end/benchmark tests (`tests/test_benchmark_black_scholes.py`) measure performance drift—update baseline expectations cautiously.
- When introducing numerical schemes, cross-check against analytical or previously validated solutions to avoid silent instability.
- Hypothesis-based tests exist for probabilistic coverage; keep them fast and deterministic when seeding is possible.

## Documentation & Knowledge Sharing
- Mirror changes in solver behaviour, calibration routines, or interfaces in `docs/` and `README.md`.
- Add plots or benchmark artefacts to `docs/images/` when they aid comprehension; ensure generated assets are deterministic or provide scripts to reproduce them.
- When updating Streamlit flows or CLIs, include usage snippets in the README or `docs/benchmarking.md`.
- Changelog updates should summarise new capabilities, bug fixes, and backwards-incompatible changes.

## Project-Specific Guidance
- Always formulate PDE problems as forward parabolic IVPs before discretisation; document variable transforms in code comments and docstrings.
- Preserve numerical stability guardrails (time step, mesh density, CFL-like constraints). If altering defaults, justify and update tests.
- Greeks must be validated against analytical references (Black–Scholes) or high-fidelity numerical baselines; log discrepancies with rationale.

## Coordination With Other Agents
- Treat this AGENTS.md as the canonical source. If other agent configs exist (e.g., `GEMINI.md`, `QWEN.md`, `.cursorrules`), make them lightweight pointers back here to avoid divergence.
- When generating ancillary agent files, include only deltas (e.g., environment quirks) and link to this document for shared standards.
- Keep instructions synchronised across agents whenever workflows change; flag outdated files in reviews.

## Escalation & Logging
- Record failing commands, flaky tests, or environment anomalies in `.gemini_project` with timestamps and reproduction steps.
- If automation cannot proceed (e.g., missing dependency, CI drift), stop, capture the state, and request maintainer guidance rather than guessing.

Following this playbook keeps human and AI contributors aligned, ensures numerical credibility, and preserves a high signal-to-noise workflow for the derivative pricing library.
