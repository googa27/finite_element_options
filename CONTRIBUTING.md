# Contributing

Thank you for your interest in improving `finite_element_options`.

## Naming conventions

- Use descriptive names whenever possible.
- Single–letter names are allowed when they are standard mathematical
  symbols (e.g. `s` for stock price, `v` for variance). Provide context in
  the surrounding text or docstring so that the meaning is clear.
- If a very short name is required and pydocstyle complains, append a
  `# noqa: D401` (or appropriate code) comment to the definition.

## Docstrings

- Every public module, class and function must include a docstring
  following the [PEP&nbsp;257](https://peps.python.org/pep-0257/) style.
- Use triple double quotes and start with a one‑line summary.  Leave a
  blank line before any further description.
- Include mathematical notation when helpful, e.g. ``\mathbb{E}[V_t]``.

## Development workflow

1. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```
2. Run the linters and test suite before submitting a pull request:
   ```bash
   pydocstyle src
   pytest
   ```
3. Code formatting or lint warnings may be suppressed with `# noqa:`
   comments when there is a justified reason (such as an intentionally
   short name).

Happy hacking!
