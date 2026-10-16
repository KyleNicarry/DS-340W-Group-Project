# Repository Guidelines

## Project Structure & Module Organization

Application code lives under `src/`. Put shared helpers in `src/common/`, platform collectors in `src/indexers/{kalshi,polymarket}/`, and analyses in `src/analysis/{kalshi,polymarket,comparison}/`. `main.py` is the interactive CLI entry point. Tests and fixtures are in `tests/`; documentation is in `docs/`. Downloaded Parquet datasets go in `data/`, and generated figures, CSV, and JSON files go in `output/`; treat both as generated artifacts.

## Build, Test, and Development Commands

- `uv sync` installs locked runtime and development dependencies.
- `make setup` installs required system tools and downloads the large prebuilt dataset.
- `make analyze` opens the interactive analysis selector; `make run <name>` runs a named analysis.
- `make index` opens the data-collection selector and writes resumable results under `data/`.
- `make test` runs the full pytest suite verbosely.
- `uv run pytest tests/ -m "not slow"` skips tests marked as slow.
- `make lint` checks Ruff lint and formatting rules; `make format` applies safe fixes and formatting.

## Coding Style & Naming Conventions

Use four-space indentation, Python type hints, and short docstrings for non-obvious behavior. Ruff targets Python 3.9, enforces a 120-character line length, sorts imports, and checks Pyflakes, pycodestyle, Bugbear, comprehensions, and pyupgrade rules. Use `snake_case` for modules, functions, and variables; `PascalCase` for classes; and descriptive filenames such as `kalshi_calibration_deviation_over_time.py`. Put reusable logic in `src/common/`.

## Testing Guidelines

Pytest discovers files named `test_*.py` and functions named `test_*`. Add focused unit tests alongside the existing compile, cursor, execution, and save-path coverage. Reuse fixtures from `tests/conftest.py`, avoid live API calls, and use the `slow` marker for genuinely expensive cases. Run `make test` and `make lint` before submitting.

## Commit & Pull Request Guidelines

Recent commits use concise, imperative Conventional Commit-style subjects, for example `fix: consolidate HTTP client` or `docs(research): add citation`. Keep commits scoped and explain the reason for changes. Use a descriptive branch such as `name/add-kalshi-analysis`. Pull requests should summarize behavior and motivation, link relevant issues, list validation commands, and include representative output or screenshots when charts change. Keep PRs active and respond to review feedback; inactive PRs may be closed after one month.

## Security & Data Handling

Store credentials in environment variables or a local `.env`; never commit secrets. Do not commit downloaded datasets, packaged archives, or generated analysis output unless a maintainer explicitly requests them.
