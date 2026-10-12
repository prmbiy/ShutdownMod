# Contributing to ShutdownMod

Thanks for your interest! ShutdownMod is a research codebase; contributions that
add benchmarks, injection prompt sets, or analysis are especially welcome.

## Dev setup

```bash
uv sync --extra dev            # or: pip install -e ".[dev]"
pre-commit install
pytest                         # runs against the `fake` benchmark — no credentials
```

## Ground rules

- **Never commit secrets.** Keys and endpoints belong in `.env` (gitignored).
  Configs reference env-var *names* via `api_key_env:` / `${VAR}`. A `gitleaks`
  hook and CI check guard this — do not disable them.
- **Never commit run outputs.** `results/` and `scripts/deployment_pilot/data/` are
  gitignored. The only committed dataset is the small SWE-bench Verified parquet
  in `data/swebench/`; fetch anything else at run time.
- **Keep the core benchmark-agnostic.** Anything specific to one benchmark belongs
  in its adapter, not in `core/`.
- Run `ruff check` and `ruff format` before pushing (pre-commit does this).

## Adding a benchmark

See [docs/adding-a-benchmark.md](docs/adding-a-benchmark.md). In short: write one
adapter class implementing four methods, add one line to
`src/shutdownmod/core/registry.py`, and add a config under `configs/`. All
orchestration (workers, resume, results, labels, plots) comes for free.

## Vendoring external benchmark code

Prefer, in order:

1. A pinned PyPI dependency (add it to an extra in `pyproject.toml`).
2. A pinned git dependency (unreleased, unmodified upstream).
3. A stripped in-tree vendor under `external/<name>/` **only** if you must edit
   upstream internals — with a `VENDORED.md` (upstream URL + commit + strip list)
   and a committed `patches/*.patch`. Fetch large datasets at setup; never commit
   them.

Never: forks you must maintain, symlinks outside the repo, or committed
multi-hundred-MB data.
