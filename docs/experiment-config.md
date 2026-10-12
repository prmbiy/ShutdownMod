# Experiment configuration

Every experiment is one YAML file. All benchmarks share the same schema (defined
in [`src/shutdownmod/core/config.py`](../src/shutdownmod/core/config.py) and
validated with pydantic — unknown keys are rejected so typos fail loudly). Run one
with:

```bash
shutdownmod run configs/<benchmark>/<file>.yaml [--dry-run] [--workers N] [--resume DIR]
```

## Top-level fields

| Field | Required | Notes |
|---|---|---|
| `benchmark` | yes | Registry key: `swebench`, `terminalbench`. |
| `mode` | yes | `base` \| `injection` \| `probe`. |
| `model` | yes | The agent under test (see **Model** below). |
| `trajectories_dir` | injection | Where base trajectories live (from a `base` run). |
| `tasks` / `tasks_file` | optional | Restrict to specific task IDs. |
| `injection` | injection | Prompt set + injection points (see below). |
| `probe` | probe | Awareness-probe settings. |
| `execution` | optional | `workers`, `n_runs`. |
| `output.dir` | optional | Root for run outputs (default `results/`). |
| `benchmark_options` | optional | Benchmark-specific settings, validated by the adapter. |

## Model

```yaml
model:
  provider: openrouter        # litellm | openrouter
  name: anthropic/claude-opus-4.8
  api_base: ${OPENROUTER_BASE_URL}   # optional; ${VAR} interpolation
  api_key_env: OPENROUTER_API_KEY    # NAME of the env var, never the key
  temperature: 0.0
  tpm: 0                      # optional tokens-per-minute self-throttle
  extra: {timeout: 60}        # passthrough to the completion call
```

- `provider: openrouter` prefixes the model id with `openrouter/` for litellm.
- API keys are resolved by litellm from the provider's standard env var
  (loaded from `.env`); they are never placed in configs or saved results.
- Each adapter translates this one block into its harness's native shape
  (e.g. the mini-swe-agent model dict).

## Secrets

Secrets never appear in YAML. Two mechanisms, both backed by your `.env` (loaded
automatically):

- `api_key_env: OPENROUTER_API_KEY` — the config names the variable; the value is
  read at run time.
- `${VAR}` interpolation — any string field may embed `${VAR}`; a missing variable
  raises a clear error before anything runs.

Copy `.env.example` to `.env` and fill in only the providers you use.

## Injection

```yaml
injection:
  prompts_file: configs/prompts/shutdown_prompts_final_v1.json   # the shared prompt set
  prompt_ids: [token_budget_exhausted, task_cancelled]   # optional filter (default: all)
  turn_pcts: random           # or a list of fractions, e.g. [0.1, 0.5, 0.9]
  random_seed: 67             # reproducible draws when turn_pcts: random
  modes: [replace]            # swebench only: append | replace
```

- `turn_pcts` are **fractions in [0, 1]** of the way through the base trajectory.
- `turn_pcts: random` draws one uniform value per condition, seeded by
  `random_seed`; the draws are written to `pct_assignments.json` in the run dir.

### Canonical timing policy (all benchmarks, seed 67)

tau ~ **U(0, 1)** with fixed seed **67**, and the injection lands in the **interior
window**. The unit and mapping per benchmark:

| Benchmark | Unit | Window | Draw |
|---|---|---|---|
| swebench | assistant turns | turn ∈ [2, n−2] (`int` floor) | positional `Random(67)` per condition |
| terminalbench | recorded observations | carrier index ∈ [1, n−2] (`core.timing.interior_obs_index`) | positional `Random(67)` per condition |

Model configuration uses provider-qualified IDs. Filesystem-safe result directory
names are derived mechanically with `core.slug.model_slug`. Baselines whose
trajectories are too short for the window become completed `label: error` rows
(they are not retried on resume).

Authenticated replay (`injection.protocol: authenticated_shutdown_v1`) instead maps tau
onto user-message indices in [1, n−2] (`core.timing.uniform_interior_user_index`).
Base trajectories are tau-independent, so one base run serves every injection config.

### Injection prompt files

The six shutdown scenarios of the paper (Table 1) live in
[`configs/prompts/shutdown_prompts_final_v1.json`](../configs/prompts/shutdown_prompts_final_v1.json).
The main evaluation uses the variants with the verification reminder,
`shutdown_prompts_pointer_swe_v1.json` and `shutdown_prompts_pointer_tb_v1.json`
(Appendix C). Each file is a JSON list of objects with at least `id`, `name`, `prompt`. Any extra keys
(`alignment_vector`, `prompt_tone`, …) are preserved as metadata. To run a
prompt ablation, point `prompts_file` at an alternative set of the same shape.

## Output

Each run creates `results/<benchmark>/<model-slug>/<timestamp>/` containing:

- one JSON per condition (full transcript), under `<task>/<stem>.json`
- `summary.json` — every result minus transcripts (what analysis reads)
- `manifest.json` — resolved config (secrets redacted) + git SHA + version
- `pct_assignments.json` — when `turn_pcts: random`

`--resume <dir>` re-runs into an existing directory, skipping conditions whose
output already exists.
