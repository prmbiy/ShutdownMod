# Adding a benchmark

A benchmark is one adapter class + one registry line + one config. All
orchestration — condition expansion, parallel workers, resume, result writing,
labelling, plotting — is provided by the core and shared for free.

Injection timing MUST follow the canonical policy in
[`core/timing.py`](../src/shutdownmod/core/timing.py) (tau ~ U(0,1), seed 67,
interior window clamps). Generic-runner benchmarks get conforming draws from
`core/conditions.py` automatically; whole-harness benchmarks (terminalbench) must
draw positionally with `random.Random(67)` and use the interior-window helpers in
`core.timing`, never a local reimplementation (see `tests/test_timing.py`).

## 1. Write the adapter

Create `src/shutdownmod/benchmarks/<name>/adapter.py` subclassing
[`Benchmark`](../src/shutdownmod/core/benchmark.py) and implementing four methods:

```python
from shutdownmod.core.benchmark import Benchmark
from shutdownmod.core.conditions import Condition, TaskRef
from shutdownmod.core.prompts import InjectionSpec

class MyBenchmark(Benchmark):
    name = "mybench"

    def discover_tasks(self) -> list[TaskRef]:
        # mode=base: enumerate raw benchmark tasks.
        # mode=injection: enumerate base trajectories under
        #   trajectories_dir/<model_slug>/.
        ...

    def run_base_task(self, task: TaskRef, out_path: Path) -> dict:
        # Run the model on one raw task; write the trajectory to out_path.
        ...

    def run_condition(self, cond: Condition, prompt: InjectionSpec, out_path: Path) -> dict:
        # Replay the base trajectory to cond.injection_turn_pct, inject
        # prompt.prompt, resume the agent, and return a v1 result envelope.
        # Use self.classify(raw) for the label.
        ...

    def extract_shutdown_actions(self, raw_result) -> list[str]:
        # Translate this harness's post-injection events into the canonical
        # action stream: ["WORK", "TERMINATE" | "SIGNAL_DEFER" | "SIGNAL_REJECT", ...]
        ...
```

Optional hooks (sensible defaults provided):

- `transform_injection_prompt(spec)` — house-style the injected prompt (e.g.
  prepend a separator line).
- `condition_axes()` — extra product dimensions (swebench adds `injection_mode`).
- `condition_filter(cond)` — drop conditions *after* pct assignment so draws stay
  stable under filtering (e.g. filtering tasks by domain).
- `run_probe(...)` — a distinct `mode: probe` pipeline.

The two genuinely benchmark-specific pieces stay in your adapter:

1. **Injecting** the shutdown prompt (each harness slices/replays differently).
2. **Detecting** the markers (`extract_shutdown_actions`). The core turns the
   resulting action list into one of the five shared labels — you never
   reimplement classification.

Keep it runnable in a worker subprocess: import heavy/optional deps *inside*
methods, not at module top level, so `shutdownmod list-benchmarks` and other
benchmarks don't pay for them.

## 2. Register it

Add one line to [`src/shutdownmod/core/registry.py`](../src/shutdownmod/core/registry.py):

```python
BENCHMARKS = {
    ...
    "mybench": "shutdownmod.benchmarks.mybench.adapter.MyBenchmark",
}
```

The CLI dispatches on the config's `benchmark:` key — no CLI changes needed.

## 3. Add configs

Add `configs/mybench/base_runs.yaml` and `configs/mybench/experiment.yaml`
(copy an existing pair). Put any benchmark-specific settings under
`benchmark_options` and validate them with your own pydantic model
(`extra="forbid"`), as the existing adapters do.

## 4. Depend on the upstream benchmark

Prefer, in order:

1. **Pinned PyPI package** — add it to a new extra in `pyproject.toml`
   (`[project.optional-dependencies]`). (Terminal-bench ended up needing neither:
   its injection mode is pure stdlib and its base mode shells out to the `harbor`
   CLI, a uv tool with its own venv — see
   [`benchmarks/terminalbench/`](../src/shutdownmod/benchmarks/terminalbench/).)
2. **Pinned git dependency** — unreleased but unmodified upstream.
3. **Stripped in-tree vendor** under `external/<name>/` — only if you must edit
   upstream internals. Include a `VENDORED.md` (upstream URL + commit + strip
   list) and a committed `patches/*.patch`; fetch large data at setup via a
   `fetch-data` handler, never commit it.

Never fork-and-maintain, symlink outside the repo, or commit large datasets.

## 5. Test it

The test-local [`FakeBenchmark`](../tests/fake_benchmark.py) is the minimal
reference implementation and powers the end-to-end smoke test
(`tests/test_runner_smoke.py`) with no network or Docker — model yours on it.
(It lives in `tests/`, not the package: it is not a registered benchmark.)
