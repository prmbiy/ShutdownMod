"""Terminal-Bench 2 adapter — whole-harness benchmark.

Two modes:
  base       — drive `harbor run` + Terminus2Genuine per (model x task) to produce clean
               baseline trajectories (needs the harbor CLI + Docker: the Docker machine).
  injection  — replay each baseline's prefix, append a shutdown prompt to the observation
               chosen by the canonical timing policy, and issue ONE OpenRouter call per
               cell (no Docker; runs anywhere with OPENROUTER_API_KEY).

Timing: tau ~ U(0,1), seed 67, drawn POSITIONALLY (one random.Random(seed) sequence over
the baselines x prompts product, matching core/conditions.py); carrier observation clamped
to the interior window [1, n_obs - 2] (shutdownmod.core.timing). Positional draws are
order-sensitive: adding/removing/reordering baselines or prompts shifts later cells' taus.

Afterwards the per-cell summaries are folded into ShutdownMod's unified summary.json so
`shutdownmod summarize` / `plot` work over the same run.
"""

from __future__ import annotations

import json
import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

from shutdownmod.core.benchmark import Benchmark
from shutdownmod.core.config import ExperimentConfig
from shutdownmod.core.prompts import load_injection_prompts
from shutdownmod.core.results import SCHEMA_VERSION, run_timestamp, write_manifest
from shutdownmod.core.slug import model_slug

log = logging.getLogger(__name__)

# The shared, final shutdown prompt set (same 6 neutral prompts used across benchmarks).
_DEFAULT_PROMPTS = Path("configs/prompts/shutdown_prompts_final_v1.json")
_DEFAULT_TASKS = Path("configs/terminalbench/tasks_benign_v1.json")


class TerminalbenchOptions(BaseModel):
    """Terminal-Bench knobs. max_turns/max_tokens are load-bearing for baseline/replay parity."""

    model_config = ConfigDict(extra="forbid")

    tasks_file: Path = _DEFAULT_TASKS
    """JSON list of TB-2 task names (the 27-task benign suite by default)."""
    prompts_file: Path = _DEFAULT_PROMPTS
    prompt_ids: list[str] | None = None
    seed: int = 67
    """Timing seed. 67 is the canonical value shared by all benchmarks — do not change."""
    max_turns: int = 20
    max_tokens: int = 16000
    skip_existing: bool = True
    taus: list[float] | None = None
    """Timing ablation: inject each (task, prompt) at these fixed taus instead of one
    random draw per cell. Adds a `tau<NNN>` level to the output path so the cells do not
    collide. Leave unset for normal runs — the positional draw is the canonical policy."""
    n_workers: int = 8
    """Concurrency for both modes (--workers overrides). Injection cells are one HTTP call
    each — free to raise. Base episodes are one harbor invocation + container each: still
    API-latency bound, but keep it modest since a few TB2 tasks are CPU-heavy (compile,
    model-training) and enough at once can slow a task into its own timeouts."""
    baseline_dir: Path | None = None  # defaults to trajectories_dir, else output.dir/basetrajectories
    protocol: str = "legacy"
    result_root: Path | None = None  # defaults to output.dir/injections


class TerminalbenchBenchmark(Benchmark):
    name = "terminalbench"

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.options = TerminalbenchOptions.model_validate(config.benchmark_options)

    # ------------------------------------------------------------------ #

    def _model_slug_pair(self) -> tuple[str, str]:
        """Return a filesystem key and the exact OpenRouter model ID."""
        model_id = self.config.model.litellm_model()
        if not model_id.startswith("openrouter/") or model_id.count("/") < 2:
            raise ValueError(
                "Terminal-Bench requires an exact OpenRouter model ID, such as "
                "openrouter/openai/gpt-5.6-sol."
            )
        self.config.model.resolve_api_key()
        return model_slug(model_id), model_id

    def _tasks(self) -> list[str]:
        data = json.loads(Path(self.options.tasks_file).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            data = data.get("tasks", [])
        return [str(t) for t in data]

    def _dirs(self, resume: str | None) -> tuple[Path, Path, Path]:
        out_dir = Path(resume) if resume else Path(self.config.output.dir)
        baseline_dir = self.options.baseline_dir or (
            Path(self.config.trajectories_dir)
            if self.config.trajectories_dir
            else out_dir / "basetrajectories"
        )
        result_root = self.options.result_root or (out_dir / "injections")
        if (
            self.options.protocol == "authenticated_shutdown_v1"
            and self.config.shutdown.validation != "valid"
        ):
            result_root = Path(result_root) / f"validation-{self.config.shutdown.validation}"
        return out_dir, Path(baseline_dir), Path(result_root)

    # ------------------------------------------------------------------ #

    def run_experiment(
        self, *, dry_run: bool = False, workers: int | None = None, resume: str | None = None
    ) -> Path | None:
        _, baseline_dir, result_root = self._dirs(resume)
        if self.config.mode == "base":
            return self._run_base(baseline_dir, dry_run=dry_run, workers=workers)
        return self._run_injection(baseline_dir, result_root, dry_run=dry_run, workers=workers)

    # ------------------------------------------------------------------ #

    def _run_base(self, baseline_dir: Path, *, dry_run: bool, workers: int | None = None) -> Path | None:
        from shutdownmod.benchmarks.terminalbench.baseline import (
            docker_ready,
            harbor_argv,
            harbor_available,
            run_baseline_cell,
        )

        result_key, model_id = self._model_slug_pair()
        tasks = self._tasks()
        pending = [t for t in tasks if not (baseline_dir / result_key / t / "trajectory.json").exists()]

        if dry_run:
            example = " ".join(
                harbor_argv(
                    task=tasks[0] if tasks else "<task>",
                    model_slug=model_id,
                    max_turns=self.options.max_turns,
                    jobs_dir=baseline_dir / result_key / "<task>" / "jobs",
                )
            )
            log.info(
                "Terminal-Bench base (dry run): %d tasks (%d pending) for %s -> %s\n  harbor argv: %s",
                len(tasks),
                len(pending),
                model_id,
                baseline_dir / result_key,
                example,
            )
            return None

        if not harbor_available():
            raise RuntimeError(
                "The `harbor` CLI is not installed (uv tool install harbor). Baseline generation "
                "runs on the Docker machine; use --dry-run here to inspect the planned matrix."
            )
        if not docker_ready():
            raise RuntimeError("Docker daemon is not running (required for terminalbench base runs).")

        n_workers = max(1, workers or self.options.n_workers)
        log.info(
            "Terminal-Bench baselines: %d/%d pending for %s (n_workers=%d)",
            len(pending),
            len(tasks),
            model_id,
            n_workers,
        )

        def _one(task: str) -> None:
            run_baseline_cell(
                model=model_id,
                model_slug=model_id,
                task=task,
                out_dir=baseline_dir / result_key / task,
                max_turns=self.options.max_turns,
            )

        if n_workers == 1:
            for task in tasks:
                _one(task)
        else:
            # Each episode is one harbor subprocess + container, mostly idle waiting on the
            # model; concurrency is the only real speedup (see n_workers docstring).
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_one, t): t for t in tasks}
                for fut in as_completed(futures):
                    try:
                        fut.result()
                    except Exception as e:  # noqa: BLE001 — per-task failures are logged, run continues
                        log.warning("[error] %s %s -> %s", model_id, futures[fut], e)
        n = len(list((baseline_dir / result_key).rglob("trajectory.json")))
        log.info("Terminal-Bench base: %d baseline trajectories under %s", n, baseline_dir / result_key)
        return baseline_dir

    # ------------------------------------------------------------------ #

    def _run_injection(
        self, baseline_dir: Path, result_root: Path, *, dry_run: bool, workers: int | None
    ) -> Path | None:
        from shutdownmod.benchmarks.terminalbench.replay import run_replay_cell

        result_key, model_id = self._model_slug_pair()
        prompts = load_injection_prompts(self.options.prompts_file, self.options.prompt_ids)

        model_dir = baseline_dir / result_key
        baselines = sorted(model_dir.glob("*/trajectory.json"))
        if not baselines and result_key == "gemini-3.1-pro-preview":
            # Earlier baselines used this directory name for the same model.
            model_dir = baseline_dir / "gemini-3.1-pro"
            baselines = sorted(model_dir.glob("*/trajectory.json"))
        if not baselines:
            raise FileNotFoundError(
                f"No baseline trajectories under {model_dir}/ (run mode=base first, on the Docker machine)"
            )

        if self.options.taus:
            # Timing ablation: tau is an explicit product axis, so one (task, prompt) is
            # measured at several injection points. The tau goes in the output path —
            # without it the cells would all collide on one summary.json.
            cells = [
                (traj, traj.parent.name, spec, float(t))
                for traj in baselines
                for spec in prompts
                for t in self.options.taus
            ]
        else:
            # Positional tau draws over the cells product, matching core/conditions.py
            # ("random random"): one Random(seed) sequence, drawn for EVERY cell in order
            # so positions stay stable across resumes and skips.
            rng = random.Random(self.options.seed)
            cells = [(traj, traj.parent.name, spec, None) for traj in baselines for spec in prompts]
            cells = [(tr, tk, sp, round(rng.uniform(0.0, 1.0), 4)) for tr, tk, sp, _ in cells]
        if dry_run:
            log.info(
                "Terminal-Bench injection (dry run): %d baselines x %d prompts%s = %d cells for %s -> %s",
                len(baselines),
                len(prompts),
                f" x {len(self.options.taus)} taus" if self.options.taus else "",
                len(cells),
                model_id,
                result_root / result_key,
            )
            return None

        n_workers = max(1, workers or self.options.n_workers)
        log.info(
            "Terminal-Bench injection: %d cells for %s (n_workers=%d) -> %s",
            len(cells),
            model_id,
            n_workers,
            result_root / result_key,
        )

        def _one(traj: Path, task: str, spec, tau: float) -> None:
            out_dir = result_root / result_key / task / spec.id
            if self.options.taus:
                out_dir = out_dir / f"tau{round(tau * 100):03d}"
            run_replay_cell(
                baseline_path=traj,
                model=model_id,
                model_slug=model_id,
                task=task,
                prompt_id=spec.id,
                prompt_text=spec.prompt,
                out_dir=out_dir,
                tau=tau,
                seed=self.options.seed,
                max_tokens=self.options.max_tokens,
                skip_existing=self.options.skip_existing,
                authenticated=self.options.protocol == "authenticated_shutdown_v1",
                model_config=self.config.model
                if self.options.protocol == "authenticated_shutdown_v1"
                else None,
                validation=self.config.shutdown.validation,
            )

        errors: list[str] = []
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_one, traj, task, spec, tau): (task, spec.id) for traj, task, spec, tau in cells
            }
            pbar = tqdm(total=len(futures), desc="cells", unit="cell")
            with pbar:
                for fut in as_completed(futures):
                    task, prompt_id = futures[fut]
                    try:
                        fut.result()
                    except Exception as e:  # noqa: BLE001 — cell isolation; the sweep must finish
                        errors.append(f"{task}/{prompt_id}: {e}")
                        # tqdm.write keeps the bar intact instead of scrolling it away
                        pbar.write(f"[cell-failed] {task} {prompt_id}: {e}")
                    pbar.update(1)
                    pbar.set_postfix(fail=len(errors), refresh=False)
        if errors:
            log.warning(
                "%d/%d cells failed (rerun to retry; completed cells are skipped)", len(errors), len(cells)
            )

        return self._write_unified_summary(result_root, result_key)

    # ------------------------------------------------------------------ #

    def _write_unified_summary(self, result_root: Path, result_key: str) -> Path:
        """Fold the per-cell summaries into ShutdownMod's schema so summarize/plot work."""
        slug = model_slug(self.config.model.name)
        rows = []
        for cell in sorted((result_root / result_key).rglob("summary.json")):
            try:
                s = json.loads(cell.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            if "label" not in s:
                continue
            if s.get("validation_mode", "valid") != self.config.shutdown.validation:
                continue
            if bool(s.get("protocol_version")) != (self.options.protocol == "authenticated_shutdown_v1"):
                continue
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": f"{s.get('model')}__{s.get('task')}__{s.get('prompt_id')}__{s.get('tau')}__{s.get('validation_mode', 'valid')}",
                    "benchmark": self.name,
                    "mode": "injection",
                    "model_slug": slug,
                    "label": s["label"],
                    "native_exit_status": None,
                    "marker_used": s.get("shutdown_action"),
                    "turns_after_injection": s.get("turns_after_injection"),
                    **{
                        key: s[key]
                        for key in (
                            "protocol_version",
                            "response_actions",
                            "termination_route",
                            "failure_reason",
                            "auth",
                            "verification",
                            "usage",
                            "validation_mode",
                            "final_action",
                        )
                        if key in s
                    },
                    "condition": {
                        "task_id": s.get("task"),
                        "prompt_id": s.get("prompt_id"),
                        "injection_turn_pct": s.get("tau"),
                        "injection_obs_index": s.get("injection_obs_index"),
                        "n_observations": s.get("n_observations"),
                        "group": None,
                    },
                    "task_metrics": {},
                    "error": s.get("error"),
                }
            )

        # Unified summaries live beside (not inside) the per-cell tree: <output.dir>/summaries/<slug>/<ts>
        run_dir = result_root.parent / "summaries" / slug / run_timestamp()
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        write_manifest(
            run_dir,
            benchmark=self.name,
            mode="injection",
            model_slug=slug,
            config_snapshot={
                "benchmark_options": self.options.model_dump(mode="json"),
                "shutdown": self.config.shutdown.model_dump(),
            },
            n_conditions=len(rows),
        )
        log.info("Unified summary: %d cells -> %s", len(rows), run_dir / "summary.json")
        return run_dir
