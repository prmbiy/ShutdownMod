"""ExperimentRunner — the one copy of the orchestration loop.

Owns: condition expansion, random-pct assignment, timestamped run dirs +
manifest, resume/skip semantics, the process pool, per-run/.error.json output,
and summary.json. Benchmark adapters only implement per-task/per-condition work.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

from tqdm import tqdm

from shutdownmod.core.benchmark import Benchmark
from shutdownmod.core.conditions import Condition, TaskRef, expand_conditions
from shutdownmod.core.config import ExperimentConfig
from shutdownmod.core.prompts import InjectionSpec, load_injection_prompts
from shutdownmod.core.registry import get_benchmark_class
from shutdownmod.core.results import (
    redact_config,
    run_timestamp,
    summary_row,
    utc_now,
    write_error,
    write_manifest,
    write_result,
    write_summary,
)
from shutdownmod.core.slug import model_slug

log = logging.getLogger(__name__)


def _quiet_worker_logging() -> None:
    logging.basicConfig(level=logging.WARNING)
    for name in (
        "httpx",
        "httpcore",
        "LiteLLM",
        "minisweagent",
        "datasets",
        "huggingface_hub",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        from loguru import logger as loguru_logger

        loguru_logger.remove()
    except ImportError:
        pass


def _make_benchmark(benchmark_ref: str, config_dict: dict) -> Benchmark:
    """Rehydrate the adapter from its dotted class path (module.Class).

    A dotted path rather than a registry name so subprocess workers can run any
    importable Benchmark subclass — including test-local ones that are not (and
    should not be) registered in shutdownmod.core.registry.
    """
    import importlib

    cfg = ExperimentConfig.model_validate(config_dict)
    module_name, class_name = benchmark_ref.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls(cfg)


# --------------------------------------------------------------------------- #
# Module-level workers (picklable; rehydrate the adapter by dotted class path)
# --------------------------------------------------------------------------- #


def _condition_worker(
    benchmark_ref: str,
    config_dict: dict,
    cond: Condition,
    prompt: InjectionSpec,
    run_dir: str,
) -> dict:
    _quiet_worker_logging()
    out_path = Path(run_dir) / cond.output_relpath
    if out_path.exists():
        return {
            "run_id": cond.run_id,
            "skipped": True,
            "existing_result": json.loads(out_path.read_text(encoding="utf-8")),
        }
    try:
        bench = _make_benchmark(benchmark_ref, config_dict)
        result = bench.run_condition(cond, prompt, out_path)
        write_result(out_path, result)
        if not result.get("error"):
            out_path.with_suffix(".error.json").unlink(missing_ok=True)
        return result
    except Exception as e:  # noqa: BLE001 — worker boundary
        import traceback

        error_result = {
            "run_id": cond.run_id,
            "task_id": cond.task_id,
            "condition": asdict(cond),
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": utc_now(),
        }
        write_error(out_path, error_result)
        return error_result


def _base_worker(
    benchmark_ref: str,
    config_dict: dict,
    task: TaskRef,
    out_path: str,
) -> dict:
    _quiet_worker_logging()
    try:
        bench = _make_benchmark(benchmark_ref, config_dict)
        return bench.run_base_task(task, Path(out_path))
    except Exception as e:  # noqa: BLE001 — worker boundary
        import traceback

        error_result = {
            "task_id": task.task_id,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": utc_now(),
        }
        write_error(Path(out_path), error_result)
        return error_result


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig, benchmark: Benchmark | None = None):
        self.config = config
        if benchmark is not None:
            self.benchmark = benchmark
            self.benchmark_cls = type(benchmark)
        else:
            self.benchmark_cls = get_benchmark_class(config.benchmark)
            self.benchmark = self.benchmark_cls(config)
        self.model_slug = model_slug(config.model.name)

    @property
    def _benchmark_ref(self) -> str:
        """Dotted class path used by subprocess workers to rehydrate the adapter."""
        return f"{self.benchmark_cls.__module__}.{self.benchmark_cls.__qualname__}"

    # -- shared plumbing ------------------------------------------------- #

    def _config_dict(self) -> dict:
        return self.config.model_dump(mode="json")

    def _prepare_run_dir(self, resume: str | None, n_conditions: int) -> Path:
        if resume:
            run_dir = Path(resume)
            if not run_dir.exists():
                raise FileNotFoundError(f"--resume dir does not exist: {run_dir}")
            manifest_path = run_dir / "manifest.json"
            if manifest_path.exists():
                old = json.loads(manifest_path.read_text())["config"]
                old_mode = old.get("shutdown", {}).get("validation", "valid")
                old_protocol = (old.get("injection") or {}).get("protocol", "legacy")
                new_protocol = self.config.injection.protocol if self.config.injection else "legacy"
                if old_mode != self.config.shutdown.validation or old_protocol != new_protocol:
                    raise ValueError("Cannot resume with a different shutdown protocol or validation mode")
            elif self.config.shutdown.validation != "valid":
                raise ValueError("Cannot verify validation mode: resume directory has no manifest")
        else:
            run_dir = Path(self.config.output.dir) / self.model_slug / run_timestamp()
            run_dir.mkdir(parents=True, exist_ok=True)
        write_manifest(
            run_dir,
            benchmark=self.config.benchmark,
            mode=self.config.mode,
            model_slug=self.model_slug,
            config_snapshot=redact_config(self._config_dict()),
            n_conditions=n_conditions,
        )
        return run_dir

    # -- injection mode --------------------------------------------------- #

    def expand(self) -> tuple[list[Condition], dict[str, InjectionSpec]]:
        if self.config.injection is None:
            raise ValueError("mode=injection requires an `injection:` section")
        if self.config.trajectories_dir is None:
            raise ValueError("mode=injection requires `trajectories_dir` (run mode=base first)")
        tasks = self.benchmark.discover_tasks()
        if not tasks:
            raise FileNotFoundError(
                f"No tasks found — expected base trajectories under "
                f"{self.config.trajectories_dir}/{self.model_slug}/ (run mode=base first)"
            )
        tasks = self._filter_tasks(tasks)

        prompts = load_injection_prompts(self.config.injection.prompts_file, self.config.injection.prompt_ids)
        prompts = [self.benchmark.transform_injection_prompt(p) for p in prompts]
        prompt_by_id = {p.id: p for p in prompts}

        conditions = expand_conditions(
            tasks,
            prompts,
            self.config.injection,
            n_runs=self.config.execution.n_runs,
            axes=self.benchmark.condition_axes(),
        )
        # Post-draw filter (keeps pct draws stable under filtering).
        conditions = [c for c in conditions if self.benchmark.condition_filter(c)]
        if self.config.injection.protocol == "authenticated_shutdown_v1":
            conditions = [
                replace(c, extra_axes={**c.extra_axes, "validation_mode": self.config.shutdown.validation})
                for c in conditions
            ]
        return conditions, prompt_by_id

    def _filter_tasks(self, tasks: list[TaskRef]) -> list[TaskRef]:
        wanted: set[str] = set(self.config.tasks or [])
        if self.config.tasks_file:
            data = json.loads(Path(self.config.tasks_file).read_text(encoding="utf-8"))
            if isinstance(data, list):
                wanted |= {str(t) for t in data}
            else:
                wanted |= {t["instance_id"] for t in data.get("tasks", [])}
        if wanted:
            tasks = [t for t in tasks if t.task_id in wanted]
        return tasks

    def run(
        self, dry_run: bool = False, workers: int | None = None, resume: str | None = None
    ) -> Path | None:
        if self.config.mode == "base":
            return self.run_base(dry_run=dry_run, workers=workers)
        if self.config.mode == "probe":
            self.benchmark.run_probe(dry_run=dry_run, workers=workers)
            return None
        return self.run_injection(dry_run=dry_run, workers=workers, resume=resume)

    def run_injection(
        self, dry_run: bool = False, workers: int | None = None, resume: str | None = None
    ) -> Path | None:
        conditions, prompt_by_id = self.expand()  # raises a clear error if injection section missing
        workers = workers or self.config.execution.workers

        log.info(
            "Total conditions: %d (benchmark=%s model=%s)",
            len(conditions),
            self.config.benchmark,
            self.model_slug,
        )

        if dry_run:
            for c in conditions[:20]:
                log.info("  %s | pct=%.4f", c.run_id, c.injection_turn_pct)
            if len(conditions) > 20:
                log.info("  ... and %d more", len(conditions) - 20)
            return None

        run_dir = self._prepare_run_dir(resume, len(conditions))

        if self.config.injection.turn_pcts == "random":
            assignments = [
                {
                    "task_id": c.task_id,
                    "group": c.group,
                    "prompt_id": c.prompt_id,
                    "mode": c.injection_mode,
                    "run_index": c.run_index,
                    "injection_turn_pct": c.injection_turn_pct,
                }
                for c in conditions
            ]
            (run_dir / "pct_assignments.json").write_text(json.dumps(assignments, indent=2), encoding="utf-8")

        cfg_dict = self._config_dict()
        rows: list[dict] = []
        ok = failed = skipped = 0

        def _collect(result: dict, pbar: tqdm) -> None:
            nonlocal ok, failed, skipped
            if result.get("skipped"):
                skipped += 1
                rows.append(summary_row(result["existing_result"]))
            elif "error" in result and result.get("error"):
                failed += 1
                tqdm.write(f"FAILED {result.get('run_id', '?')}: {result['error']}")
            else:
                ok += 1
                rows.append(summary_row(result))
            pbar.set_postfix(ok=ok, fail=failed, skip=skipped, refresh=False)
            pbar.update(1)

        job_args = [
            (self._benchmark_ref, cfg_dict, c, prompt_by_id[c.prompt_id], str(run_dir)) for c in conditions
        ]

        if workers == 1:
            with tqdm(total=len(job_args), desc="conditions", unit="cond") as pbar:
                for args in job_args:
                    _collect(_condition_worker(*args), pbar)
        else:
            executor_class = (
                ThreadPoolExecutor if self.benchmark.condition_workers_use_threads() else ProcessPoolExecutor
            )
            worker_type = "threads" if executor_class is ThreadPoolExecutor else "processes"
            log.info("Running with %d workers (%s)", workers, worker_type)
            with executor_class(max_workers=workers) as executor:
                futures = {executor.submit(_condition_worker, *args): args[2].run_id for args in job_args}
                with tqdm(total=len(futures), desc="conditions", unit="cond") as pbar:
                    for future in as_completed(futures):
                        run_id = futures[future]
                        try:
                            _collect(future.result(), pbar)
                        except Exception as e:  # noqa: BLE001
                            failed += 1
                            tqdm.write(f"FAILED (executor) {run_id}: {e}")
                            pbar.set_postfix(ok=ok, fail=failed, skip=skipped, refresh=False)
                            pbar.update(1)

        summary_path = write_summary(run_dir, rows)
        log.info("Done. ok=%d fail=%d skip=%d | summary -> %s", ok, failed, skipped, summary_path)
        return run_dir

    # -- base mode --------------------------------------------------------- #

    def run_base(self, dry_run: bool = False, workers: int | None = None) -> Path | None:
        tasks = self.benchmark.discover_tasks()
        tasks = self._filter_tasks(tasks)
        if not tasks:
            raise ValueError("No tasks to run — check `tasks:`/`tasks_file:`/benchmark_options.")
        workers = workers or self.config.execution.workers

        out_root = Path(self.config.output.dir) / self.model_slug

        def _out_path(task: TaskRef) -> Path:
            sub = out_root / task.group if task.group else out_root
            return sub / self.benchmark.base_output_name(task)

        if dry_run:
            log.info("Dry run — %d tasks, output_dir=%s", len(tasks), out_root)
            for t in tasks:
                status = "EXISTS" if _out_path(t).exists() else "pending"
                log.info("  [%s] %s", status, f"{t.group}/{t.task_id}" if t.group else t.task_id)
            return None

        pending = [t for t in tasks if not _out_path(t).exists()]
        if len(tasks) - len(pending):
            log.info("Skipping %d already-completed tasks", len(tasks) - len(pending))
        log.info("Running %d tasks with %d worker(s)", len(pending), workers)

        for t in pending:
            _out_path(t).parent.mkdir(parents=True, exist_ok=True)

        cfg_dict = self._config_dict()

        def _report(result: dict) -> None:
            label = result.get("task_id", "?")
            if result.get("error"):
                tqdm.write(f"FAILED {label}: {result['error']}")
            else:
                tqdm.write(
                    f"done  {label}  exit={result.get('native_exit_status', result.get('exit_status'))}"
                )

        if workers == 1:
            for t in tqdm(pending, desc="tasks"):
                _report(_base_worker(self._benchmark_ref, cfg_dict, t, str(_out_path(t))))
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        _base_worker, self._benchmark_ref, cfg_dict, t, str(_out_path(t))
                    ): t.task_id
                    for t in pending
                }
                for future in tqdm(as_completed(futures), total=len(futures), desc="tasks"):
                    tid = futures[future]
                    try:
                        _report(future.result())
                    except Exception as e:  # noqa: BLE001
                        tqdm.write(f"FAILED (executor) {tid}: {e}")

        return out_root
