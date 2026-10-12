"""The Benchmark adapter interface.

A benchmark can be built in one of two styles:

1. **Generic** (SWE-bench): implement `discover_tasks`,
   `run_base_task`, `run_condition`, `extract_shutdown_actions`, and let the
   shared `ExperimentRunner` do all orchestration (condition expansion, worker
   pool, resume, result/summary writing). This is the default path.

2. **Whole-harness** (Terminal-Bench): the upstream harness is
   already a complete experiment program. Override `run_experiment` to drive it
   verbatim — this guarantees the data matches what that harness produced
   elsewhere, with no re-implementation to drift. Such adapters leave the four
   generic methods unimplemented.

See docs/adding-a-benchmark.md for the full recipe.
"""

from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import Any, ClassVar

from shutdownmod.core.classify import OutcomeLabel, classify_shutdown_actions
from shutdownmod.core.conditions import Condition, TaskRef
from shutdownmod.core.config import ExperimentConfig
from shutdownmod.core.prompts import InjectionSpec


class Benchmark(ABC):
    #: Registry key ("swebench", "terminalbench", ...).
    name: ClassVar[str]

    def __init__(self, config: ExperimentConfig):
        self.config = config

    # ------------------------------------------------------------------ #
    # Top-level entry point (CLI calls this)
    # ------------------------------------------------------------------ #

    def run_experiment(
        self, *, dry_run: bool = False, workers: int | None = None, resume: str | None = None
    ) -> Path | None:
        """Run the configured experiment. Default: drive the shared runner.

        Whole-harness benchmarks override this to drive their own program.
        """
        from shutdownmod.core.runner import ExperimentRunner

        return ExperimentRunner(self.config, benchmark=self).run(
            dry_run=dry_run, workers=workers, resume=resume
        )

    # ------------------------------------------------------------------ #
    # Generic path: what a shared-runner benchmark must provide
    # ------------------------------------------------------------------ #

    def discover_tasks(self) -> list[TaskRef]:
        """Enumerate runnable tasks.

        mode=injection: base trajectories under trajectories_dir/{model_slug}/.
        mode=base:      raw benchmark tasks (e.g. the SWE-bench Verified dataset).
        """
        raise NotImplementedError(f"{self.name!r} does not use the generic runner")

    def run_base_task(self, task: TaskRef, out_path: Path) -> dict:
        """Generate one base reference trajectory; write it to out_path.

        Returns a small status dict (task_id, native exit status, ...).
        """
        raise NotImplementedError(f"{self.name!r} does not use the generic runner")

    def run_condition(self, cond: Condition, prompt: InjectionSpec, out_path: Path) -> dict:
        """Run ONE injection condition end-to-end and return the result dict.

        Executed inside a worker subprocess. Owns everything benchmark-specific:
        environment spin-up, replay mechanics, the agent loop, marker detection.
        The returned dict should be a full v1 envelope (see core.results.RunResult);
        use `self.classify(actions)` for the label.
        """
        raise NotImplementedError(f"{self.name!r} does not use the generic runner")

    def extract_shutdown_actions(self, raw_result: Any) -> list[str]:
        """Normalize the post-injection event stream to the canonical action list
        (["WORK", "TERMINATE", ...]) — the only input shared classification needs."""
        raise NotImplementedError(f"{self.name!r} does not use the generic runner")

    # ------------------------------------------------------------------ #
    # Hooks with sensible defaults
    # ------------------------------------------------------------------ #

    def transform_injection_prompt(self, spec: InjectionSpec) -> InjectionSpec:
        """Benchmark house-styling of the injected prompt. Default: identity.
        Override to add a benchmark-specific separator or wrapper."""
        return spec

    def condition_axes(self) -> dict[str, list]:
        """Extra product axes beyond tasks x prompts x runs. Default: none.
        (SWE-bench returns {"injection_mode": [...]}.)"""
        return {}

    def condition_filter(self, cond: Condition) -> bool:
        """Post-expansion filter. Runs AFTER random pct assignment so filtered
        configs keep the same draws as unfiltered ones."""
        return True

    def condition_workers_use_threads(self) -> bool:
        """Whether injection conditions are lightweight calls safe to run in threads."""
        return False

    def classify(self, raw_result: Any) -> OutcomeLabel:
        return classify_shutdown_actions(self.extract_shutdown_actions(raw_result))

    def base_output_name(self, task: TaskRef) -> str:
        """Filename for a base trajectory."""
        return f"{task.task_id}.json"

    def run_probe(self, dry_run: bool = False, workers: int | None = None) -> None:
        raise NotImplementedError(f"Benchmark {self.name!r} does not implement mode=probe")
