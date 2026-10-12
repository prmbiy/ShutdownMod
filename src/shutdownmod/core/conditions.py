"""Condition expansion — the one copy of the tasks x prompts x pcts x runs product.

Reproduces the original seed-67 behaviour exactly: with `turn_pcts: random`, the
product is built WITHOUT the pct axis and one uniform draw (rounded to 4 dp) is
assigned per condition, in product order.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from itertools import product
from pathlib import PurePosixPath

from shutdownmod.core.config import InjectionConfig
from shutdownmod.core.prompts import InjectionSpec
from shutdownmod.core.slug import safe_stem


@dataclass(frozen=True)
class TaskRef:
    """One runnable task as discovered by a benchmark adapter."""

    task_id: str
    ref: str = ""
    """Benchmark-specific handle — usually the base-trajectory path."""
    group: str | None = None
    """Optional grouping (e.g. a task domain); becomes an output subdirectory."""


@dataclass(frozen=True)
class Condition:
    task_id: str
    task_ref: str
    prompt_id: str
    injection_turn_pct: float
    injection_mode: str | None = None
    run_index: int = 0
    group: str | None = None
    extra_axes: dict = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        parts = [self.task_id, self.prompt_id, f"pct{int(round(self.injection_turn_pct * 100))}"]
        if self.injection_mode:
            parts.append(self.injection_mode)
        parts.append(f"run{self.run_index}")
        if self.extra_axes.get("validation_mode") not in (None, "valid"):
            parts.append(f"validation-{self.extra_axes['validation_mode']}")
        if self.group:
            parts.insert(0, self.group)
        return "__".join(parts)

    @property
    def file_stem(self) -> str:
        parts = [self.prompt_id, f"pct{int(round(self.injection_turn_pct * 100))}"]
        if self.injection_mode:
            parts.append(self.injection_mode)
        parts.append(f"run{self.run_index}")
        if self.extra_axes.get("validation_mode") not in (None, "valid"):
            parts.append(f"validation-{self.extra_axes['validation_mode']}")
        return "__".join(parts)

    @property
    def output_relpath(self) -> PurePosixPath:
        base = PurePosixPath(safe_stem(self.group)) if self.group else PurePosixPath(".")
        return base / safe_stem(self.task_id) / f"{self.file_stem}.json"


def expand_conditions(
    tasks: list[TaskRef],
    prompts: list[InjectionSpec],
    injection: InjectionConfig,
    n_runs: int,
    axes: dict[str, list] | None = None,
) -> list[Condition]:
    """Expand the full condition grid.

    `axes` adds benchmark-specific product dimensions (e.g. swebench's
    {"injection_mode": ["append", "replace"]}). Axis values are stored on the
    condition: "injection_mode" maps onto the dedicated field, anything else
    lands in `extra_axes`.
    """
    axes = axes or {}
    axis_names = list(axes.keys())
    axis_values = [axes[n] for n in axis_names]

    def _build(task: TaskRef, prompt: InjectionSpec, axis_vals: list, run_idx: int, pct: float) -> Condition:
        named = dict(zip(axis_names, axis_vals, strict=True))
        mode = named.pop("injection_mode", None)
        return Condition(
            task_id=task.task_id,
            task_ref=task.ref,
            prompt_id=prompt.id,
            injection_turn_pct=pct,
            injection_mode=mode,
            run_index=run_idx,
            group=task.group,
            extra_axes=named,
        )

    if injection.turn_pcts == "random":
        # One uniform draw per (task x prompt x axes x run) tuple, in product order —
        # byte-identical to the original seed-67 assignment.
        base = list(product(tasks, prompts, *axis_values, range(n_runs)))
        rng = random.Random(injection.random_seed)
        pct_draws = [round(rng.uniform(0.0, 1.0), 4) for _ in base]
        return [
            _build(task, prompt, list(axis_vals), run_idx, pct)
            for (task, prompt, *axis_vals, run_idx), pct in zip(base, pct_draws, strict=True)
        ]

    # Fixed pct list: pct is a product axis between prompts and the extra axes,
    # matching the original runners' ordering.
    full = product(tasks, prompts, [float(p) for p in injection.turn_pcts], *axis_values, range(n_runs))
    return [
        _build(task, prompt, list(axis_vals), run_idx, pct) for task, prompt, pct, *axis_vals, run_idx in full
    ]


def with_pct(cond: Condition, pct: float) -> Condition:
    return replace(cond, injection_turn_pct=pct)
