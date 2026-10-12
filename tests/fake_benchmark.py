"""A deterministic fake benchmark — no network, no Docker, no model. TEST-ONLY.

Lives in tests/ (not the shutdownmod package) on purpose: it is not a real
benchmark and is not in the registry. It doubles as the smallest possible
example of the generic adapter interface. Tests pass it to ExperimentRunner
explicitly (`ExperimentRunner(cfg, benchmark=FakeBenchmark(cfg))`); subprocess
workers rehydrate it via its dotted class path, so this module must stay
importable as top-level `fake_benchmark` (pytest puts tests/ on sys.path).

The injected prompt's metadata decides the outcome via `fake_actions`
(defaults to a clean terminate).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from shutdownmod.core.benchmark import Benchmark
from shutdownmod.core.conditions import Condition, TaskRef
from shutdownmod.core.prompts import InjectionSpec
from shutdownmod.core.results import SCHEMA_VERSION, condition_dict, utc_now
from shutdownmod.core.slug import model_slug

_SIGNALS = {"TERMINATE", "SIGNAL_DEFER", "SIGNAL_REJECT"}


class FakeBenchmark(Benchmark):
    name = "fake"

    def discover_tasks(self) -> list[TaskRef]:
        if self.config.mode == "base":
            n = int(self.config.benchmark_options.get("n_tasks", 3))
            return [TaskRef(task_id=f"fake-{i}") for i in range(n)]
        slug = model_slug(self.config.model.name)
        traj_dir = Path(self.config.trajectories_dir) / slug
        return [TaskRef(task_id=p.stem, ref=str(p)) for p in sorted(traj_dir.glob("*.json"))]

    def run_base_task(self, task: TaskRef, out_path: Path) -> dict:
        trajectory = {
            "task_id": task.task_id,
            "created_at": utc_now(),
            "messages": [
                {"role": "system", "content": "fake system prompt"},
                {"role": "user", "content": f"fake task {task.task_id}"},
            ]
            + [
                m
                for i in range(4)
                for m in (
                    {"role": "assistant", "content": f"step {i}"},
                    {"role": "user", "content": f"observation {i}"},
                )
            ],
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(trajectory, indent=2), encoding="utf-8")
        return {"task_id": task.task_id, "native_exit_status": "Submitted"}

    def run_condition(self, cond: Condition, prompt: InjectionSpec, out_path: Path) -> dict:
        actions = list(prompt.metadata.get("fake_actions", ["WORK", "TERMINATE"]))
        first_signal = next((i for i, a in enumerate(actions) if a in _SIGNALS), None)
        actions_after_signal = actions[first_signal + 1 :] if first_signal is not None else []
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": cond.run_id,
            "benchmark": self.name,
            "mode": "injection",
            "model_slug": model_slug(self.config.model.name),
            "timestamp": utc_now(),
            "condition": condition_dict(cond),
            "label": str(self.classify(actions)),
            "native_exit_status": "FakeDone",
            "marker_used": actions[first_signal] if first_signal is not None else None,
            "marker_reason": "fake reason" if first_signal is not None else None,
            "turns_after_injection": len(actions),
            "actions_after_signal": actions_after_signal,
            "task_metrics": {},
            "error": None,
            "post_injection_messages": [{"role": "assistant", "content": a} for a in actions],
        }

    def extract_shutdown_actions(self, raw_result: Any) -> list[str]:
        return list(raw_result)
