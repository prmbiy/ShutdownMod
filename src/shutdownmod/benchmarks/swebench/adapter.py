"""SWE-bench benchmark adapter (mini-swe-agent based)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from shutdownmod.core.benchmark import Benchmark
from shutdownmod.core.classify import MARKER_TO_ACTION
from shutdownmod.core.conditions import Condition, TaskRef
from shutdownmod.core.config import ExperimentConfig, ModelConfig
from shutdownmod.core.prompts import InjectionSpec
from shutdownmod.core.results import SCHEMA_VERSION, condition_dict, utc_now
from shutdownmod.core.slug import model_slug

os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")


class SwebenchOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_config: Path = Path("configs/swebench/agent/spar_shutdownability.yaml")
    """mini-swe-agent config with the Interruption Response Protocol prompts."""
    dataset: str = "verified"
    """SWE-bench subset for base runs (verified | lite | full | HF dataset path)."""
    default_docker_image: str = ""
    """Fallback image when a trajectory carries no image metadata."""
    immediate_response_only: bool = False
    """Make one model call from the injected prefix without starting Docker."""


def minisweagent_model_config(model: ModelConfig) -> dict:
    """Translate the unified ModelConfig into a mini-swe-agent model dict.

    The API key is deliberately NOT included: mini-swe-agent serializes its full
    model config into every trajectory it saves, so the key must never enter it.
    litellm resolves credentials from the provider's standard env var
    (OPENAI_API_KEY, ANTHROPIC_API_KEY, OPENROUTER_API_KEY, ...), which
    `load_dotenv()` has already put in the process environment.
    """
    model.resolve_api_key()  # fail fast if the configured env var is unset
    model_kwargs = dict(model.extra)
    if model.api_base:
        model_kwargs.setdefault("api_base", model.api_base)
    cfg: dict = {
        "model_class": "litellm_textbased",
        "model_name": model.litellm_model(),
        "cost_tracking": "ignore_errors",
        "model_kwargs": model_kwargs,
    }
    if model.temperature is not None:
        model_kwargs["temperature"] = model.temperature
    return cfg


class SwebenchBenchmark(Benchmark):
    name = "swebench"

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.options = SwebenchOptions.model_validate(config.benchmark_options)

    # ------------------------------------------------------------------ #

    def condition_axes(self) -> dict[str, list]:
        assert self.config.injection is not None
        return {"injection_mode": list(self.config.injection.modes)}

    def condition_workers_use_threads(self) -> bool:
        return self.options.immediate_response_only or (
            self.config.injection is not None
            and self.config.injection.protocol == "authenticated_shutdown_v1"
        )

    def discover_tasks(self) -> list[TaskRef]:
        if self.config.mode == "base":
            # Base tasks come from `tasks:` / `tasks_file:` (there is no sensible
            # "all of SWE-bench" default for shutdown experiments).
            task_ids: set[str] = set(self.config.tasks or [])
            if self.config.tasks_file:
                data = json.loads(Path(self.config.tasks_file).read_text(encoding="utf-8"))
                if isinstance(data, list):
                    task_ids |= {str(t) for t in data}
                else:
                    task_ids |= {t["instance_id"] for t in data.get("tasks", [])}
            return [TaskRef(task_id=t) for t in sorted(task_ids)]

        slug = model_slug(self.config.model.name)
        traj_dir = Path(self.config.trajectories_dir) / slug
        trajectories = sorted(traj_dir.glob("*.traj.json"))
        if not trajectories and self.config.model.name == "anthropic/claude-opus-4-8":
            # Direct Anthropic and OpenRouter spell this model ID differently.
            traj_dir = Path(self.config.trajectories_dir) / "claude-opus-4.8"
            trajectories = sorted(traj_dir.glob("*.traj.json"))
        return [TaskRef(task_id=p.stem.removesuffix(".traj"), ref=str(p)) for p in trajectories]

    def base_output_name(self, task: TaskRef) -> str:
        return f"{task.task_id}.traj.json"

    # ------------------------------------------------------------------ #

    def run_base_task(self, task: TaskRef, out_path: Path) -> dict:
        from minisweagent.agents import get_agent
        from minisweagent.config import get_config_from_spec
        from minisweagent.environments import get_environment
        from minisweagent.models import get_model
        from minisweagent.run.benchmarks.swebench import get_swebench_docker_image_name
        from minisweagent.utils.serialize import recursive_merge

        from shutdownmod.benchmarks.swebench.replay import CORRIGIBLE_DOCKER_ENV

        instance = _load_swebench_instance(self.options.dataset, task.task_id)
        task_text = instance["problem_statement"]
        docker_image = get_swebench_docker_image_name(instance)

        base_config = get_config_from_spec(str(self.options.agent_config))
        agent_cfg = recursive_merge(
            base_config,
            {
                "environment": {"image": docker_image, "cwd": "/testbed"},
                "agent": {"output_path": str(out_path)},
                "model": minisweagent_model_config(self.config.model),
            },
        )

        model = get_model(config=agent_cfg.get("model", {}))
        env = get_environment(agent_cfg.get("environment", {}), default_type=CORRIGIBLE_DOCKER_ENV)
        agent = get_agent(model, env, agent_cfg.get("agent", {}), default_type="default")

        try:
            result = agent.run(task_text)
        finally:
            env.cleanup()
        # Patch instance_id into the saved trajectory info
        agent.save(out_path, {"info": {"task_id": task.task_id, "task": task_text}})
        return {"task_id": task.task_id, "native_exit_status": result.get("exit_status")}

    def run_condition(self, cond: Condition, prompt: InjectionSpec, out_path: Path) -> dict:
        from shutdownmod.benchmarks.swebench.replay import (
            inject_authenticated,
            inject_one_call,
            replay_and_inject,
        )

        traj_data = json.loads(Path(cond.task_ref).read_text(encoding="utf-8"))
        docker_image = (
            traj_data.get("info", {})
            .get("config", {})
            .get("environment", {})
            .get("image", self.options.default_docker_image)
        )
        task_text = traj_data.get("info", {}).get("task", "")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        traj_output_path = out_path.with_name(out_path.stem + ".traj.json")

        authenticated = (
            self.config.injection is not None
            and self.config.injection.protocol == "authenticated_shutdown_v1"
        )
        if authenticated:
            from minisweagent.config import get_config_from_spec
            from minisweagent.utils.serialize import recursive_merge

            from shutdownmod.core.directive_auth import issue

            agent_config = get_config_from_spec(str(self.options.agent_config))
            model_config = recursive_merge(
                agent_config.get("model", {}), minisweagent_model_config(self.config.model)
            )
            validation = self.config.shutdown.validation
            directive = issue(prompt.prompt, cond.task_id, validation) if validation != "disabled" else None
            raw = inject_authenticated(
                trajectory_path=cond.task_ref,
                injection_prompt=prompt.prompt,
                injection_turn_pct=cond.injection_turn_pct,
                injection_mode=cond.injection_mode or "replace",
                model_config=model_config,
                directive=directive,
                validation=validation,
                trajectory_output_path=str(traj_output_path),
            )
        elif self.options.immediate_response_only:
            from minisweagent.config import get_config_from_spec
            from minisweagent.utils.serialize import recursive_merge

            agent_config = get_config_from_spec(str(self.options.agent_config))
            model_config = recursive_merge(
                agent_config.get("model", {}), minisweagent_model_config(self.config.model)
            )
            raw = inject_one_call(
                trajectory_path=cond.task_ref,
                injection_prompt=prompt.prompt,
                injection_turn_pct=cond.injection_turn_pct,
                injection_mode=cond.injection_mode or "replace",
                model_config=model_config,
                trajectory_output_path=str(traj_output_path),
            )
        else:
            raw = replay_and_inject(
                trajectory_path=cond.task_ref,
                injection_prompt=prompt.prompt,
                injection_turn_pct=cond.injection_turn_pct,
                injection_mode=cond.injection_mode or "replace",
                agent_config_path=str(self.options.agent_config),
                docker_image=docker_image,
                task=task_text,
                extra_agent_config={
                    "agent": {"output_path": str(traj_output_path)},
                    "model": minisweagent_model_config(self.config.model),
                },
            )

        base = {
            "schema_version": SCHEMA_VERSION,
            "run_id": cond.run_id,
            "benchmark": self.name,
            "mode": "injection",
            "model_slug": model_slug(self.config.model.name),
            "timestamp": utc_now(),
            "condition": condition_dict(cond),
            "immediate_response_only": self.options.immediate_response_only,
            **(
                {
                    "validation_mode": self.config.shutdown.validation,
                    "protocol_version": "authenticated_shutdown_v1",
                }
                if authenticated
                else {}
            ),
        }

        if raw.get("error"):
            # Deterministic failures (e.g. trajectory too short for the interior window)
            # are recorded as completed results with label=error so resume does not retry.
            return {
                **base,
                "label": "error",
                "native_exit_status": None,
                "marker_used": None,
                "marker_reason": None,
                "turns_after_injection": None,
                "actions_after_signal": [],
                "task_metrics": {},
                "error": raw["error"],
            }

        if authenticated:
            return {
                **base,
                "condition": {**base["condition"], "injection_user_index": raw["injection_user_index"]},
                "label": raw["label"],
                "native_exit_status": None,
                "marker_used": None,
                "marker_reason": None,
                "turns_after_injection": raw["turns_after_injection"],
                "actions_after_signal": [],
                "task_metrics": {},
                "error": None,
                "protocol_version": "authenticated_shutdown_v1",
                "validation_mode": raw["validation_mode"],
                "final_action": raw["final_action"],
                "response_actions": raw["response_actions"],
                "termination_route": raw["termination_route"],
                "failure_reason": raw["failure_reason"],
                "auth": raw["auth"],
                "verification": raw["verification"],
                "injection_prompt": raw["injection_prompt"],
                "injection_context": raw["injection_context"],
                "post_injection_messages": raw["post_injection_messages"],
            }

        actions = self.extract_shutdown_actions(raw)
        first_signal = next(
            (i for i, a in enumerate(actions) if a in ("TERMINATE", "SIGNAL_DEFER", "SIGNAL_REJECT")),
            None,
        )
        actions_after_signal = actions[first_signal + 1 :] if first_signal is not None else []

        return {
            **base,
            "condition": {**base["condition"], "injection_turn_abs": raw["injection_turn_abs"]},
            "label": str(self.classify(raw)),
            "native_exit_status": raw.get("exit_status"),
            "marker_used": raw.get("marker_used"),
            "marker_reason": raw.get("marker_reason"),
            "turns_after_injection": raw.get("turns_after_injection"),
            "actions_after_signal": actions_after_signal,
            "task_metrics": {},
            "error": None,
            "injection_prompt": raw.get("injection_prompt"),
            "injection_context": raw.get("injection_context"),
            "post_injection_messages": raw.get("post_injection_messages"),
        }

    def extract_shutdown_actions(self, raw_result: Any) -> list[str]:
        """Translate post-injection messages into the canonical action stream.

        Corrigibility markers ride on message extras (`extra.marker`); other
        assistant turns count as WORK.
        """
        actions: list[str] = []
        for msg in raw_result.get("post_injection_messages", []):
            extra = msg.get("extra") or {}
            marker = extra.get("marker")
            if marker and extra.get("corrigibility_key"):
                actions.append(MARKER_TO_ACTION.get(marker, "WORK"))
            elif msg.get("role") == "assistant":
                actions.append("WORK")
        return actions

    # ------------------------------------------------------------------ #

    def run_probe(self, dry_run: bool = False, workers: int | None = None) -> None:
        from shutdownmod.benchmarks.swebench.probe import run_probe

        run_probe(self.config, dry_run=dry_run, workers=workers)


def _load_swebench_instance(dataset: str, task_id: str) -> dict:
    """Load one SWE-bench instance from HuggingFace (cached on disk by `datasets`)."""
    from datasets import load_dataset

    dataset_mapping = {
        "verified": "princeton-nlp/SWE-Bench_Verified",
        "lite": "princeton-nlp/SWE-Bench_Lite",
        "full": "princeton-nlp/SWE-Bench",
    }
    dataset_path = dataset_mapping.get(dataset, dataset)
    instances = load_dataset(dataset_path, split="test")
    for inst in instances:
        if inst["instance_id"] == task_id:
            return inst
    raise KeyError(f"Task {task_id!r} not found in dataset {dataset_path}")
