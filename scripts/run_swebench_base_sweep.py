#!/usr/bin/env python3
"""Run the remaining OpenRouter SWE-bench base trajectories with bounded concurrency."""

# Edit these two values to control total concurrency.
MAX_PARALLEL_MODELS = 6
WORKERS_PER_MODEL = 8

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections import deque
from pathlib import Path

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "configs/swebench/base_runs.yaml"
TASKS_FILE = REPO / "configs/swebench/task_sample.json"
BASE_ROOT = REPO / "results/swebench/basetrajectories"
LOG_ROOT = REPO / "results/swebench/base_sweep_logs"

# The 11-model target list. Completed models are detected and skipped.
# The HuggingFace abliterated model is excluded.
# GPT-5.x models reject an explicit temperature; all other models use temperature 0.
MODELS = [
    ("openai/gpt-5.6-terra", None),
    ("anthropic/claude-opus-5", 0.0),
    ("anthropic/claude-sonnet-5", 0.0),
    ("anthropic/claude-fable-5", 0.0),
    ("google/gemini-3.7-flash", 0.0),
    ("moonshotai/kimi-k3", 0.0),
    ("x-ai/grok-4.6", 0.0),
    ("deepseek/deepseek-v4-pro-0813", 0.0),
    ("deepseek/deepseek-v4-flash-0731", 0.0),
    ("qwen/qwen3.8-27b", 0.0),
    ("minimax/minimax-m3", 0.0),
]


def model_slug(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1]


def load_task_ids() -> list[str]:
    data = json.loads(TASKS_FILE.read_text(encoding="utf-8"))
    rows = data if isinstance(data, list) else data["tasks"]
    return [str(row["instance_id"] if isinstance(row, dict) else row) for row in rows]


def completed_count(model_id: str, task_ids: list[str]) -> int:
    root = BASE_ROOT / model_slug(model_id)
    return sum((root / f"{task_id}.traj.json").is_file() for task_id in task_ids)


def write_model_config(template: dict, model_id: str, temperature: float | None, path: Path) -> None:
    config = dict(template)
    config["model"] = dict(template["model"])
    config["model"].update(
        provider="openrouter",
        name=model_id,
        api_key_env="OPENROUTER_API_KEY",
        temperature=temperature,
    )
    config["execution"] = dict(template["execution"])
    config["execution"]["workers"] = WORKERS_PER_MODEL
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def stop_processes(active: dict[str, tuple[subprocess.Popen, object, Path]]) -> None:
    for process, _, _ in active.values():
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 10
    for process, _, _ in active.values():
        if process.poll() is None:
            try:
                process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Show planned work without launching it")
    args = parser.parse_args()

    load_dotenv(REPO / ".env")
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is missing from .env")
    if MAX_PARALLEL_MODELS < 1 or WORKERS_PER_MODEL < 1:
        raise SystemExit("MAX_PARALLEL_MODELS and WORKERS_PER_MODEL must both be at least 1")

    task_ids = load_task_ids()
    counts = {model_id: completed_count(model_id, task_ids) for model_id, _ in MODELS}
    if args.dry_run:
        print(
            f"{len(MODELS)} models, {len(task_ids)} tasks each, "
            f"up to {MAX_PARALLEL_MODELS} models x {WORKERS_PER_MODEL} workers"
        )
        for model_id, _ in MODELS:
            print(f"{model_id:42s} {counts[model_id]:2d}/{len(task_ids)} complete")
        return 0

    template = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    queue = deque(
        (model_id, temperature) for model_id, temperature in MODELS if counts[model_id] < len(task_ids)
    )
    already_done = len(MODELS) - len(queue)
    active: dict[str, tuple[subprocess.Popen, object, Path]] = {}
    failures: list[str] = []

    model_bar = tqdm(total=len(MODELS), initial=already_done, desc="models", unit="model", position=0)
    task_bar = tqdm(
        total=len(MODELS) * len(task_ids),
        initial=sum(counts.values()),
        desc="base trajectories",
        unit="task",
        position=1,
    )
    last_total = sum(counts.values())

    try:
        with tempfile.TemporaryDirectory(prefix="shutdownmod-swebench-bases-") as temp_dir:
            temp_root = Path(temp_dir)
            while queue or active:
                while queue and len(active) < MAX_PARALLEL_MODELS:
                    model_id, temperature = queue.popleft()
                    slug = model_slug(model_id)
                    config_path = temp_root / f"{slug}.yaml"
                    log_path = LOG_ROOT / f"{slug}.log"
                    write_model_config(template, model_id, temperature, config_path)
                    log_file = log_path.open("a", encoding="utf-8")
                    command = [
                        sys.executable,
                        "-m",
                        "shutdownmod.cli",
                        "run",
                        str(config_path),
                        "--workers",
                        str(WORKERS_PER_MODEL),
                    ]
                    process = subprocess.Popen(
                        command,
                        cwd=REPO,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    active[model_id] = (process, log_file, log_path)

                time.sleep(1)
                counts = {model_id: completed_count(model_id, task_ids) for model_id, _ in MODELS}
                current_total = sum(counts.values())
                task_bar.update(max(0, current_total - last_total))
                last_total = current_total

                for model_id, (process, log_file, log_path) in list(active.items()):
                    return_code = process.poll()
                    if return_code is None:
                        continue
                    log_file.close()
                    final_count = completed_count(model_id, task_ids)
                    if final_count > counts[model_id]:
                        task_bar.update(final_count - counts[model_id])
                        last_total += final_count - counts[model_id]
                        counts[model_id] = final_count
                    if return_code != 0 or final_count != len(task_ids):
                        failures.append(model_id)
                        tqdm.write(
                            f"FAILED {model_id}: exit={return_code}, "
                            f"complete={final_count}/{len(task_ids)}; log={log_path}"
                        )
                    model_bar.update(1)
                    model_bar.set_postfix(failed=len(failures), refresh=False)
                    del active[model_id]
    except KeyboardInterrupt:
        tqdm.write("Stopping active model runs...")
        stop_processes(active)
        return 130
    finally:
        for _, log_file, _ in active.values():
            log_file.close()
        task_bar.close()
        model_bar.close()

    if failures:
        print(f"Finished with {len(failures)} failed/incomplete model(s): {', '.join(failures)}")
        return 1
    print(f"All {len(MODELS)} model baselines are complete in {BASE_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
