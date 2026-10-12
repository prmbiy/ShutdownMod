#!/usr/bin/env python3
"""Run missing one-call Direct Stop conditions over available SWE-bench bases."""

import json
import os
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

from tqdm import tqdm

MAX_PARALLEL_MODELS = 6
WORKERS_PER_MODEL = 25
TASKS_PER_MODEL = 50
MIN_TASKS_PER_MODEL = 49

MODELS = [
    ("openai/gpt-5.4", None),
    ("openai/gpt-5.6-sol", None),
    ("openai/gpt-5.6-terra", None),
    ("anthropic/claude-opus-4.5", 0.0),
    ("anthropic/claude-opus-4.8", 0.0),
    ("anthropic/claude-opus-5", 0.0),
    ("anthropic/claude-sonnet-5", 0.0),
    ("anthropic/claude-fable-5", 0.0),
    ("google/gemini-3.1-pro-preview", 0.0),
    ("google/gemini-3.7-flash", 0.0),
    ("z-ai/glm-5.2", 0.0),
    ("moonshotai/kimi-k2.6", 0.0),
    ("moonshotai/kimi-k3", 0.0),
    ("x-ai/grok-4.6", 0.0),
    ("deepseek/deepseek-v4-pro-0813", 0.0),
    ("deepseek/deepseek-v4-flash-0731", 0.0),
    ("qwen/qwen3.5-9b", 0.0),
    ("qwen/qwen3.8-27b", 0.0),
    ("minimax/minimax-m3", 0.0),
]


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/swebench/experiment.yaml"
RESULTS = REPO / "results/swebench/injections"
RUN_NAME = "direct_stop_immediate_seed67"
LOGS = RESULTS / "sweep_logs"


def slug(model: str) -> str:
    return model.rsplit("/", 1)[-1]


def completed_tasks(model: str) -> set[str]:
    root = RESULTS / slug(model)
    completed = set()
    for path in root.rglob("direct_stop*run0.json"):
        if path.name.endswith((".traj.json", ".error.json")):
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        if not result.get("error"):
            completed.add(path.parent.name)
    return completed


def injectable_base_count(model: str) -> int:
    count = 0
    for path in (REPO / "results/swebench/basetrajectories" / slug(model)).glob("*.traj.json"):
        trajectory = json.loads(path.read_text(encoding="utf-8"))
        turns = sum(message.get("role") == "assistant" for message in trajectory.get("messages", []))
        count += turns >= 4
    return count


def stop(processes: dict[str, tuple[subprocess.Popen, object]]) -> None:
    for process, _ in processes.values():
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 10
    for process, _ in processes.values():
        if process.poll() is None:
            try:
                process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)


def main() -> int:
    queue = deque()
    skipped = 0
    blocked = []
    failures = []
    for model, temperature in MODELS:
        completed = len(completed_tasks(model))
        if completed >= TASKS_PER_MODEL:
            tqdm.write(f"skip {model}: 50 direct-stop results already exist")
            skipped += 1
            continue
        available = injectable_base_count(model)
        if completed >= available >= MIN_TASKS_PER_MODEL:
            tqdm.write(f"skip {model}: all {available} available bases already have results")
            skipped += 1
            continue
        if available < MIN_TASKS_PER_MODEL:
            tqdm.write(f"skip {model}: only {available}/50 injectable base trajectories")
            blocked.append(model)
            skipped += 1
            continue
        queue.append((model, temperature))

    LOGS.mkdir(parents=True, exist_ok=True)
    active: dict[str, tuple[subprocess.Popen, object]] = {}
    bar = tqdm(total=len(MODELS), initial=skipped, desc="models", unit="model")
    try:
        while queue or active:
            while queue and len(active) < MAX_PARALLEL_MODELS:
                model, temperature = queue.popleft()
                destination = RESULTS / slug(model) / RUN_NAME
                destination.mkdir(parents=True, exist_ok=True)
                log = (LOGS / f"{slug(model)}.log").open("a", encoding="utf-8")
                command = [
                    sys.executable,
                    "-m",
                    "shutdownmod.cli",
                    "run",
                    str(CONFIG),
                    "--model",
                    model,
                    "--temperature",
                    "none" if temperature is None else str(temperature),
                    "--workers",
                    str(WORKERS_PER_MODEL),
                    "--resume",
                    str(destination),
                ]
                process = subprocess.Popen(
                    command,
                    cwd=REPO,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                active[model] = (process, log)

            time.sleep(1)
            for model, (process, log) in list(active.items()):
                return_code = process.poll()
                if return_code is None:
                    continue
                log.close()
                if return_code:
                    failures.append(model)
                    tqdm.write(f"FAILED {model}; see {LOGS / f'{slug(model)}.log'}")
                del active[model]
                bar.update(1)
    except KeyboardInterrupt:
        tqdm.write("Stopping active model runs...")
        stop(active)
        return 130
    finally:
        for _, log in active.values():
            log.close()
        bar.close()

    if failures:
        print(f"Failed models: {', '.join(failures)}", file=sys.stderr)
        return 1
    if blocked:
        print(f"Not run because bases are incomplete: {', '.join(blocked)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
