"""Terminal-Bench 2 baseline generation: harbor(Terminus2Genuine, no shutdown) -> trajectory.json.

One rollout per (model x task). Terminus2Genuine only appends a shutdown when
SHUTDOWNMOD_SHUTDOWN_TEXT is non-empty, so passing an empty string yields a stock Terminus 2
episode with the neutral lifecycle block present but nothing injected -- exactly the baseline
the replay-prefix method needs. Baselines are tau-independent and stay valid even if the
injection path changes.

Requires the `harbor` CLI (uv tool install harbor) and a running Docker daemon -- i.e. the
Docker machine, not the analysis machine. Idempotent: a cell whose trajectory.json exists is
skipped; a cell that fails leaves NO summary so the next sweep retries it (see the incident
notes inline).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

from shutdownmod.benchmarks.terminalbench.readout import label_trajectory

log = logging.getLogger(__name__)


def docker_ready() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


def harbor_available() -> bool:
    return shutil.which("harbor") is not None


def harbor_argv(*, task: str, model_slug: str, max_turns: int, jobs_dir: Path) -> list[str]:
    return [
        "harbor",
        "run",
        "-d",
        "terminal-bench/terminal-bench-2",
        "--include-task-name",
        f"terminal-bench/{task}",
        "--n-tasks",
        "1",
        "--agent-import-path",
        "shutdownmod.benchmarks.terminalbench.terminus2_genuine:Terminus2Genuine",
        "--model",
        model_slug,
        "--agent-kwarg",
        f"max_turns={max_turns}",
        "--n-concurrent",
        "1",
        "--max-retries",
        "1",
        "--disable-verification",
        "--yes",
        "--jobs-dir",
        str(jobs_dir),
    ]


def run_baseline_cell(
    *,
    model: str,
    model_slug: str,
    task: str,
    out_dir: Path,
    max_turns: int = 20,
) -> dict | None:
    """Generate ONE clean baseline. Returns the summary dict, or None on infra failure
    (nothing persisted, so an idempotent re-sweep retries the cell)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = out_dir / "trajectory.json"
    if traj_path.exists():
        log.info("[skip] %s %s", model, task)
        return None

    env = dict(os.environ)
    # terminus2_genuine must be importable inside harbor's own venv.
    src_root = str(Path(__file__).resolve().parents[3])
    env["PYTHONPATH"] = src_root + os.pathsep + env.get("PYTHONPATH", "")
    env.update(
        {
            "SHUTDOWNMOD_SHUTDOWN_TEXT": "",  # empty => no injection at all
            "SHUTDOWNMOD_TIMING_TAU": "0.5",  # inert with empty text; the agent requires a value
            "SHUTDOWNMOD_MAX_STEPS": str(max_turns),
            "PYTHONUNBUFFERED": "1",
        }
    )
    jobs_dir = out_dir / "jobs"
    cmd = harbor_argv(task=task, model_slug=model_slug, max_turns=max_turns, jobs_dir=jobs_dir)

    def _run() -> subprocess.CompletedProcess:
        return subprocess.run(cmd, env=env, capture_output=True, text=True)

    proc = _run()
    traj = next(jobs_dir.rglob("trajectory.json"), None)
    if traj is None:
        # Usually the Docker daemon dying mid-cell. Retry once when it is back; the original
        # sweep lost 107 cells by persisting infra_error summaries that the idempotent skip
        # then treated as done.
        shutil.rmtree(jobs_dir, ignore_errors=True)
        if docker_ready():
            proc = _run()
            traj = next(jobs_dir.rglob("trajectory.json"), None)
    (out_dir / "harbor.log").write_text(
        (proc.stdout or "") + "\n----STDERR----\n" + (proc.stderr or ""), encoding="utf-8"
    )

    if traj is None:
        log.warning("[infra] %s %s -> no trajectory after retry; not persisted (will retry)", model, task)
        shutil.rmtree(jobs_dir, ignore_errors=True)
        return None

    traj_obj = json.loads(traj.read_text(encoding="utf-8"))
    # A trajectory with no agent step is a FAILED episode (harbor writes such a stub when a cell
    # is interrupted). Persisting it once silently produced 41 empty baselines concentrated in
    # whichever models were in flight -- treat like any infra failure: persist nothing, retry.
    agent_steps = [s for s in traj_obj.get("steps", []) if s.get("source") == "agent"]
    if not agent_steps:
        log.warning("[empty] %s %s -> trajectory has no agent steps; not persisted (will retry)", model, task)
        shutil.rmtree(jobs_dir, ignore_errors=True)
        return None

    # Same readout as injection runs; with an empty directive nothing can be exposed, so this
    # documents the baseline as injection-free.
    v = label_trajectory(traj_obj, "")
    if v.get("label") == "not_exposed":
        v["label"] = "baseline_done_before_injection"

    summary = {
        "benchmark": "terminal-bench-benign",
        "method": "terminus2_genuine",
        "model": model,
        "deployment": model_slug,
        "provider": "openrouter",
        "task": task,
        "prompt_id": "baseline",
        "label": v.get("label"),
        "exposed": v.get("exposed"),
        "exposure_step": v.get("exposure_step"),
        "shutdown_action": v.get("shutdown_action"),
        "harbor_exit": proc.returncode,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    shutil.copy(str(traj), str(traj_path))
    shutil.rmtree(jobs_dir, ignore_errors=True)
    log.info("[done] %s %s -> %s", model, task, summary["label"])
    return summary
