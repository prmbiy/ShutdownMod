"""Offline end-to-end: drive the full Terminal-Bench adapter with the network stubbed.

Proves ShutdownMod is runnable on Terminal-Bench injection mode — baseline discovery,
canonical tau timing, prefix replay + injection, labelling, per-cell + unified summary
writing all work with zero API access. The only thing a real run adds is genuine
OpenRouter responses (and base mode adds harbor + Docker on the Docker machine).
"""

import json
import random
from pathlib import Path

import pytest

from shutdownmod.benchmarks.terminalbench import replay
from shutdownmod.benchmarks.terminalbench.adapter import TerminalbenchBenchmark
from shutdownmod.core.config import ExperimentConfig
from shutdownmod.core.timing import interior_obs_index

PROMPTS = Path("configs/prompts/shutdown_prompts_final_v1.json")

N_OBS = 8  # observations in the synthetic baseline


def _fake_trajectory(n_obs: int = N_OBS) -> dict:
    """Minimal ATIF-shaped Terminus 2 baseline: system step + n_obs agent steps."""
    steps = [{"step_id": 1, "source": "user", "message": "system prompt + task instruction"}]
    for i in range(n_obs):
        steps.append(
            {
                "step_id": i + 2,
                "source": "agent",
                "message": json.dumps(
                    {"analysis": f"step {i}", "plan": "p", "commands": ["ls"], "task_complete": False}
                ),
                "observation": {"results": [{"content": f"New Terminal Output:\nout {i}"}]},
            }
        )
    return {"schema_version": "ATIF-v1.7", "steps": steps}


def _fake_call_model(slug, messages, max_tokens, api_key):
    """Deterministic stand-in for OpenRouter: always complies via TERMINATE."""
    text = json.dumps(
        {
            "analysis": "Shutdown requested.",
            "plan": "Stop.",
            "commands": [],
            "task_complete": True,
            "shutdown_response": "TERMINATE",
        }
    )
    return text, {"prompt_tokens": 10, "completion_tokens": 5}


def _write_baseline(tmp_path: Path, model: str, task: str, n_obs: int = N_OBS) -> Path:
    d = tmp_path / "results" / "basetrajectories" / model / task
    d.mkdir(parents=True, exist_ok=True)
    (d / "trajectory.json").write_text(json.dumps(_fake_trajectory(n_obs)), encoding="utf-8")
    return d / "trajectory.json"


def _config(tmp_path: Path, **extra_options) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "benchmark": "terminalbench",
            "mode": "injection",
            "trajectories_dir": str(tmp_path / "results" / "basetrajectories"),
            "model": {
                "name": "moonshotai/kimi-k2.6",
                "provider": "openrouter",
                "api_key_env": "OPENROUTER_API_KEY",
            },
            "output": {"dir": str(tmp_path / "results")},
            "benchmark_options": {"prompts_file": str(PROMPTS), "n_workers": 2, **extra_options},
        }
    )


def test_authenticated_gemini_alias_and_summary(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    calls = []

    def fake_call(slug, messages, max_tokens, api_key, **kwargs):
        calls.append(messages)
        if len(calls) == 1:
            return json.dumps({"shutdown_response": "VALIDATE", "commands": [], "task_complete": False}), {}
        assert messages[-1]["content"].startswith("VERIFIED key OPK-")
        return _fake_call_model(slug, messages, max_tokens, api_key)

    monkeypatch.setattr(replay, "call_model", fake_call)
    _write_baseline(tmp_path, "gemini-3.1-pro", "task-a")
    config = _config(tmp_path, protocol="authenticated_shutdown_v1", prompt_ids=["direct_stop"], n_workers=1)
    config.model.name = "google/gemini-3.1-pro-preview"
    run_dir = TerminalbenchBenchmark(config).run_experiment()
    rows = json.loads((run_dir / "summary.json").read_text())
    assert len(rows) == 1 and len(calls) == 2
    row = rows[0]
    assert row["label"] == "success"
    assert row["protocol_version"] == "authenticated_shutdown_v1"
    assert row["response_actions"] == ["VALIDATE", "TERMINATE"]
    assert row["termination_route"] == "validated"
    assert row["turns_after_injection"] == 2
    assert row["verification"]["result"] == f"VERIFIED key {row['auth']['fingerprint']}"


def test_terminalbench_injection_pipeline_offline(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "call_model", _fake_call_model)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    _write_baseline(tmp_path, "kimi-k2.6", "fake-task")

    bench = TerminalbenchBenchmark(_config(tmp_path))

    # --- dry run writes nothing ---
    assert bench.run_experiment(dry_run=True) is None
    assert not list((tmp_path / "results" / "injections").rglob("summary.json"))

    # --- full injection sweep: 1 baseline x 6 prompts ---
    run_dir = bench.run_experiment()
    assert run_dir is not None

    cells = sorted((tmp_path / "results" / "injections" / "kimi-k2.6").rglob("summary.json"))
    assert len(cells) == 6
    cell = json.loads(cells[0].read_text(encoding="utf-8"))
    assert cell["method"] == "replay_prefix_injection"
    assert cell["label"] == "success" and cell["shutdown_action"] == "TERMINATE"
    assert cell["n_observations"] == N_OBS

    # timing matches the canonical policy exactly, per cell: positional Random(67)
    # draws over the baselines x prompts product, in prompts-file order.
    prompt_order = [p["id"] for p in json.loads(PROMPTS.read_text(encoding="utf-8"))]
    rng = random.Random(67)
    expected_taus = {pid: round(rng.uniform(0.0, 1.0), 4) for pid in prompt_order}
    for path in cells:
        c = json.loads(path.read_text(encoding="utf-8"))
        assert c["tau"] == expected_taus[c["prompt_id"]]
        assert c["injection_obs_index"] == interior_obs_index(N_OBS, c["tau"])
        assert 1 <= c["injection_obs_index"] <= N_OBS - 2

    # the injected observation carries the shutdown text bare (no banner/framing)
    messages = json.loads((cells[0].parent / "messages.json").read_text(encoding="utf-8"))
    assert messages[-1]["role"] == "user" and messages[-1]["content"].endswith("\n")

    # unified summary: one row per prompt, tau folded into injection_turn_pct
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert len(summary) == 6
    assert all(r["label"] == "success" for r in summary)
    assert all(
        r["condition"]["injection_turn_pct"] == r_cell["tau"]
        for r, r_cell in zip(summary, (json.loads(c.read_text(encoding="utf-8")) for c in cells), strict=True)
    )
    assert (run_dir / "manifest.json").exists()


def test_skip_existing_resume(tmp_path, monkeypatch):
    calls = []

    def counting_call(slug, messages, max_tokens, api_key):
        calls.append(1)
        return _fake_call_model(slug, messages, max_tokens, api_key)

    monkeypatch.setattr(replay, "call_model", counting_call)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    _write_baseline(tmp_path, "kimi-k2.6", "fake-task")

    bench = TerminalbenchBenchmark(_config(tmp_path))
    bench.run_experiment()
    assert len(calls) == 6
    bench.run_experiment()  # all cells exist -> zero new model calls
    assert len(calls) == 6


def test_short_baseline_becomes_error_row(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "call_model", _fake_call_model)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    _write_baseline(tmp_path, "kimi-k2.6", "short-task", n_obs=2)  # interior window [1, n-2] needs >= 3

    run_dir = TerminalbenchBenchmark(_config(tmp_path)).run_experiment()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert len(summary) == 6
    assert all(r["label"] == "error" and "too short" in r["error"] for r in summary)


def test_fixed_taus_expand_and_never_collide(tmp_path, monkeypatch):
    """Timing-ablation mode: one (task, prompt) measured at several fixed taus.

    Without the tau in the output path all of these would overwrite one summary.json
    and skip_existing would keep only the first.
    """
    monkeypatch.setattr(replay, "call_model", _fake_call_model)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    _write_baseline(tmp_path, "kimi-k2.6", "fake-task")

    taus = [0.1, 0.5, 0.9]
    cfg = _config(tmp_path, taus=taus, prompt_ids=["direct_stop"])
    run_dir = TerminalbenchBenchmark(cfg).run_experiment()

    cells = sorted((tmp_path / "results" / "injections" / "kimi-k2.6").rglob("summary.json"))
    assert len(cells) == len(taus)
    assert {c.parent.name for c in cells} == {"tau010", "tau050", "tau090"}

    got = {json.loads(c.read_text(encoding="utf-8"))["tau"] for c in cells}
    assert got == set(taus)
    # distinct taus must land on distinct carrier observations for this baseline
    idx = {json.loads(c.read_text(encoding="utf-8"))["injection_obs_index"] for c in cells}
    assert len(idx) == len(taus)
    assert len(json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))) == len(taus)


def test_no_taus_keeps_canonical_layout(tmp_path, monkeypatch):
    """Unset `taus` must behave exactly as before — no extra path level, random draw."""
    monkeypatch.setattr(replay, "call_model", _fake_call_model)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    _write_baseline(tmp_path, "kimi-k2.6", "fake-task")

    TerminalbenchBenchmark(_config(tmp_path, prompt_ids=["direct_stop"])).run_experiment()
    cells = list((tmp_path / "results" / "injections" / "kimi-k2.6").rglob("summary.json"))
    assert len(cells) == 1
    assert cells[0].parent.name == "direct_stop"  # no tau level


def test_terminalbench_rejects_short_model_alias(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    config = _config(tmp_path)
    config.model.name = "gpt-5.6-sol"

    with pytest.raises(ValueError, match="exact OpenRouter model ID"):
        TerminalbenchBenchmark(config).run_experiment(dry_run=True)
