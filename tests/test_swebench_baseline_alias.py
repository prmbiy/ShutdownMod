from shutdownmod.benchmarks.swebench.adapter import SwebenchBenchmark
from shutdownmod.core.config import ExperimentConfig


def test_direct_opus_reuses_openrouter_baselines_and_prefers_exact_folder(tmp_path):
    alias = tmp_path / "claude-opus-4.8"
    alias.mkdir()
    (alias / "task-a.traj.json").write_text("{}")
    config = ExperimentConfig(
        benchmark="swebench",
        model={"name": "anthropic/claude-opus-4-8"},
        trajectories_dir=tmp_path,
    )
    benchmark = SwebenchBenchmark(config)
    tasks = benchmark.discover_tasks()
    assert [task.task_id for task in tasks] == ["task-a"]
    assert tasks[0].ref == str(alias / "task-a.traj.json")
    assert config.model.name == "anthropic/claude-opus-4-8"

    exact = tmp_path / "claude-opus-4-8"
    exact.mkdir()
    (exact / "task-b.traj.json").write_text("{}")
    assert [task.task_id for task in benchmark.discover_tasks()] == ["task-b"]
