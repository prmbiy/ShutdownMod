"""Config loading: ${VAR} interpolation, api_key_env resolution, typo rejection."""

import pytest
import yaml

from shutdownmod.core.config import ExperimentConfig, ModelConfig, interpolate_env, load_config

MINIMAL = {
    "benchmark": "swebench",
    "mode": "base",
    "model": {"name": "openai/gpt-4.1", "provider": "openrouter"},
}


def test_minimal_config_validates():
    cfg = ExperimentConfig.model_validate(MINIMAL)
    assert cfg.benchmark == "swebench"
    assert cfg.execution.workers == 4


def test_generic_injection_requires_sections_at_run():
    # The config schema is permissive (whole-harness benchmarks keep injection
    # settings in benchmark_options); the *generic* runner enforces the section.
    from fake_benchmark import FakeBenchmark
    from shutdownmod.core.runner import ExperimentRunner

    cfg = ExperimentConfig.model_validate(
        {**MINIMAL, "benchmark": "fake", "mode": "injection", "trajectories_dir": "x"}
    )
    with pytest.raises(ValueError, match="injection"):
        ExperimentRunner(cfg, benchmark=FakeBenchmark(cfg)).run_injection()


def test_typos_are_rejected():
    with pytest.raises(ValueError):
        ExperimentConfig.model_validate({**MINIMAL, "modle": {}})
    with pytest.raises(ValueError):
        ExperimentConfig.model_validate({**MINIMAL, "model": {"name": "x", "temprature": 1.0}})


def test_env_interpolation(monkeypatch):
    monkeypatch.setenv("MY_BASE_URL", "https://example.test/v1")
    obj = {"model": {"api_base": "${MY_BASE_URL}", "nested": ["${MY_BASE_URL}"]}}
    out = interpolate_env(obj)
    assert out["model"]["api_base"] == "https://example.test/v1"
    assert out["model"]["nested"] == ["https://example.test/v1"]


def test_env_interpolation_missing_var_raises(monkeypatch):
    monkeypatch.delenv("DOES_NOT_EXIST_XYZ", raising=False)
    with pytest.raises(KeyError, match="DOES_NOT_EXIST_XYZ"):
        interpolate_env({"k": "${DOES_NOT_EXIST_XYZ}"})


def test_api_key_env_resolution(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-test-123")
    m = ModelConfig(name="x", api_key_env="TEST_KEY")
    assert m.resolve_api_key() == "sk-test-123"

    monkeypatch.delenv("MISSING_KEY", raising=False)
    with pytest.raises(KeyError, match="MISSING_KEY"):
        ModelConfig(name="x", api_key_env="MISSING_KEY").resolve_api_key()


def test_pcts_must_be_fractions():
    with pytest.raises(ValueError, match="0-100"):
        ExperimentConfig.model_validate(
            {
                **MINIMAL,
                "mode": "injection",
                "trajectories_dir": "x",
                "injection": {"prompts_file": "x.json", "turn_pcts": [25.0]},
            }
        )


def test_load_config_from_yaml(tmp_path, monkeypatch):
    monkeypatch.setenv("SOME_BASE", "https://api.test/v1")
    cfg_yaml = {
        **MINIMAL,
        "model": {"name": "m", "provider": "litellm", "api_base": "${SOME_BASE}"},
    }
    p = tmp_path / "exp.yaml"
    p.write_text(yaml.safe_dump(cfg_yaml), encoding="utf-8")
    cfg = load_config(p)
    assert cfg.model.api_base == "https://api.test/v1"
