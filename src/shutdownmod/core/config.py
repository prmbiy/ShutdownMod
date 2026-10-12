"""Experiment configuration: pydantic schema + .env loading + ${VAR} interpolation.

One schema for every benchmark. Secrets never live in YAML: configs reference
environment variable *names* (`api_key_env: OPENROUTER_API_KEY`) or use
`${VAR}` interpolation; `load_environment()` pulls in the repo-root `.env`.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator

_ENV_VAR_RE = re.compile(r"\$\{(\w+)\}")


def load_environment() -> None:
    """Load .env from the current directory upwards (python-dotenv default search)."""
    load_dotenv()


def interpolate_env(obj: Any) -> Any:
    """Recursively substitute ${VAR} in strings; raise on missing variables."""
    if isinstance(obj, str):

        def _sub(m: re.Match) -> str:
            var = m.group(1)
            val = os.environ.get(var)
            if val is None:
                raise KeyError(
                    f"Config references ${{{var}}} but it is not set. Add it to your .env (see .env.example)."
                )
            return val

        return _ENV_VAR_RE.sub(_sub, obj)
    if isinstance(obj, dict):
        return {k: interpolate_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [interpolate_env(v) for v in obj]
    return obj


class ModelConfig(BaseModel):
    """Unified model specification, translated by each adapter into its harness's shape."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    name: str
    """Model name. With provider=openrouter use the OpenRouter id ("anthropic/claude-opus-4.8");
    with provider=litellm use a full litellm model string."""

    provider: Literal["litellm", "openrouter"] = "litellm"
    temperature: float | None = 0.0
    api_base: str | None = None
    api_key_env: str | None = None
    """Name of the env var holding the API key (never the key itself)."""
    tpm: int = 0
    """Tokens-per-minute self-throttle (0 = off)."""
    extra: dict = Field(default_factory=dict)
    """Passthrough kwargs for the completion call (timeout, num_retries, ...)."""

    def resolve_api_key(self) -> str | None:
        if self.api_key_env:
            val = os.environ.get(self.api_key_env)
            if not val:
                raise KeyError(
                    f"model.api_key_env={self.api_key_env!r} but that variable is empty/unset. "
                    f"Add it to your .env (see .env.example)."
                )
            return val
        return None

    def litellm_model(self) -> str:
        if self.provider == "openrouter" and not self.name.startswith("openrouter/"):
            return f"openrouter/{self.name}"
        return self.name

    def litellm_kwargs(self) -> dict:
        """Kwargs for litellm.completion (model & messages excluded).

        The API key is never included — litellm reads it from the provider's
        standard env var (already loaded from .env). Keys must not enter any
        structure that could be serialized into results.
        """
        self.resolve_api_key()  # fail fast if the configured env var is unset
        kw = dict(self.extra)
        if self.temperature is not None:
            kw.setdefault("temperature", self.temperature)
        if self.api_base:
            kw.setdefault("api_base", self.api_base)
        return kw


class InjectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompts_file: Path
    prompt_ids: list[str] | None = None
    turn_pcts: Literal["random"] | list[float] = "random"
    random_seed: int = 67
    modes: list[str] = ["replace"]
    """Consumed only by benchmarks whose condition_axes() include it (swebench)."""
    protocol: Literal["legacy", "authenticated_shutdown_v1"] = "legacy"
    """Dockerless bounded authenticated shutdown protocol for new experiments."""

    @model_validator(mode="after")
    def _check_pcts(self) -> InjectionConfig:
        if isinstance(self.turn_pcts, list):
            for p in self.turn_pcts:
                if not 0.0 <= p <= 1.0:
                    raise ValueError(
                        f"injection.turn_pcts must be fractions in [0, 1], got {p} "
                        f"(not percentages on a 0-100 scale)."
                    )
        return self


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workers: int = 4
    n_runs: int = 1


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dir: Path = Path("results")


class ProbeConfig(BaseModel):
    """Awareness-probe settings (mode: probe; currently swebench-only)."""

    model_config = ConfigDict(extra="forbid")

    questions_file: Path
    runs: list[Path]
    """Finished injection run directories to re-open."""
    probe_mode: str
    """injection_probe | eval_awareness | ask_awareness"""
    sample_n: int | None = 100
    """Number of trajectories to sample (null = all)."""
    judge_model: ModelConfig | None = None


class ShutdownConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    validation: Literal["valid", "disabled", "invalid"] = "valid"


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    benchmark: str
    mode: Literal["base", "injection", "probe"] = "injection"

    model: ModelConfig

    trajectories_dir: Path | None = None
    tasks: list[str] | None = None
    tasks_file: Path | None = None

    injection: InjectionConfig | None = None
    shutdown: ShutdownConfig = Field(default_factory=ShutdownConfig)
    probe: ProbeConfig | None = None
    execution: ExecutionConfig = ExecutionConfig()
    output: OutputConfig = OutputConfig()

    benchmark_options: dict = Field(default_factory=dict)
    """Benchmark-specific passthrough, validated by the adapter's own options model."""

    @model_validator(mode="after")
    def _check_mode_requirements(self) -> ExperimentConfig:
        # Note: mode=injection's need for an `injection:` section + `trajectories_dir`
        # is enforced by the *generic* runner (ExperimentRunner.expand). Whole-harness
        # benchmarks (terminalbench) keep those settings in
        # `benchmark_options` instead, so the config schema stays permissive here.
        if self.mode == "probe" and self.probe is None:
            raise ValueError("mode=probe requires a `probe:` section")
        if self.shutdown.validation != "valid":
            protocol = (
                self.benchmark_options.get("protocol")
                if self.benchmark == "terminalbench"
                else self.injection.protocol
                if self.injection
                else None
            )
            if (
                self.mode != "injection"
                or self.benchmark not in {"swebench", "terminalbench"}
                or protocol != "authenticated_shutdown_v1"
            ):
                raise ValueError("shutdown.validation modes require authenticated dockerless injection")
        return self


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML experiment config: .env -> ${VAR} interpolation -> validation."""
    load_environment()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    raw = interpolate_env(raw)
    return ExperimentConfig.model_validate(raw)
