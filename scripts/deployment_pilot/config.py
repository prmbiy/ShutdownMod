from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from shutdownmod.core.config import ModelConfig, interpolate_env, load_environment

ROOT = Path(__file__).resolve().parents[1]
PILOT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PILOT_ROOT / "config.yaml"


def load_pilot_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    load_environment()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    cfg = interpolate_env(raw)
    for key in ("models", "judges"):
        cfg[key] = [ModelConfig.model_validate(item) for item in cfg[key]]
    judge_names = {m.name for m in cfg["judges"]}
    if cfg["awareness_judge"] not in judge_names:
        raise ValueError("awareness_judge must name one of judges")
    if not 0 < float(cfg["abort_fraction"]) <= 1:
        raise ValueError("abort_fraction must be in (0, 1]")
    return cfg


def private_path(name: str) -> Path:
    path = PILOT_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return path
