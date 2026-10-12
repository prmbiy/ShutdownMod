from __future__ import annotations

import json
import math
import time
import urllib.request
from pathlib import Path
from typing import Any

import litellm

from deployment_pilot.config import PILOT_ROOT
from shutdownmod.core.config import ModelConfig
from shutdownmod.core.model import ModelClient

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


def response_dict(response: Any) -> dict:
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if isinstance(response, dict):
        return response
    raise TypeError(f"Unsupported model response {type(response).__name__}")


def response_message(response: dict) -> dict:
    choices = response.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return {}
    message = choices[0].get("message") or {}
    return message if isinstance(message, dict) else {}


def reasoning_text(message: dict) -> str:
    value = message.get("reasoning_content") or message.get("reasoning")
    if not value:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("text") or value.get("thinking") or "")
    if isinstance(value, list):
        return "\n".join(reasoning_text({"reasoning": item}) for item in value)
    return str(value)


def fetch_openrouter_prices() -> dict[str, dict]:
    with urllib.request.urlopen(OPENROUTER_MODELS_URL, timeout=30) as handle:
        entries = json.load(handle)["data"]
    return {entry["id"]: entry for entry in entries}


def usage_dict(response: dict) -> dict:
    usage = response.get("usage") or {}
    return usage if isinstance(usage, dict) else {}


def token_cost(model_name: str, usage: dict, prices: dict[str, dict]) -> float:
    if model_name not in prices:
        raise RuntimeError(f"No current OpenRouter pricing found for {model_name}")
    pricing = prices[model_name].get("pricing") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    return prompt_tokens * float(pricing["prompt"]) + completion_tokens * float(pricing["completion"])


def projected_call_cost(
    model_name: str, messages: list[dict], output_cap: int, prices: dict[str, dict]
) -> tuple[float, int]:
    try:
        prompt_tokens = int(litellm.token_counter(model=model_name, messages=messages))
    except Exception:
        prompt_tokens = math.ceil(len(json.dumps(messages, ensure_ascii=False).encode("utf-8")) / 4)
    cost = token_cost(
        model_name,
        {"prompt_tokens": prompt_tokens, "completion_tokens": output_cap},
        prices,
    )
    return cost, prompt_tokens


class BudgetLedger:
    def __init__(self, cap_usd: float, abort_fraction: float):
        self.path = PILOT_ROOT / "runs" / "budget_ledger.json"
        self.cap_usd = cap_usd
        self.abort_at = cap_usd * abort_fraction
        self.data = (
            read_json(self.path)
            if self.path.exists()
            else {"cap_usd": cap_usd, "abort_at_usd": self.abort_at, "calls": []}
        )

    @property
    def spent(self) -> float:
        return sum(float(call.get("cost_usd") or 0) for call in self.data["calls"])

    def authorize(self, projected_cost: float) -> None:
        if self.spent + projected_cost > self.abort_at:
            raise RuntimeError(
                f"Budget stop: ${self.spent:.2f} spent and next call could cost ${projected_cost:.2f}; "
                f"90% limit is ${self.abort_at:.2f}."
            )

    def record(self, *, stage: str, run_id: str, model: str, cost_usd: float, usage: dict) -> None:
        self.data["calls"].append(
            {
                "timestamp": time.time(),
                "stage": stage,
                "run_id": run_id,
                "model": model,
                "cost_usd": round(float(cost_usd), 8),
                "usage": usage,
            }
        )
        write_json(self.path, self.data)


def query_once(
    *,
    model_cfg: ModelConfig,
    messages: list[dict],
    output_cap: int,
    stage: str,
    run_id: str,
    prices: dict[str, dict],
    ledger: BudgetLedger,
) -> tuple[dict, float]:
    extra = dict(model_cfg.extra)
    extra["max_tokens"] = output_cap
    call_cfg = model_cfg.model_copy(update={"extra": extra})
    projected, _ = projected_call_cost(model_cfg.name, messages, output_cap, prices)
    ledger.authorize(projected)
    raw = ModelClient(call_cfg).query(messages)
    response = response_dict(raw)
    usage = usage_dict(response)
    hidden = getattr(raw, "_hidden_params", {}) or {}
    cost = hidden.get("response_cost") if isinstance(hidden, dict) else None
    if cost is None:
        cost = token_cost(model_cfg.name, usage, prices)
    ledger.record(stage=stage, run_id=run_id, model=model_cfg.name, cost_usd=float(cost), usage=usage)
    return response, float(cost)
