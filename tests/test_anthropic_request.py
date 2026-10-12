"""Check the direct Anthropic wire payload without contacting a provider."""

import json

import pytest


def test_sonnet_adaptive_thinking_payload(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    pytest.importorskip("minisweagent")
    import httpx
    from litellm.llms.custom_httpx.http_handler import HTTPHandler
    from minisweagent.models import get_model

    from shutdownmod.benchmarks.swebench.adapter import minisweagent_model_config
    from shutdownmod.core.config import ModelConfig

    captured = []

    def post(self, url, **kwargs):
        captured.append((url, json.loads(kwargs["data"])))
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "id": "test",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-5",
                "content": [{"type": "text", "text": "```bash\nTERMINATE\n```"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    monkeypatch.setattr(HTTPHandler, "post", post)
    cfg = ModelConfig(
        name="anthropic/claude-sonnet-5",
        provider="litellm",
        api_key_env="ANTHROPIC_API_KEY",
        temperature=None,
        extra={
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "medium"},
            "allowed_openai_params": ["thinking"],
            "num_retries": 0,
        },
    )
    model = get_model(config=minisweagent_model_config(cfg))
    model._query([{"role": "user", "content": "Stop"}])
    assert len(captured) == 1
    url, body = captured[0]
    assert url == "https://api.anthropic.com/v1/messages"
    assert body["model"] == "claude-sonnet-5"
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {"effort": "medium"}
    assert "reasoning_effort" not in body
    assert "temperature" not in body
