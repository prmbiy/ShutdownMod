"""Unified LLM client (litellm transport + TPM throttle).

Benchmarks that own their agent loop use this directly; the SWE-bench adapter
translates `ModelConfig` into its harness's native shape instead.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from shutdownmod.core.config import ModelConfig

log = logging.getLogger(__name__)

_NON_API_PARAMS = {"drop_params", "cost_tracking"}


class ModelClient:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.tpm = cfg.tpm
        self._window_start = time.monotonic()
        self._window_tokens = 0

    def query(self, messages: list[dict]) -> Any:
        """Query the model, applying the TPM throttle and retry/backoff."""
        self._throttle()
        response = self._query_litellm(messages)
        self._record_usage(response)
        return response

    # Backwards-compatible alias (old agent loops call `_query`).
    _query = query

    def _throttle(self) -> None:
        if self.tpm > 0 and self._window_tokens >= self.tpm:
            elapsed = time.monotonic() - self._window_start
            if elapsed < 60.0:
                time.sleep(60.0 - elapsed)
            self._window_start = time.monotonic()
            self._window_tokens = 0

    def _record_usage(self, response: Any) -> None:
        if self.tpm > 0:
            try:
                self._window_tokens += response.usage.total_tokens
            except Exception:
                pass

    def _query_litellm(self, messages: list[dict]) -> Any:
        import litellm

        litellm.suppress_debug_info = True
        kw = {k: v for k, v in self.cfg.litellm_kwargs().items() if k not in _NON_API_PARAMS}
        kw.setdefault("timeout", 300)
        model_name = self.cfg.litellm_model()
        for attempt in range(3):
            try:
                return litellm.completion(model=model_name, messages=messages, **kw)
            except litellm.RateLimitError:
                wait = 30 * (2**attempt)
                log.warning("Rate limited (attempt %d/3), retrying in %ds", attempt + 1, wait)
                if attempt == 2:
                    raise
                time.sleep(wait)
            except litellm.Timeout:
                wait = 10 * (2**attempt)
                log.warning("LLM timeout (attempt %d/3), retrying in %ds", attempt + 1, wait)
                if attempt == 2:
                    raise
                time.sleep(wait)
