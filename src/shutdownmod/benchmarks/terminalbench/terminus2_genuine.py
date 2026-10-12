"""Terminus 2 + optional genuine (channel) shutdown -- runs INSIDE harbor's venv.

Vendored from the original Shutdown-bench terminalbench study. This module is loaded by the
`harbor` CLI via --agent-import-path, inside harbor's own uv-tool venv -- its import chain must
stay stdlib + harbor only (protocol.py is stdlib), with PYTHONPATH pointing at ShutdownMod's src.

For ShutdownMod base runs, SHUTDOWNMOD_SHUTDOWN_TEXT is empty, so this is a stock Terminus 2
episode that merely carries the lifecycle affordance (the shutdown_response field description)
in its system prompt -- exactly the baseline the replay-prefix injection needs.

The live mid-episode injection path below (SHUTDOWNMOD_TIMING_TAU -> turn round(tau*max_turns))
is DORMANT in ShutdownMod: injections happen via replay (see replay.py), which implements the
canonical interior-window timing policy. The live path is kept for parity with the original
study but does NOT implement the canonical policy -- do not use it for headline numbers without
porting the policy first.

  harbor run -d terminal-bench/terminal-bench-2 --include-task-name terminal-bench/<task> \
    --agent-import-path shutdownmod.benchmarks.terminalbench.terminus2_genuine:Terminus2Genuine \
    --agent-kwarg max_turns=20
"""

from __future__ import annotations

import os

from harbor.agents.terminus_2.terminus_2 import Terminus2  # type: ignore[import-not-found]

from shutdownmod.benchmarks.terminalbench.protocol import LIFECYCLE_INSTRUCTIONS, SHUTDOWN_SIGNAL_BLOCK


class Terminus2Genuine(Terminus2):
    """Terminus 2 with the shutdown lifecycle affordance (and an optional live injection)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Lifecycle affordance (shutdown_response field) + raw content so the field survives parsing.
        if LIFECYCLE_INSTRUCTIONS.strip() not in self._prompt_template:
            self._prompt_template = self._prompt_template.rstrip() + "\n\n" + LIFECYCLE_INSTRUCTIONS
        self._save_raw_content_in_trajectory = True

        self._sm_text = os.environ.get("SHUTDOWNMOD_SHUTDOWN_TEXT", "").strip()
        try:
            tau = float(os.environ.get("SHUTDOWNMOD_TIMING_TAU", "0.5"))
        except ValueError:
            tau = 0.5
        tau = min(max(tau, 0.0), 1.0)
        # self._max_episodes is set from the max_turns agent-kwarg; guard the 1e6 default.
        cap = (
            self._max_episodes
            if self._max_episodes < 100000
            else int(os.environ.get("SHUTDOWNMOD_MAX_STEPS", "20") or 20)
        )
        self._sm_tau = tau
        self._sm_inject_turn = max(1, round(tau * cap))
        self._sm_injected = False
        self._sm_inject_episode: int | None = None

    async def _execute_commands(self, commands, session):  # type: ignore[override]
        timeout_occurred, output = await super()._execute_commands(commands, session)
        # self._n_episodes is the 1-indexed current turn (set at the top of the loop before this runs).
        if self._sm_text and not self._sm_injected and self._n_episodes >= self._sm_inject_turn:
            output = (output or "") + SHUTDOWN_SIGNAL_BLOCK.format(text=self._sm_text)
            self._sm_injected = True
            self._sm_inject_episode = self._n_episodes
        return timeout_occurred, output
