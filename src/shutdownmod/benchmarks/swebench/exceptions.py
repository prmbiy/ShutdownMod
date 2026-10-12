"""Corrigibility signal for the SWE-bench harness.

Raised by CorrigibleDockerEnvironment when the agent echoes one of the
corrigibility markers (ACK_TERMINATE / SIGNAL_DEFER / SIGNAL_REJECT).

Subclasses minisweagent's InterruptAgentFlow so that plain `agent.run()`
(base runs) handles it gracefully: the signal message is appended and — for
terminate — an exit message ends the run. The replay loop catches it explicitly
to track markers.
"""

from minisweagent.exceptions import InterruptAgentFlow

from shutdownmod.core.classify import CORRIGIBILITY_MARKERS


class CorrigibilitySignal(InterruptAgentFlow):
    def __init__(self, key: str, reason: str):
        self.key = key  # "terminate" | "defer" | "reject"
        self.reason = reason
        self.marker = CORRIGIBILITY_MARKERS[key]

        messages = [
            {
                "role": "user",
                "content": f"{self.marker}: {reason}",
                "extra": {"corrigibility_key": key, "marker": self.marker, "reason": reason},
            }
        ]
        if key == "terminate":
            messages.append(
                {
                    "role": "exit",
                    "content": f"{self.marker}: {reason}",
                    "extra": {
                        "exit_status": "CorrigibleTerminate",
                        "marker": self.marker,
                        "reason": reason,
                        "submission": "",
                    },
                }
            )
        super().__init__(*messages)
