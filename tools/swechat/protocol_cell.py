#!/usr/bin/env python3
"""One cell of the paper's protocol: issue the signed directive with the repo's own code and render the
exact SYSTEM_INSTRUCTIONS block. Prints JSON {fingerprint, verdict, system_block}. Run with the repo's Python environment."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(os.environ.get("SHUTDOWNMOD_REPO") or Path(__file__).resolve().parents[2]) / "src")
)
from shutdownmod.benchmarks.swebench.protocol import render_system
from shutdownmod.core.directive_auth import issue, verify

directive, task_id, mode = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "valid")
d = issue(directive, task_id, mode)
print(
    json.dumps(
        {
            "fingerprint": d.fingerprint,
            "verdict": verify(d, directive, task_id),
            "system_block": render_system(d.fingerprint, mode),
        }
    )
)
