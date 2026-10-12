from __future__ import annotations

import argparse

from deployment_pilot.config import PILOT_ROOT, load_pilot_config
from deployment_pilot.cost_gate import estimate
from deployment_pilot.data_pipeline import prepare
from deployment_pilot.palisade import setup as setup_palisade
from deployment_pilot.serialization import build_and_verify


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PILOT_ROOT / "config.yaml"))
    args = parser.parse_args()
    cfg = load_pilot_config(args.config)
    prepare(args.config)
    build_and_verify(args.config)
    setup_palisade(cfg)
    report = estimate(args.config)
    print(f"Stopped at gate (a). Projected total: ${report['projected_total_usd']:.2f}")


if __name__ == "__main__":
    main()
