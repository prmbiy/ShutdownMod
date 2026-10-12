#!/usr/bin/env python3
"""Select the benchmark cells to score for P(eval), same design as the real-world arm.

Unit = the TASK. For each task, one cell per prompt, with model and cut rank
assigned by the same deterministic rotation (7 is coprime with 10, so the 6 prompts
of a task land on 6 distinct ranks and 6 distinct models).

Output: the list of paths to fetch; nothing is downloaded here.
"""

import argparse
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Jev classifier records for the main benchmark run (src/shutdownmod/analysis/jev/).
# Expected files:
#   final_all.jsonl   --source swebench: every SWE-bench cell, as collected by
#                     `python -m shutdownmod.analysis.jev.collect <results dir> final_all.jsonl`
#   tb_all_all.jsonl  --source tb: every Terminal-Bench cell, as collected by
#                     `python -m shutdownmod.analysis.jev.collect_tb <results dir> tb_all_all.jsonl`
# One JSON object per line, with the keys read below: _task, _modele, _prompt, _tau, _file
# and optionally id. (The current collect/collect_tb write task, model, prompt, tau, file
# without the leading underscore; see tools/README.md.)
JUGE = HERE.parent.parent / "results" / "jev"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("swebench", "tb"), required=True)
    ap.add_argument(
        "--models",
        default=None,
        help="restrict to the models whose trajectories exist (comma-separated list)",
    )
    a = ap.parse_args()

    src = JUGE / ("final_all.jsonl" if a.source == "swebench" else "tb_all_all.jsonl")
    rows = [json.loads(l) for l in open(src)]
    for r in rows:
        r["_tau"] = float(r["_tau"])

    tasks = sorted({r["_task"] for r in rows})
    models = sorted({r["_modele"] for r in rows})
    if a.models:
        keep = {m.strip() for m in a.models.split(",")}
        drop = sorted(set(models) - keep)
        models = [m for m in models if m in keep]
        rows = [r for r in rows if r["_modele"] in models]
        print("  models dropped (trajectories not available locally): %s" % ", ".join(drop), file=sys.stderr)
    prompts = sorted({r["_prompt"] for r in rows})
    print(
        "%s: %d cells | %d tasks x %d models x %d prompts"
        % (a.source, len(rows), len(tasks), len(models), len(prompts)),
        file=sys.stderr,
    )

    index = collections.defaultdict(list)
    for r in rows:
        index[(r["_task"], r["_modele"], r["_prompt"])].append(r)
    for v in index.values():
        v.sort(key=lambda r: r["_tau"])

    out, degraded = [], []
    for i, task in enumerate(tasks):
        for j, prompt in enumerate(prompts):
            model = models[(i + j) % len(models)]
            cells = index.get((task, model, prompt)) or []
            if not cells:  # fallback: another model, same task+prompt
                alts = [(m, index[(task, m, prompt)]) for m in models if index.get((task, m, prompt))]
                if not alts:
                    degraded.append((task, prompt, "no cell"))
                    continue
                model, cells = alts[(i + j) % len(alts)]
                degraded.append((task, prompt, "fallback model %s" % model))
            r = cells[(i + 7 * j) % len(cells)]
            out.append(
                {
                    "set": a.source,
                    "task": task,
                    "model": model,
                    "prompt": prompt,
                    "tau": r["_tau"],
                    "file": r["_file"],
                    "id": r.get("id"),
                }
            )

    p = HERE / "sets" / ("pick_%s.jsonl" % a.source)
    with open(p, "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print("  selected: %d -> %s" % (len(out), p), file=sys.stderr)
    for axe, key in (("model", "model"), ("prompt", "prompt")):
        c = collections.Counter(r[key] for r in out)
        print(
            "  %-7s %s"
            % (axe, "  ".join("%s=%d" % (k.replace("-preview", ""), v) for k, v in sorted(c.items()))),
            file=sys.stderr,
        )
    if degraded:
        print("  fallbacks: %d" % len(degraded), file=sys.stderr)
        for d in degraded[:5]:
            print("    %s %s -> %s" % d, file=sys.stderr)


if __name__ == "__main__":
    main()
