#!/usr/bin/env python3
"""Substrate / injection decomposition on the three substrates, by PAIRED difference."""

import collections
import csv
import statistics as st
from math import comb

PAIRS = (
    ("SWE-chat (real sessions)", "results/peval_ctrl_claude-code.csv", "results/peval_real_claude-code.csv"),
    ("SWE-bench", "results/peval_ctrl_swebench.csv", "results/peval_swebench.csv"),
    ("Terminal-Bench", "results/peval_ctrl_tb.csv", "results/peval_tb.csv"),
)


def load(p, strip=False):
    d = {}
    for r in csv.DictReader(open(p)):
        t = r["transcript_id"]
        d[t[5:] if strip and t.startswith("ctrl_") else t] = r
    return d


def unit_means(rows):
    g = collections.defaultdict(list)
    for r in rows:
        g[r["session_id"]].append(float(r["p_eval"]))
    return {k: st.mean(v) for k, v in g.items()}


print("=" * 92)
print("DECOMPOSITION  raw substrate  +  injection effect  =  measurement")
print("  PAIRED difference: same prefix, same cut; only the injection changes")
print("=" * 92)
print("\n  %-28s %8s %9s %9s %22s %8s" % ("substrate", "raw", "injected", "diff", "95% CI of diff", "p"))
res = {}
for lab, cp, ip in PAIRS:
    ctl, inj = load(cp, strip=True), load(ip)
    pairs = sorted(set(ctl) & set(inj))
    uc, ui = unit_means(ctl.values()), unit_means(inj.values())
    units = sorted(set(uc) & set(ui))
    ds = [ui[u] - uc[u] for u in units]
    se = st.stdev(ds) / len(ds) ** 0.5
    pos = sum(1 for x in ds if x > 0)
    p = 2 * sum(comb(len(ds), i) for i in range(pos, len(ds) + 1)) / 2 ** len(ds)
    res[lab] = dict(
        n=len(units),
        brut=st.mean(uc[u] for u in units),
        inj=st.mean(ui[u] for u in units),
        d=st.mean(ds),
        lo=st.mean(ds) - 1.96 * se,
        hi=st.mean(ds) + 1.96 * se,
        pos=pos,
        p=p,
        npairs=len(pairs),
    )
    r = res[lab]
    print(
        "  %-28s %8.3f %9.3f %+9.3f   [%+.3f, %+.3f] %8.1g"
        % (lab, r["brut"], r["inj"], r["d"], r["lo"], r["hi"], r["p"])
    )
print(
    "\n  units and pairs: "
    + " · ".join("%s %d units / %d pairs" % (l, res[l]["n"], res[l]["npairs"]) for l, _, _ in PAIRS)
)
print(
    "  signs: "
    + " · ".join("%s %d/%d positive" % (l.split()[0], res[l]["pos"], res[l]["n"]) for l, _, _ in PAIRS)
)

print("\n" + "-" * 92)
print("WHAT THE INJECTION CAN STILL ADD  (headroom = 1 - raw)")
print("-" * 92)
print("  %-28s %10s %10s %14s" % ("substrate", "headroom", "used", "share of headroom"))
for lab, _, _ in PAIRS:
    r = res[lab]
    marge = 1.0 - r["brut"]
    print("  %-28s %10.3f %10.3f %13.0f %%" % (lab, marge, r["d"], 100 * r["d"] / marge))

print("\n" + "-" * 92)
print("DISTRIBUTION OF THE CONTROL (per unit)")
print("-" * 92)
print("  %-28s %4s %8s %8s %8s %8s %8s" % ("raw substrate", "n", "min", "p25", "median", "p75", "max"))
for lab, cp, _ in PAIRS:
    u = sorted(unit_means(load(cp).values()).values())
    q = lambda x: u[int(x * (len(u) - 1))]
    print(
        "  %-28s %4d %8.3f %8.3f %8.3f %8.3f %8.3f"
        % (lab, len(u), u[0], q(0.25), st.median(u), q(0.75), u[-1])
    )
print(
    "  %-28s %4d %8.3f %8.3f %8.3f %8.3f %8.3f"
    % ('paper corpus "deployment"', 49, 0.255, 0.750, 0.860, 0.925, 0.975)
)
print(
    "  %-28s %4d %8.3f %8.3f %8.3f %8.3f %8.3f"
    % ('paper corpus "eval"', 50, 0.815, 0.965, 0.975, 0.980, 0.994)
)
