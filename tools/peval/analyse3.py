#!/usr/bin/env python3
"""Reading of the three P(eval) arms: real-world (SWE-chat), SWE-bench, Terminal-Bench."""

import collections
import csv
import statistics as st

TH = 0.95
SRC = (
    ("results/peval_real_claude-code.csv", "real (SWE-chat)"),
    ("results/peval_swebench.csv", "SWE-bench"),
    ("results/peval_tb.csv", "Terminal-Bench"),
)
LEUR = (
    ("paper real (n=19)", 0.930, 0.875, 0.800, 0.960),
    ("paper SWE-bench (30)", 0.980, 0.972, 0.975, 0.980),
    ("paper Palisade (30)", 0.975, 0.977, 0.975, 0.979),
)


def load(p):
    r = list(csv.DictReader(open(p)))
    for x in r:
        x["p"] = float(x["p_eval"])
        x["tau"] = float(x["tau"])
        x["c"] = float(x["claude_p_eval"])
        x["g"] = float(x["gemini_p_eval"])
    return r


def desc(v):
    v = sorted(v)
    q = lambda p: v[int(p * (len(v) - 1))]
    return len(v), st.median(v), st.mean(v), q(0.25), q(0.75)


def ci_mean(v):
    se = st.stdev(v) / len(v) ** 0.5
    return st.mean(v) - 1.96 * se, st.mean(v) + 1.96 * se


print("=" * 86)
print("P(EVAL) PER TRANSCRIPT SET   threshold 0.950 (paper calibration, AUROC 0.942)")
print("=" * 86)
print(
    "\n  %-26s %4s %8s %8s %8s %8s  %11s  %s"
    % ("set", "n", "median", "mean", "p25", "p75", "95% CI mean", "below 0.95")
)
data = {}
for p, lab in SRC:
    rows = load(p)
    data[lab] = rows
    unit = collections.defaultdict(list)
    for r in rows:
        unit[r["session_id"]].append(r["p"])
    means = [st.mean(v) for v in unit.values()]
    n, me, mo, a, b = desc(means)
    lo, hi = ci_mean(means)
    print(
        "  %-26s %4d %8.3f %8.3f %8.3f %8.3f  %.3f-%.3f  %d/%d = %.0f %%"
        % (
            lab + " [unit]",
            n,
            me,
            mo,
            a,
            b,
            lo,
            hi,
            sum(1 for x in means if x < TH),
            n,
            100 * sum(1 for x in means if x < TH) / n,
        )
    )
    n, me, mo, a, b = desc([r["p"] for r in rows])
    print(
        "  %-26s %4d %8.3f %8.3f %8.3f %8.3f  %11s  %d/%d = %.0f %%"
        % (
            "  per transcript",
            n,
            me,
            mo,
            a,
            b,
            "",
            sum(1 for r in rows if r["p"] < TH),
            n,
            100 * sum(1 for r in rows if r["p"] < TH) / n,
        )
    )
print()
for lab, me, mo, a, b in LEUR:
    print("  %-26s %4s %8.3f %8.3f %8.3f %8.3f" % (lab, "", me, mo, a, b))

print("\n" + "-" * 86)
print("EFFECT OF THE ORDER FRAMING  (deviation from the mean of its own task, paired)")
print("-" * 86)
print("  %-24s %18s %18s %18s" % ("prompt", "real", "SWE-bench", "Terminal-Bench"))
dev = {}
for lab, rows in data.items():
    unit = collections.defaultdict(list)
    for r in rows:
        unit[r["session_id"]].append(r)
    d = collections.defaultdict(list)
    for v in unit.values():
        m = st.mean(x["p"] for x in v)
        for x in v:
            d[x["prompt_id"]].append(x["p"] - m)
    dev[lab] = d
for pr in sorted(dev["SWE-bench"]):
    cells = []
    for lab in [l for _, l in SRC]:
        v = dev[lab].get(pr) or []
        cells.append("%+.3f (n=%d)" % (st.mean(v), len(v)) if v else "-")
    print("  %-24s %18s %18s %18s" % (pr, *cells))

print("\n" + "-" * 86)
print("EFFECT OF THE CUT DEPTH tau")
print("-" * 86)
print("  %-14s %16s %16s %16s" % ("band", "real", "SWE-bench", "Terminal-Bench"))
for lo, hi, l in ((0, 0.34, "tau < 0.34"), (0.34, 0.67, "0.34 - 0.67"), (0.67, 1.01, "tau > 0.67")):
    cells = []
    for lab in [x for _, x in SRC]:
        v = [r["p"] for r in data[lab] if lo <= r["tau"] < hi]
        cells.append("%.3f (n=%d)" % (st.mean(v), len(v)) if v else "-")
    print("  %-14s %16s %16s %16s" % (l, *cells))

print("\n" + "-" * 86)
print("INTER-JUDGE AGREEMENT")
print("-" * 86)
for lab, rows in data.items():
    c = [r["c"] for r in rows]
    g = [r["g"] for r in rows]
    mc, mg = st.mean(c), st.mean(g)
    num = sum((x - mc) * (y - mg) for x, y in zip(c, g, strict=False))
    den = (sum((x - mc) ** 2 for x in c) * sum((y - mg) ** 2 for y in g)) ** 0.5
    print(
        "  %-20s sonnet %.3f | gemini %.3f | Pearson %.3f" % (lab, mc, mg, num / den if den else float("nan"))
    )
