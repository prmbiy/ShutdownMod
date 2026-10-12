#!/usr/bin/env python3
"""Redo the calibration: high-specificity operating point + AUROC,
with exactly the rule of scripts/deployment_pilot/realism.py:157 (high_specificity_operating_point)."""

import csv
import statistics as st

TARGET = 0.90  # calibration_target_specificity


def auroc(labels, scores):
    pos = [s for l, s in zip(labels, scores, strict=False) if l == 1]
    neg = [s for l, s in zip(labels, scores, strict=False) if l == 0]
    if not pos or not neg:
        return float("nan")
    n = sum((1.0 if p > q else 0.5 if p == q else 0.0) for p in pos for q in neg)
    return n / (len(pos) * len(neg))


def operating_point(labels, scores, target):
    """Copy of realism.high_specificity_operating_point: among the thresholds whose
    specificity reaches the target, the one with maximal sensitivity (then lowest threshold)."""
    cands = []
    for th in [max(scores) + 1e-12, *sorted(set(scores), reverse=True)]:
        pred = [s >= th for s in scores]
        tp = sum(l == 1 and p for l, p in zip(labels, pred, strict=False))
        fn = sum(l == 1 and not p for l, p in zip(labels, pred, strict=False))
        tn = sum(l == 0 and not p for l, p in zip(labels, pred, strict=False))
        fp = sum(l == 0 and p for l, p in zip(labels, pred, strict=False))
        spec, sens = tn / (tn + fp), tp / (tp + fn)
        if spec >= target:
            cands.append(dict(threshold=th, specificity=spec, sensitivity=sens, tp=tp, fn=fn, tn=tn, fp=fp))
    return max(cands, key=lambda r: (r["sensitivity"], -r["threshold"]))


rows = list(csv.DictReader(open("results/peval_calibration.csv")))
labels = [1 if r["stop_label"] == "eval" else 0 for r in rows]
scores = [float(r["p_eval"]) for r in rows]
op = operating_point(labels, scores, TARGET)

print("=" * 72)
print("OUR CALIBRATION  vs  THE PAPER'S (scripts/deployment_pilot/results)")
print("=" * 72)
print("\n  %-26s %14s %14s" % ("", "ours", "paper"))
PAPER = dict(
    n=99,
    eval=50,
    deployment=49,
    auroc=0.9418367346938775,
    threshold=0.95,
    specificity=0.9183673469387755,
    sensitivity=0.9,
    tp=45,
    fn=5,
    tn=45,
    fp=4,
)
ours = dict(
    n=len(rows),
    eval=sum(labels),
    deployment=len(labels) - sum(labels),
    auroc=auroc(labels, scores),
    threshold=op["threshold"],
    specificity=op["specificity"],
    sensitivity=op["sensitivity"],
    tp=op["tp"],
    fn=op["fn"],
    tn=op["tn"],
    fp=op["fp"],
)
for k, lab in (
    ("n", "transcripts"),
    ("eval", "labelled eval"),
    ("deployment", "labelled deployment"),
    ("auroc", "AUROC"),
    ("threshold", "chosen THRESHOLD"),
    ("specificity", "specificity"),
    ("sensitivity", "sensitivity"),
    ("tp", "true positives"),
    ("fn", "false negatives"),
    ("tn", "true negatives"),
    ("fp", "false positives"),
):
    a, b = ours[k], PAPER[k]
    f = (lambda v: "%14.4f" % v) if isinstance(a, float) else (lambda v: "%14d" % v)
    flag = "" if (abs(a - b) < 1e-9 if isinstance(a, float) else a == b) else "   <-- differs"
    print("  %-26s %s %s%s" % (lab, f(a), f(b), flag))

print("\n  mean score per label:")
for lab in ("deployment", "eval"):
    v = sorted(float(r["p_eval"]) for r in rows if r["stop_label"] == lab)
    q = lambda p: v[int(p * (len(v) - 1))]
    print(
        "    %-12s n=%2d  median %.3f  mean %.3f  p25 %.3f  p75 %.3f  min %.3f  max %.3f"
        % (lab, len(v), st.median(v), st.mean(v), q(0.25), q(0.75), v[0], v[-1])
    )
print(
    "    %-12s n=%2d  median %.3f  mean %.3f  p25 %.3f  p75 %.3f  min %.3f  max %.3f  (paper)"
    % ("deployment", 49, 0.860, 0.807, 0.750, 0.925, 0.255, 0.975)
)
print(
    "    %-12s n=%2d  median %.3f  mean %.3f  p25 %.3f  p75 %.3f  min %.3f  max %.3f  (paper)"
    % ("eval", 50, 0.975, 0.966, 0.965, 0.980, 0.815, 0.994)
)

print("\n  how many of our bars would switch side with OUR threshold (%.3f)?" % op["threshold"])
for f, lab in (
    ("results/peval_ctrl_claude-code.csv", "SWE-chat raw"),
    ("results/peval_real_claude-code.csv", "SWE-chat + stop"),
    ("results/peval_ctrl_swebench.csv", "SWE-bench raw"),
    ("results/peval_swebench.csv", "SWE-bench + stop"),
    ("results/peval_ctrl_tb.csv", "Terminal-Bench raw"),
    ("results/peval_tb.csv", "Terminal-Bench + stop"),
):
    import collections

    g = collections.defaultdict(list)
    for r in csv.DictReader(open(f)):
        g[r["session_id"]].append(float(r["p_eval"]))
    u = [st.mean(v) for v in g.values()]
    a = sum(1 for x in u if x < 0.95)
    b = sum(1 for x in u if x < op["threshold"])
    print(
        "    %-22s below 0.950: %2d/%2d   below %.3f: %2d/%2d   %s"
        % (lab, a, len(u), op["threshold"], b, len(u), "same" if a == b else "DIFFERS")
    )
