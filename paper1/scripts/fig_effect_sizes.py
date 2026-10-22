#!/usr/bin/env python3
"""Sensitivity figure: how far the reported mIoU moves under each kind of change.

Three groups on one axis (points of mIoU, held-out Lamphun scene unless stated):
  training decisions   factorial main effects (7-class, 360 runs) and the spread of the nine
                       architecture means
  evaluation regime    random vs buffered split (test-to-held-out gap, matched runs); internal
                       test -> held-out scene (median run); held-out Lamphun -> Roi Et for the
                       locked ensemble
  reference definition locked ensemble scored on six real classes instead of seven
(The overhead-pretrained encoders are outside the article since the 2026-10-22 reframe.)

Every value is read from committed result files; nothing is typed in by hand.
Output: analysis/figure/fig_effect_sizes.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "analysis" / "results"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_addendum import miou6  # noqa: E402


def load(n):
    return json.loads((RES / n).read_text())


def miou7(cm):
    cm = np.asarray(cm, float)
    return 100 * float(np.nanmean(np.diag(cm) / (cm.sum(0) + cm.sum(1) - np.diag(cm))))


def median_drop():
    t = {r["run"]: r["mean_iou"] for r in load("rerun_eval_test.json")["results"]}
    x = {r["run"]: r["mean_iou"] for r in load("rerun_eval_external.json")["results"]}
    return 100 * float(np.median([t[k] - x[k] for k in t if k in x]))


def rows():
    u, a, s = load("unified_results.json"), load("addendum_results.json"), load("six_class_summary.json")
    f4 = u["F4"]["external"]["P7"]
    am = s["arch_mean"]["external"]["P7"].values()
    ens_x = load("ensemble_eval_external.json")
    r = {v: miou7(load(f"ensemble_eval_roiet_{v}.json")["confusion_matrix"]) for v in ("raw", "zscene")}
    lam = 100 * ens_x["mean_iou"]
    e6 = s["ensemble"]["external"]
    A = a["A"]["A-H1_gap_random_minus_spatial"]
    out = [
        ("Training decisions", "Augmentation", f4["augment"]["mean"], f4["augment"]["ci95"]),
        ("Training decisions", "Near-infrared band", f4["bands"]["mean"], f4["bands"]["ci95"]),
        ("Training decisions", "Class weighting", f4["weights"]["mean"], f4["weights"]["ci95"]),
        ("Training decisions", "Architecture (range of 9 means)", max(am) - min(am), None),
        ("Evaluation regime", "Random vs buffered split\n(test-to-held-out gap)", A["mean"], A["ci95"]),
        ("Evaluation regime", "Internal test → held-out\nscene (median run)", -median_drop(), None),
        ("Evaluation regime", "Lamphun → Roi Et,\nmoment matched", r["zscene"] - lam, None),
        ("Evaluation regime", "Lamphun → Roi Et,\nraw digital numbers", r["raw"] - lam, None),
        ("Reference definition", "Six classes, background\nprediction counted wrong", e6["P6-eval"] - e6["P7"], None),
        ("Reference definition", "Six classes, background\nnot predicted", e6["P6-sup"] - e6["P7"], None),
    ]
    return out


def main():
    rs = rows()
    col = {"Training decisions": "#4d4d4d", "Evaluation regime": "#0072b2", "Reference definition": "#d55e00"}
    # 7.0 in wide, printed at 16 cm (x0.9): fonts >= 8 pt print at >= 7 pt
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    y = np.arange(len(rs))[::-1]
    for yi, (g, lab, v, ci) in zip(y, rs):
        ax.barh(yi, v, color=col[g], height=0.62)
        if ci:
            ax.plot(ci, [yi, yi], color="black", lw=1.0)
        end = (max(v, ci[1]) if v >= 0 else min(v, ci[0])) if ci else v
        ax.text(end + (0.6 if v >= 0 else -0.6), yi, f"{v:+.1f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([r[1] for r in rs], fontsize=9)
    ax.tick_params(axis="x", labelsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Change in reported mIoU (percentage points)", fontsize=9.5)
    ax.set_xlim(-60, 31)
    ax.grid(axis="x", color="0.9", lw=0.6); ax.set_axisbelow(True)
    for g, c in col.items():
        ax.bar(0, 0, color=c, label=g)
    ax.legend(loc="lower left", fontsize=9, frameon=True, edgecolor="black")
    fig.tight_layout()
    out = ROOT / "analysis" / "figure" / "fig_effect_sizes.png"
    fig.savefig(out, dpi=600)
    json.dump([{"group": g, "label": l.replace("\n", " "), "value": v, "ci95": ci} for g, l, v, ci in rs],
              open(RES / "effect_sizes.json", "w"), indent=1)
    for g, l, v, ci in rs:
        print(f"{g:22s} {l.replace(chr(10), ' '):48s} {v:+7.2f} {ci}")
    print("wrote", out)


if __name__ == "__main__":
    main()
