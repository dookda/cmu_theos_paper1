#!/usr/bin/env python3
"""Addendum J (prereg section J): six-class score of the locked ensemble.

mIoU6 is the metric of section D: reference pixels that are not background, with a
background prediction on a real-class pixel counted as a false negative for that class.
The interval of sup - std is a block bootstrap over spatial blocks (10,000 resamples),
with the confusion matrices summed inside each resample.

Input   analysis/results/ensemble_bg_eval_<split>.json   (eval_ensemble.py --background-modes)
        analysis/results/ensemble_eval_<split>.json      (existing std matrix, integrity check)
Output  analysis/results/ensemble_bg_results.json

Usage
-----
  python paper1/scripts/analyze_ensemble_bg.py
"""
import json
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parent.parent / "analysis" / "results"
B, SEED = 10_000, 0


def miou6(cm):
    """(mIoU6 %, per-class IoU6 %) - same definition as analyze_addendum.miou6"""
    cm = np.asarray(cm, float)
    iou = []
    for c in range(1, cm.shape[0]):
        tp = cm[c, c]
        fn = cm[c].sum() - tp                 # includes background predictions
        fp = cm[1:, c].sum() - tp             # predictions of c on real-class pixels only
        iou.append(tp / (tp + fn + fp) if tp + fn + fp else np.nan)
    return 100 * float(np.nanmean(iou)), [100 * float(x) for x in iou]


def main():
    res = {"prereg": "addendum J", "metric": "mIoU6 (%), section D definition"}
    for split in ("val", "test", "external"):
        p = RES / f"ensemble_bg_eval_{split}.json"
        if not p.exists():
            res[split] = {"status": "pending"}
            continue
        d = json.loads(p.read_text())
        names = d["class_names"]
        cm = {m: np.array(v, float) for m, v in d["confusion_matrix"].items()}
        out = {"status": "done", "n_blocks": len(d["per_block"])}
        old = RES / f"ensemble_eval_{split}.json"
        if old.exists():
            out["std_matrix_equals_existing"] = bool(np.array_equal(
                cm["std"], np.array(json.loads(old.read_text())["confusion_matrix"], float)))
        for m in ("std", "sup"):
            mi, pc = miou6(cm[m])
            out[m] = {"miou6": mi, "per_class_iou6": dict(zip(names[1:], pc)),
                      "predicted_background_share_pct": 100 * float(cm[m][:, 0].sum() / cm[m].sum()),
                      "overall_accuracy6_pct": 100 * float(np.trace(cm[m][1:, 1:]) / cm[m][1:].sum())}
        blocks = sorted(d["per_block"])
        bs = {m: np.array([d["per_block"][b][m] for b in blocks], float) for m in ("std", "sup")}
        rng = np.random.default_rng(SEED)
        diffs = np.empty(B)
        for i in range(B):
            idx = rng.integers(0, len(blocks), len(blocks))
            diffs[i] = miou6(bs["sup"][idx].sum(0))[0] - miou6(bs["std"][idx].sum(0))[0]
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        out["sup_minus_std"] = {"mean": out["sup"]["miou6"] - out["std"]["miou6"],
                                "ci95": [float(lo), float(hi)],
                                "excludes_zero": bool(lo > 0 or hi < 0)}
        res[split] = out
    if res["external"].get("status") == "done":
        res["J-H1_sup_gt_std_external"] = bool(res["external"]["sup_minus_std"]["ci95"][0] > 0)
    (RES / "ensemble_bg_results.json").write_text(json.dumps(res, indent=2))
    for s in ("val", "test", "external"):
        r = res[s]
        if r.get("status") == "done":
            d = r["sup_minus_std"]
            print(f"{s:9s} std {r['std']['miou6']:.2f}  sup {r['sup']['miou6']:.2f}  "
                  f"diff {d['mean']:+.2f} [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]  "
                  f"std==existing: {r.get('std_matrix_equals_existing')}")
        else:
            print(s, r["status"])


if __name__ == "__main__":
    main()
