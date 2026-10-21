#!/usr/bin/env python3
"""Addendum I (prereg section I): 2x2 factorial of augmentation x class weighting for DINOv3-SAT.

Estimator as in analyze_rerun.py: within a seed, a main effect is the mean of the two
cells at the high level minus the mean of the two at the low level, the interaction uses
the product of the +-1 codes; the five per-seed estimates are summarised by their mean
and a 95% t interval.

Input   analysis/results/dinov3rgb_eval_<split>.json    b0, b2, b3 (addendum I)
        analysis/results/foundation_eval_<split>.json   b1 (addendum E, reused)
Output  analysis/results/dinov3_factorial.json

Usage
-----
  python paper1/scripts/analyze_dinov3_factorial.py
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

RES = Path(__file__).resolve().parent.parent / "analysis" / "results"
ARCH = "dinov3_vitl_sat"
# condition -> (augment, weights) as +-1
CODES = {"b0_rgb_plain": (-1, -1), "b1_rgb_aug": (1, -1),
         "b2_rgb_wt": (-1, 1), "b3_rgb_aug_wt": (1, 1)}
SPLITS = ["val", "test", "external", "roiet_zscene"]


def cells(split):
    out = {c: {} for c in CODES}
    for name in (f"dinov3rgb_eval_{split}.json", f"foundation_eval_{split}.json"):
        p = RES / name
        if not p.exists():
            return None
        for r in json.loads(p.read_text())["results"]:
            arch, cond, seed = r["run"].split("__")
            if arch == ARCH and cond in CODES:
                out[cond][int(seed[1:])] = 100 * r["mean_iou"]
    return out


def summary(x):
    x = np.asarray(x, float)
    half = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    lo, hi = float(x.mean() - half), float(x.mean() + half)
    return {"mean": float(x.mean()), "ci95": [lo, hi], "n": len(x),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def main():
    res = {"prereg": "addendum I", "architecture": ARCH, "unit": "mIoU percentage points"}
    for split in SPLITS:
        c = cells(split)
        if c is None or any(len(v) == 0 for v in c.values()):
            res[split] = {"status": "pending"}
            continue
        seeds = sorted(set.intersection(*(set(v) for v in c.values())))
        eff = {}
        for label, sign in (("augment", lambda k: k[0]), ("weights", lambda k: k[1]),
                            ("augment x weights", lambda k: k[0] * k[1])):
            per_seed = [np.mean([c[k][s] for k, code in CODES.items() if sign(code) > 0])
                        - np.mean([c[k][s] for k, code in CODES.items() if sign(code) < 0])
                        for s in seeds]
            eff[label] = summary(per_seed)
        res[split] = {"status": "done", "seeds": seeds,
                      "cell_mean": {k: float(np.mean([c[k][s] for s in seeds])) for k in CODES},
                      "effects": eff}
    if res["val"].get("status") == "done":
        e = res["val"]["effects"]
        res["I-H1_augment_positive_on_val"] = bool(e["augment"]["ci95"][0] > 0)
        res["I-H2_weights_interval_includes_zero"] = not e["weights"]["excludes_zero"]
    (RES / "dinov3_factorial.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: (v.get("status") if isinstance(v, dict) else v)
                      for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
