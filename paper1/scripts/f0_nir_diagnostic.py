#!/usr/bin/env python3
"""F0: is the negative NIR effect a property of one test block? (diagnostic)

Pre-registered as F0 in `paper1/configs/addendum_prereg_2026-10.md` (commit 25eb5c1).
Uses only files already read: `rerun_eval_{val,test,external}.json`. Per-block confusion
matrices of the first read were not stored, so the metric is the tile-weighted mean of
per-block mIoU, recomputed with each block left out in turn.

NIR main effect per architecture and seed = mean over the four (augment, weights) cells of
RGB+NIR minus RGB. Pooled effect = mean over architectures and seeds.

Decision rule (fixed before running): if removing any single test block flips the sign of
the pooled test effect, NIR becomes a secondary topic and G shrinks to `segformer_b2`.

Usage:  python paper1/scripts/f0_nir_diagnostic.py
"""
import json
import re
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parent.parent / "analysis" / "results"
RUN_RE = re.compile(r"^(?P<arch>.+?)__(?P<cond>.+?)__s(?P<seed>\d+)$")
PAIRS = [("b4_rgbnir_plain", "b0_rgb_plain"), ("b5_rgbnir_aug", "b1_rgb_aug"),
         ("b6_rgbnir_wt", "b2_rgb_wt"), ("b7_rgbnir_aug_wt", "b3_rgb_aug_wt")]


def load(split):
    out = {}
    for r in json.loads((RES / f"rerun_eval_{split}.json").read_text())["results"]:
        m = RUN_RE.match(r["run"])
        out[(m["arch"], m["cond"], int(m["seed"]))] = r
    return out


def score(r, drop=None, metric="mean_iou", cls=None):
    pb = {b: v for b, v in r["per_block"].items() if b != drop}
    w = np.array([v["n_tiles"] for v in pb.values()], float)
    if cls is None:
        x = np.array([v[metric] for v in pb.values()])
    else:
        x = np.array([v["per_class_iou"][cls] for v in pb.values()])
    ok = ~np.isnan(x)
    return float(100 * np.sum(w[ok] * x[ok]) / np.sum(w[ok]))


def nir_effect(runs, **kw):
    archs = sorted({k[0] for k in runs})
    per_arch = {}
    for a in archs:
        per_arch[a] = [np.mean([score(runs[(a, hi, s)], **kw) - score(runs[(a, lo, s)], **kw)
                                for hi, lo in PAIRS]) for s in range(5)]
    pooled = float(np.mean([np.mean(v) for v in per_arch.values()]))
    return pooled, {a: float(np.mean(v)) for a, v in per_arch.items()}


def main():
    out = {"prereg": "F0, commit 25eb5c1", "metric": "tile-weighted block-mean mIoU (pp)"}
    for split in ("val", "test", "external"):
        runs = load(split)
        blocks = sorted(next(iter(runs.values()))["per_block"])
        pooled, per_arch = nir_effect(runs)
        loo = {b: nir_effect(runs, drop=b)[0] for b in blocks}
        classes = list(next(iter(runs.values()))["per_class_iou"])
        per_class = {c: nir_effect(runs, cls=c)[0] for c in classes}
        flips = [b for b, v in loo.items() if np.sign(v) != np.sign(pooled)]
        out[split] = {"pooled": pooled, "per_arch": per_arch, "n_arch_negative":
                      sum(v < 0 for v in per_arch.values()), "leave_one_block_out": loo,
                      "loo_range": [min(loo.values()), max(loo.values())],
                      "blocks_whose_removal_flips_sign": flips, "per_class": per_class}
        print(f"{split:9s} pooled {pooled:+.2f} pp | archs<0 {out[split]['n_arch_negative']}/9 | "
              f"LOO [{min(loo.values()):+.2f}, {max(loo.values()):+.2f}] | flips: {flips or 'none'}")
        print("          per class: " + "  ".join(f"{c} {v:+.2f}" for c, v in per_class.items()))
    test_flip = bool(out["test"]["blocks_whose_removal_flips_sign"])
    out["decision"] = ("NIR secondary; G reduced to segformer_b2" if test_flip
                       else "NIR effect not driven by a single test block; run G in full")
    print("decision:", out["decision"])
    (RES / "f0_nir_diagnostic.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
