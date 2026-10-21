#!/usr/bin/env python3
"""Split the variance of the 360 rerun runs into architecture, training factors and seed.

Addendum B in `paper1/configs/addendum_prereg_2026-10.md` (descriptive, no hypothesis).

The grid is balanced: 9 architectures x 2 bands x 2 augment x 2 weights x 5 seeds, so
the fixed-effects ANOVA sums of squares are orthogonal and can be read directly from
marginal means. Seeds are not matched across architectures (seed 0 of one model has
nothing in common with seed 0 of another), so seed variation is the residual.

Each term's share is SS_term / SS_total (eta-squared). Interactions are pooled into
"architecture x factors" and "factor x factor" so the table answers the practical
question: does choosing the model matter more than choosing the training recipe, and
how much of the spread is just the seed?

Usage
-----
  python paper1/scripts/variance_decomposition.py
"""
import itertools
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "analysis" / "results"
OUT = RES / "variance_decomposition.json"

AXES = ["architecture", "bands", "augment", "weights"]  # seed is axis 4 (residual)
RUN_RE = re.compile(r"^(?P<arch>.+?)__(?P<cond>.+?)__s(?P<seed>\d+)$")


def load_grid(split, metric="mean_iou"):
    runs = json.load(open(RES / f"rerun_eval_{split}.json"))["results"]
    archs = sorted({RUN_RE.match(r["run"])["arch"] for r in runs})
    seeds = sorted({int(r["seed"]) for r in runs})
    y = np.full((len(archs), 2, 2, 2, len(seeds)), np.nan)
    for r in runs:
        a = archs.index(RUN_RE.match(r["run"])["arch"])
        b = int(bool(r["use_nir"]))
        g = int(bool(r["augment"]))
        w = int(r["class_weights_mode"] != "none")
        y[a, b, g, w, seeds.index(int(r["seed"]))] = 100 * r[metric]
    if np.isnan(y).any():
        raise SystemExit(f"{split}: grid incomplete ({int(np.isnan(y).sum())} missing)")
    return y, archs


def effect_ss(y):
    """SS for every main effect and interaction of the four fixed axes."""
    grand = y.mean()
    n_total = y.size
    ss = {}
    # interaction effect for subset S = inclusion-exclusion of marginal means
    for k in range(1, 5):
        for S in itertools.combinations(range(4), k):
            drop = tuple(i for i in range(5) if i not in S)
            eff = np.zeros(tuple(y.shape[i] for i in S))
            for j in range(len(S) + 1):
                for T in itertools.combinations(range(len(S)), j):
                    keep = [S[t] for t in T]
                    m = y.mean(axis=tuple(i for i in range(5) if i not in keep),
                               keepdims=False) if keep else grand
                    # broadcast the T-marginal onto the S-shaped array
                    m = np.asarray(m)
                    shape = [y.shape[S[t]] if t in T else 1 for t in range(len(S))]
                    eff = eff + (-1) ** (len(S) - j) * m.reshape(shape)
            ss[S] = float((eff ** 2).sum() * n_total / eff.size)
    ss_total = float(((y - grand) ** 2).sum())
    ss_resid = ss_total - sum(ss.values())
    return ss, ss_resid, ss_total


def summarise(y):
    ss, ss_resid, ss_total = effect_ss(y)
    name = lambda S: " x ".join(AXES[i] for i in S)
    terms = {name(S): v for S, v in ss.items()}
    pooled = {
        "architecture": ss[(0,)],
        "bands (NIR)": ss[(1,)],
        "augment": ss[(2,)],
        "weights": ss[(3,)],
        "factor x factor": sum(v for S, v in ss.items() if 0 not in S and len(S) > 1),
        "architecture x factors": sum(v for S, v in ss.items() if 0 in S and len(S) > 1),
        "seed (residual)": ss_resid,
    }
    return {
        "ss_total": ss_total,
        "share_pct": {k: 100 * v / ss_total for k, v in pooled.items()},
        "terms_share_pct": {k: 100 * v / ss_total for k, v in terms.items()},
        "seed_sd_pp": float(np.sqrt(ss_resid / (y.size - np.prod(y.shape[:4])))),
        "range_of_cell_means_pp": float(np.ptp(y.mean(axis=4))),
    }


def main():
    out = {"source": "rerun_eval_{split}.json", "metric": "mean_iou (percentage points)"}
    for split in ("val", "test", "external"):
        y, archs = load_grid(split)
        out[split] = summarise(y)
        out["architectures"] = archs
        s = out[split]["share_pct"]
        print(f"{split:9s} " + "  ".join(f"{k} {v:5.1f}%" for k, v in s.items())
              + f"  | seed SD {out[split]['seed_sd_pp']:.2f} pp")
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
