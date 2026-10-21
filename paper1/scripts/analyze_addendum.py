#!/usr/bin/env python3
"""Test the addendum hypotheses exactly as pre-registered.

`paper1/configs/addendum_prereg_2026-10.md`, sections A, C and D (B is
`variance_decomposition.py`). Reads only the evaluation files written by
`hpc/p1_addendum_eval.slurm` and the main rerun files; writes
`analysis/results/addendum_results.json`. A section whose inputs are missing is
reported as pending, not skipped silently.

Intervals: bootstrap over seeds within architecture (10,000 resamples), i.e. each
architecture's five seeds are resampled with replacement and the statistic recomputed,
so architectures are never mixed into one pool. Block-level statistics resample blocks.

Usage
-----
  python paper1/scripts/analyze_addendum.py
"""
import json
import re
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parent.parent / "analysis" / "results"
OUT = RES / "addendum_results.json"
B = 10_000
RNG = np.random.default_rng(20261007)
RUN_RE = re.compile(r"^(?P<arch>.+?)__(?P<cond>.+?)__s(?P<seed>\d+)$")
ARCH_A = ["unetpp_effb3", "segformer_b2", "upernet_convnextt"]
EXT_MIOU_LAMPHUN = None  # read from ensemble_eval_external.json


def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def by_run(payload, metric="mean_iou"):
    return {r["run"]: 100 * r[metric] for r in payload["results"]}


def seed_boot(table):
    """table: {arch: array of per-seed values} -> (mean, lo, hi) of the grand mean."""
    archs = sorted(table)
    vals = [np.asarray(table[a], float) for a in archs]
    est = float(np.mean(np.concatenate(vals)))
    bs = np.empty(B)
    for i in range(B):
        bs[i] = np.mean(np.concatenate([v[RNG.integers(0, len(v), len(v))] for v in vals]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"mean": est, "ci95": [float(lo), float(hi)], "excludes_zero": bool(lo > 0 or hi < 0)}


def block_boot(diffs):
    d = np.asarray(diffs, float)
    bs = np.array([d[RNG.integers(0, len(d), len(d))].mean() for _ in range(B)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"mean": float(d.mean()), "ci95": [float(lo), float(hi)], "n_blocks": len(d),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def kendall_tau(x, y):
    x, y = np.asarray(x), np.asarray(y)
    s = 0
    n = len(x)
    for i in range(n):
        for j in range(i + 1, n):
            s += np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
    return float(s / (n * (n - 1) / 2))


# ---------------------------------------------------------------- A
def section_a():
    rt, rx = load("randomsplit_eval_test.json"), load("randomsplit_eval_external.json")
    st, sx = load("rerun_eval_test.json"), load("rerun_eval_external.json")
    if not (rt and rx):
        return {"status": "pending: randomsplit_eval_{test,external}.json"}
    rt, rx, st, sx = by_run(rt), by_run(rx), by_run(st), by_run(sx)
    conds = ["b0_rgb_plain", "b1_rgb_aug", "b4_rgbnir_plain", "b5_rgbnir_aug"]
    runs = [f"{a}__{c}__s{s}" for a in ARCH_A for c in conds for s in range(5)]
    missing = [r for r in runs if r not in rt or r not in rx]
    if missing:
        return {"status": f"incomplete: {len(missing)} runs missing", "missing": missing}

    def per_arch(f):
        return {a: [f(a, s) for s in range(5)] for a in ARCH_A}

    def g(t, a, c, s):
        return t[f"{a}__{c}__s{s}"]

    gap = lambda own, ext, a, s, c: g(own, a, c, s) - g(ext, a, c, s)
    h1 = per_arch(lambda a, s: np.mean([gap(rt, rx, a, s, c) - gap(st, sx, a, s, c)
                                        for c in conds]))
    nir = lambda t, a, s: np.mean([g(t, a, "b4_rgbnir_plain", s) - g(t, a, "b0_rgb_plain", s),
                                   g(t, a, "b5_rgbnir_aug", s) - g(t, a, "b1_rgb_aug", s)])
    aug = lambda t, a, s: np.mean([g(t, a, "b1_rgb_aug", s) - g(t, a, "b0_rgb_plain", s),
                                   g(t, a, "b5_rgbnir_aug", s) - g(t, a, "b4_rgbnir_plain", s)])
    cells = [(a, c) for a in ARCH_A for c in conds]
    cm = lambda t, a, c: np.mean([g(t, a, c, s) for s in range(5)])
    return {
        "status": "done",
        "gap_random_mean_pp": float(np.mean([gap(rt, rx, a, s, c) for a in ARCH_A
                                             for c in conds for s in range(5)])),
        "gap_spatial_mean_pp": float(np.mean([gap(st, sx, a, s, c) for a in ARCH_A
                                              for c in conds for s in range(5)])),
        "A-H1_gap_random_minus_spatial": seed_boot(h1),
        "nir_effect_random_test": seed_boot(per_arch(lambda a, s: nir(rt, a, s))),
        "nir_effect_spatial_test": seed_boot(per_arch(lambda a, s: nir(st, a, s))),
        "nir_effect_external_random_models": seed_boot(per_arch(lambda a, s: nir(rx, a, s))),
        "A-H2_nir_random_minus_spatial": seed_boot(
            per_arch(lambda a, s: nir(rt, a, s) - nir(st, a, s))),
        "aug_effect_random_test": seed_boot(per_arch(lambda a, s: aug(rt, a, s))),
        "aug_effect_spatial_test": seed_boot(per_arch(lambda a, s: aug(st, a, s))),
        "A-H3_aug_random_minus_spatial": seed_boot(
            per_arch(lambda a, s: aug(rt, a, s) - aug(st, a, s))),
        "A-E1_kendall_tau_own_vs_external": {
            "random": kendall_tau([cm(rt, *k) for k in cells], [cm(rx, *k) for k in cells]),
            "spatial": kendall_tau([cm(st, *k) for k in cells], [cm(sx, *k) for k in cells]),
            "n_cells": len(cells)},
    }


# ---------------------------------------------------------------- C
CODES = {"b0_rgb_plain": (0, 0, 0), "b1_rgb_aug": (0, 1, 0), "b2_rgb_wt": (0, 0, 1),
         "b3_rgb_aug_wt": (0, 1, 1), "b4_rgbnir_plain": (1, 0, 0), "b5_rgbnir_aug": (1, 1, 0),
         "b6_rgbnir_wt": (1, 0, 1), "b7_rgbnir_aug_wt": (1, 1, 1)}


def factor_effect(t, factor):
    """Main effect per arch and seed: mean(high) - mean(low) over the 8 conditions."""
    i = ["bands", "augment", "weights"].index(factor)
    out = {}
    for run, v in t.items():
        m = RUN_RE.match(run)
        out.setdefault(m["arch"], {}).setdefault(int(m["seed"]), {})[m["cond"]] = v
    tab = {}
    for a, seeds in out.items():
        vals = []
        for s, cells in sorted(seeds.items()):
            if len(cells) != 8:
                continue
            hi = np.mean([v for c, v in cells.items() if CODES[c][i] == 1])
            lo = np.mean([v for c, v in cells.items() if CODES[c][i] == 0])
            vals.append(hi - lo)
        tab[a] = vals
    return tab


def block_miou(payload):
    return {b: 100 * m["mean_iou"] for b, m in payload["per_block"].items()}


def section_c():
    ens = {v: load(f"ensemble_eval_roiet_{v}.json") for v in ("raw", "zscene")}
    runs = {v: load(f"rerun_eval_roiet_{v}.json") for v in ("raw", "zscene")}
    lam = load("ensemble_eval_external.json")
    if not all(ens.values()) or not all(runs.values()):
        return {"status": "pending: ensemble_eval_roiet_* / rerun_eval_roiet_*"}
    out = {"status": "done", "lamphun_external_ensemble_miou": 100 * lam["mean_iou"]}
    for v in ("raw", "zscene"):
        bm = block_miou(ens[v])
        out[f"ensemble_{v}"] = {
            "miou": 100 * ens[v]["mean_iou"], "oa": 100 * ens[v]["overall_accuracy"],
            "per_class_iou": {k: 100 * x for k, x in ens[v]["per_class_iou"].items()},
            "C-H1_block_mean_minus_lamphun": block_boot(
                [x - 100 * lam["mean_iou"] for x in bm.values()])}
        t = by_run(runs[v])
        for f in ("augment", "bands", "weights"):
            tab = factor_effect(t, f)
            out[f"{f}_effect_{v}"] = {"pooled": seed_boot(tab),
                                      "per_arch": {a: seed_boot({a: x}) for a, x in tab.items()}}
    bz, br = block_miou(ens["zscene"]), block_miou(ens["raw"])
    out["C-H3_zscene_minus_raw_by_block"] = block_boot([bz[b] - br[b] for b in bz if b in br])
    out["C-H2"] = out["augment_effect_zscene"]["pooled"]
    return out


# ---------------------------------------------------------------- D
def miou6(cm):
    cm = np.asarray(cm, float)
    k = cm.shape[0]
    iou = []
    for c in range(1, k):
        tp = cm[c, c]
        fn = cm[c].sum() - tp                 # includes background predictions
        fp = cm[1:, c].sum() - tp             # predictions of c on real-class pixels only
        iou.append(tp / (tp + fn + fp) if tp + fn + fp else np.nan)
    return 100 * float(np.nanmean(iou)), [100 * float(x) for x in iou]


def section_d():
    out = {"status": "done"}
    for split in ("val", "test", "external"):
        p = load(f"background_eval_{split}.json")
        if not p:
            return {"status": f"pending: background_eval_{split}.json"}
        tab, bgshare, per_class = {}, {}, {}
        for r in p["results"]:
            arch = RUN_RE.match(r["run"])["arch"]
            m, pc = miou6(r["confusion_matrix"])
            cm = np.asarray(r["confusion_matrix"], float)
            tab.setdefault(r["arm"], {}).setdefault(arch, {})[r["seed"]] = m
            bgshare.setdefault(r["arm"], []).append(100 * cm[:, 0].sum() / cm.sum())
            per_class.setdefault(r["arm"], []).append(pc)
        arr = lambda arm: {a: [v[s] for s in sorted(v)] for a, v in tab[arm].items()}
        diff = lambda x, y: {a: [tab[x][a][s] - tab[y][a][s] for s in sorted(tab[x][a])]
                             for a in tab[x]}
        names = p["results"][0]["class_names"][1:]
        out[split] = {
            "miou6": {arm: seed_boot(arr(arm)) for arm in tab},
            "predicted_background_share_pct": {k: float(np.mean(v)) for k, v in bgshare.items()},
            "per_class_iou6": {arm: dict(zip(names, np.mean(v, 0).tolist()))
                               for arm, v in per_class.items()},
            "ign_minus_std": seed_boot(diff("ign", "std")),
            "sup_minus_std": seed_boot(diff("sup", "std")),
            "ign_minus_sup": seed_boot(diff("ign", "sup")),
        }
    out["D-H1"] = out["external"]["ign_minus_std"]
    out["D-H2"] = out["external"]["sup_minus_std"]
    return out


def main():
    res = {"prereg": "paper1/configs/addendum_prereg_2026-10.md",
           "bootstrap": f"{B} resamples, seeds within architecture; blocks for block stats",
           "A": section_a(), "C": section_c(), "D": section_d()}
    OUT.write_text(json.dumps(res, indent=2))
    for k in ("A", "C", "D"):
        print(k, res[k].get("status"))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
