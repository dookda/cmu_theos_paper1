#!/usr/bin/env python3
"""F5 (prereg 25eb5c1): calibration and selective prediction of the locked ensemble.

For the 7-member validation-selected ensemble, averaged softmax is computed on val, test,
external and Roi Et (zscene). A single temperature is fitted on val only (applied to the
log of the averaged probability) and then fixed. Reported per split, before and after
temperature scaling: expected calibration error (15 equal-width bins on the max
probability), multi-class Brier score, and the risk-coverage curve with its area (AURC),
on labelled pixels (labels < 7). Brier and the temperature fit use a fixed random pixel
subsample (seed 0) to bound memory; ECE and AURC use every labelled pixel.

Usage (Erawan):  python scripts/eval_calibration.py --out analysis/results/calibration.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import eval_ensemble as ee

P = Path("/project/gisnorth/paper1")
RES = P / "analysis" / "results"
NC, NB = 7, 15


def ens_probs(members, split, splits_file, device):
    total, labels = None, None
    for name in members:
        p, _, lb, _ = ee.member_probs(P / "rerun_output" / name, split, str(splits_file), device, 8)
        p = p.astype(np.float32)
        total = p if total is None else total + p
        labels = lb
    return total / len(members), labels


def flat(prob, lab):
    lab = lab.reshape(-1)
    prob = np.moveaxis(prob, 1, -1).reshape(-1, NC)
    ok = lab < NC
    return prob[ok], lab[ok].astype(np.int64)


def temper(prob, T):
    lg = np.log(np.clip(prob, 1e-8, 1)) / T
    lg -= lg.max(1, keepdims=True)
    e = np.exp(lg)
    return e / e.sum(1, keepdims=True)


def fit_T(prob, lab):
    best = (np.inf, 1.0)
    for T in np.linspace(0.5, 3.0, 51):
        q = temper(prob, T)
        nll = -np.mean(np.log(np.clip(q[np.arange(len(lab)), lab], 1e-8, 1)))
        best = min(best, (nll, float(T)))
    return best[1]


def metrics(prob, lab, sub):
    conf = prob.max(1)
    pred = prob.argmax(1)
    corr = (pred == lab).astype(np.float64)
    bins = np.minimum((conf * NB).astype(int), NB - 1)
    ece = 0.0
    for b in range(NB):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(corr[m].mean() - conf[m].mean())
    o = np.argsort(-conf)
    risk = 1 - np.cumsum(corr[o]) / np.arange(1, len(o) + 1)
    cov = np.arange(1, len(o) + 1) / len(o)
    pick = np.linspace(0, len(o) - 1, 101).astype(int)
    ps, ls = prob[sub], lab[sub]
    brier = float(np.mean(np.sum((ps - np.eye(NC)[ls]) ** 2, 1)))
    return {"ece_pct": 100 * ece, "brier": brier, "accuracy_pct": 100 * corr.mean(),
            "aurc_pct": 100 * float(risk.mean()),
            "risk_at_coverage": {f"{c:.2f}": 100 * float(risk[i]) for c, i in
                                 zip(cov[pick][::10], pick[::10])}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RES / "calibration.json"))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    members = json.loads((RES / "ensemble_selection.json").read_text())["selected"]
    rng = np.random.default_rng(0)
    out = {"prereg": "F5, commit 25eb5c1", "members": members, "bins": NB}

    prob, lab = flat(*ens_probs(members, "val", RES / "spatial_splits.json", device))
    sub = rng.choice(len(lab), size=min(2_000_000, len(lab)), replace=False)
    T = fit_T(prob[sub], lab[sub])
    out["temperature_fitted_on_val"] = T
    out["val"] = {"raw": metrics(prob, lab, sub), "tempered": metrics(temper(prob, T), lab, sub)}
    print("val", T, out["val"]["raw"]["ece_pct"], flush=True)

    jobs = [("test", RES / "spatial_splits.json", None), ("external", RES / "spatial_splits.json", None),
            ("roiet_zscene", RES / "roiet_splits.json", "/project/gisnorth/paper1_roiet/zscene")]
    for name, sf, root in jobs:
        if root:
            ee.RGB_ROOT = ee.NIR_ROOT = root
        prob, lab = flat(*ens_probs(members, "external" if name == "roiet_zscene" else name, sf, device))
        sub = rng.choice(len(lab), size=min(2_000_000, len(lab)), replace=False)
        out[name] = {"raw": metrics(prob, lab, sub), "tempered": metrics(temper(prob, T), lab, sub)}
        print(name, out[name]["raw"]["ece_pct"], out[name]["tempered"]["ece_pct"], flush=True)
    Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
