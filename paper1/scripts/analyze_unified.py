#!/usr/bin/env python3
"""Pre-registered analyses F1-F4 (prereg 25eb5c1) computed from the unified pass files.

Input:  analysis/results/unified/<root>/<run>.npz   (written by eval_unified.py)
Output: analysis/results/unified_results.json

  F1  variance shares and Kendall tau of the 72 cells per regime (val, test, external,
      Roi Et zscene; random-test for the A subset), block-free (pooled over tiles)
  F2  edge-minus-interior mIoU on the own test split, random vs buffered (A design)
  F3  external mIoU lost when NIR is shuffled / low-passed / set to the training mean
  F4  factor-effect signs and architecture ranking under P7, P6-eval and P6-sup
  E/H pretraining: E-H1 (each RS encoder vs the validation-chosen ImageNet comparator on
      external), H1 frozen vs fine-tuned, P-H1 test-minus-external gap Satlas vs ImageNet
      Swin-V2-B in the same construction
  G   N-H2 (nrg vs b1) and N-H3 (rgb_nirlp vs b5) non-inferiority on external, margin 0.5
  F6  scores against original labels vs the audit consensus; cross-run disagreement with
      the original label as a detector of audited label errors (AUPRC, precision@k)

Intervals: bootstrap over seeds within architecture (10,000), as in analyze_addendum.py.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_addendum import kendall_tau, miou6, seed_boot  # noqa: E402
from variance_decomposition import effect_ss               # noqa: E402

RES = Path(__file__).resolve().parent.parent / "analysis" / "results"
UNI = RES / "unified"
REVIEW2 = Path(__file__).resolve().parent.parent / "reports" / "label_review_round2"
AUDIT_KEY = RES / "label_review_key.json"
RUN_RE = re.compile(r"^(?P<arch>.+?)__(?P<cond>.+?)__s(?P<seed>\d+)$")
CODES = {"b0_rgb_plain": (0, 0, 0), "b1_rgb_aug": (0, 1, 0), "b2_rgb_wt": (0, 0, 1),
         "b3_rgb_aug_wt": (0, 1, 1), "b4_rgbnir_plain": (1, 0, 0), "b5_rgbnir_aug": (1, 1, 0),
         "b6_rgbnir_wt": (1, 0, 1), "b7_rgbnir_aug_wt": (1, 1, 1)}
MARGIN = 0.5


def miou7(cm):
    cm = np.asarray(cm, float)
    iou = np.diag(cm) / (cm.sum(0) + cm.sum(1) - np.diag(cm))
    return 100 * float(np.nanmean(iou))


def load_root(root):
    out = {}
    for f in sorted((UNI / root).glob("*.npz")):
        m = RUN_RE.match(f.stem)
        out[(m["arch"], m["cond"], int(m["seed"]))] = np.load(f, allow_pickle=False)
    return out


def metric(d, split, key="cm_std", fn=miou7):
    k = f"{split}__{key}"
    return fn(d[k].sum(0)) if k in d.files else np.nan


def grid(runs, split, key="cm_std", fn=miou7):
    archs = sorted({a for a, _, _ in runs})
    y = np.full((len(archs), 2, 2, 2, 5), np.nan)
    for (a, c, s), d in runs.items():
        if c in CODES:
            b, g, w = CODES[c]
            y[archs.index(a), b, g, w, s] = metric(d, split, key, fn)
    return y, archs


def shares(y):
    ss, resid, tot = effect_ss(y)
    names = ["architecture", "bands", "augment", "weights"]
    pooled = {"architecture": ss[(0,)], "bands (NIR)": ss[(1,)], "augment": ss[(2,)],
              "weights": ss[(3,)],
              "factor x factor": sum(v for S, v in ss.items() if 0 not in S and len(S) > 1),
              "architecture x factors": sum(v for S, v in ss.items() if 0 in S and len(S) > 1),
              "seed (residual)": resid}
    return {k: 100 * v / tot for k, v in pooled.items()}


def effects(y, archs):
    """Per factor: per-arch per-seed main effect table for seed_boot."""
    out = {}
    for i, f in enumerate(["bands", "augment", "weights"]):
        hi = np.take(y, 1, axis=1 + i).reshape(len(archs), -1, 5).mean(1)
        lo = np.take(y, 0, axis=1 + i).reshape(len(archs), -1, 5).mean(1)
        out[f] = {a: (hi - lo)[k].tolist() for k, a in enumerate(archs)}
    return out


def f1(main):
    res = {}
    cells = None
    for split in ("val", "test", "external", "roiet_zscene"):
        y, archs = grid(main, split)
        if np.isnan(y).any():
            res[split] = {"status": f"incomplete ({int(np.isnan(y).sum())} missing)"}
            continue
        cm = y.mean(-1).reshape(-1)
        cells = cells if cells is not None else {}
        cells[split] = cm
        res[split] = {"variance_share_pct": shares(y)}
    if cells and "val" in cells:
        res["kendall_tau_vs_val"] = {s: kendall_tau(cells["val"], v) for s, v in cells.items()
                                     if s != "val"}
    return res


def f2(main, rnd):
    if not rnd:
        return {"status": "pending: random-split unified files"}
    tab_r, tab_s = {}, {}
    for (a, c, s), d in rnd.items():
        if (a, c, s) not in main:
            continue
        er = metric(d, "test", "cm_edge") - metric(d, "test", "cm_int")
        es = metric(main[(a, c, s)], "test", "cm_edge") - metric(main[(a, c, s)], "test", "cm_int")
        tab_r.setdefault(a, []).append(er)
        tab_s.setdefault(a, []).append(es)
    diff = {a: (np.array(tab_r[a]) - np.array(tab_s[a])).tolist() for a in tab_r}
    n = sum(len(v) for v in diff.values())
    return {"status": "done" if n == 60 else f"partial ({n}/60)",
            "edge_minus_interior_random": seed_boot(tab_r),
            "edge_minus_interior_buffered": seed_boot(tab_s),
            "F2-H1_random_minus_buffered": seed_boot(diff)}


def f3(main):
    out = {}
    for var in ("nirshuf", "nirlp", "nirmean"):
        tab = {}
        for (a, c, s), d in main.items():
            if c in CODES and CODES[c][0] == 1 and f"external__cm_{var}" in d.files:
                tab.setdefault(a, {}).setdefault(s, []).append(
                    metric(d, "external") - metric(d, "external", f"cm_{var}"))
        t = {a: [np.mean(v[s]) for s in sorted(v)] for a, v in tab.items()}
        out[var] = seed_boot(t) if t else {"status": "pending"}
    if "ci95" in out["nirshuf"]:
        out["F3-H1_loss_below_margin"] = bool(out["nirshuf"]["ci95"][1] < MARGIN)
    out["margin_pp"] = MARGIN
    return out


def f4(main):
    out = {}
    protos = {"P7": ("cm_std", miou7),
              "P6-eval": ("cm_std", lambda cm: miou6(cm)[0]),
              "P6-sup": ("cm_sup", lambda cm: miou6(cm)[0])}
    for split in ("test", "external"):
        ranks, out[split] = {}, {}
        for p, (key, fn) in protos.items():
            y, archs = grid(main, split, key, fn)
            if np.isnan(y).any():
                out[split][p] = {"status": "incomplete"}
                continue
            eff = effects(y, archs)
            out[split][p] = {f: seed_boot(t) for f, t in eff.items()}
            ranks[p] = y.mean(axis=(1, 2, 3, 4))
            out[split][p]["arch_mean_miou"] = dict(zip(archs, ranks[p].tolist()))
        if len(ranks) == 3:
            out[split]["kendall_tau_arch_ranking"] = {
                "P7_vs_P6-eval": kendall_tau(ranks["P7"], ranks["P6-eval"]),
                "P7_vs_P6-sup": kendall_tau(ranks["P7"], ranks["P6-sup"])}
            out[split]["sign_changes"] = {
                f: sorted({p: np.sign(out[split][p][f]["mean"]) for p in protos}.items())
                for f in ("bands", "augment", "weights")}
    return out


def audit_consensus():
    """Round-2 points where both independent reviewers chose the same class."""
    root = REVIEW2
    key = {it["id"]: it for it in json.loads(AUDIT_KEY.read_text())["items"]}
    r1 = {int(k): v["answer"] for k, v in json.loads((root / "label_review_r1.json").read_text())["answers"].items()}
    r2 = {int(k): v["answer"] for k, v in json.loads((root / "label_review_r2.json").read_text())["answers"].items()}
    return {i: (key[i]["label"], r1[i]) for i in key
            if i in r1 and i in r2 and r1[i] == r2[i] and r1[i] >= 0}


def auprc(score, y):
    o = np.argsort(-score)
    y = y[o]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float(np.sum(prec * y) / max(y.sum(), 1))


def f6(main):
    cons = audit_consensus()
    if not main:
        return {"status": "pending"}
    preds = {}   # point id -> list of std predictions over runs
    acc = {"std_vs_original": [], "std_vs_audit": [], "sup_vs_original": [], "sup_vs_audit": []}
    for d in main.values():
        for sp in ("train_audit", "val", "test", "external"):
            if f"{sp}__audit_id" not in d.files:
                continue
            for pid, ps, pu in zip(d[f"{sp}__audit_id"], d[f"{sp}__audit_pred_std"],
                                   d[f"{sp}__audit_pred_sup"]):
                pid = int(pid)
                if pid not in cons:
                    continue
                orig, aud = cons[pid]
                preds.setdefault(pid, []).append(int(ps))
                acc["std_vs_original"].append(ps == orig)
                acc["std_vs_audit"].append(ps == aud)
                acc["sup_vs_original"].append(pu == orig)
                acc["sup_vs_audit"].append(pu == aud)
    ids = sorted(preds)
    err = np.array([cons[i][0] != cons[i][1] for i in ids], int)
    dis = np.array([np.mean(np.array(preds[i]) != cons[i][0]) for i in ids])
    k = int(err.sum())
    prec_k = float(err[np.argsort(-dis)][:k].mean()) if k else float("nan")
    rng = np.random.default_rng(0)
    rand = [auprc(rng.random(len(ids)), err) for _ in range(1000)]
    return {"status": "done", "n_points_consensus": len(ids), "n_label_errors": k,
            "accuracy_pct": {kk: 100 * float(np.mean(v)) for kk, v in acc.items()},
            "detector": {"auprc": auprc(dis, err), "auprc_random_mean": float(np.mean(rand)),
                         "auprc_random_p95": float(np.percentile(rand, 95)),
                         "precision_at_k": prec_k, "prevalence": float(err.mean())}}


def two_sample(a, b, B=10_000, seed=0):
    """Difference of seed means, bootstrap resampling seeds within each group."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    bs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(B)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"mean": float(a.mean() - b.mean()), "ci95": [float(lo), float(hi)],
            "excludes_zero": bool(lo > 0 or hi < 0), "n": [len(a), len(b)]}


def seeds_of(runs, arch, cond, split, key="cm_std"):
    return [metric(d, split, key) for (a, c, s), d in sorted(runs.items())
            if a == arch and c == cond]


def pretraining(main, fnd, gh):
    if not fnd:
        return {"status": "pending: foundation unified files"}
    val = {a: np.mean(seeds_of(main, a, "b1_rgb_aug", "val"))
           for a in sorted({a for a, _, _ in main})}
    comp = max(val, key=val.get)
    out = {"comparator": comp, "comparator_val_miou": val[comp]}
    for enc in ("satlas_swinb", "dinov3_vitl_sat"):
        for split in ("external", "test"):
            x = seeds_of(fnd, enc, "b1_rgb_aug", split)
            if len(x) == 5:
                out[f"{enc}_{split}"] = {"mean": float(np.mean(x)),
                                          "vs_comparator": two_sample(x, seeds_of(main, comp, "b1_rgb_aug", split))}
        fr = seeds_of(gh, enc, "b1_rgb_aug_frozen", "external") if gh else []
        ft = seeds_of(fnd, enc, "b1_rgb_aug", "external")
        if len(fr) == 5 and len(ft) == 5:
            out[f"{enc}_frozen_minus_finetuned_external"] = two_sample(fr, ft)
    if gh:
        sat = {s: d for (a, c, s), d in fnd.items() if a == "satlas_swinb"}
        img = {s: d for (a, c, s), d in gh.items() if a == "satlas_swinv2b_imagenet"}
        common = sorted(set(sat) & set(img))
        if len(common) == 5:
            gap = lambda d: metric(d, "test") - metric(d, "external")
            out["P-H1_gap_satlas_minus_imagenet"] = seed_boot(
                {"swin": [gap(sat[s]) - gap(img[s]) for s in common]})
            out["imagenet_swinv2b_external"] = float(np.mean([metric(img[s], "external") for s in common]))
    return out


def nir_mechanism(main, gh):
    if not gh:
        return {"status": "pending: G unified files"}
    out = {"margin_pp": MARGIN}
    for name, cond, ref in (("N-H2_nrg_minus_b1", "g1_nrg_aug", "b1_rgb_aug"),
                            ("N-H3_nirlp_minus_b5", "g2_rgbnirlp_aug", "b5_rgbnir_aug")):
        for split in ("external", "test", "roiet_zscene"):
            tab = {}
            for (a, c, s), d in gh.items():
                if c == cond and (a, ref, s) in main:
                    tab.setdefault(a, []).append(metric(d, split) - metric(main[(a, ref, s)], split))
            if sum(len(v) for v in tab.values()) == 15:
                r = seed_boot(tab)
                r["non_inferior"] = bool(r["ci95"][0] > -MARGIN)
                out[f"{name}_{split}"] = r
    return out


def main():
    main_runs = load_root("rerun_output")
    rnd = load_root("rerun_output_randomsplit")
    res = {"prereg": "F1-F4, commit 25eb5c1", "n_main_runs": len(main_runs),
           "F1": f1(main_runs), "F2": f2(main_runs, rnd), "F3": f3(main_runs),
           "F4": f4(main_runs), "F6": f6(main_runs),
           "E_H": pretraining(main_runs, load_root("rerun_output_foundation"), load_root("rerun_output_gh")),
           "G": nir_mechanism(main_runs, load_root("rerun_output_gh"))}
    (RES / "unified_results.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: (v.get("status") if isinstance(v, dict) else v)
                      for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
