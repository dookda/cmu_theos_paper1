#!/usr/bin/env python3
"""Unified evaluation pass for Paper 1 addenda F2-F6 (prereg 25eb5c1).

One read per checkpoint and split. For every run found under the given result roots it
writes `<out>/<root name>/<run>.npz` holding, per split, PER-TILE confusion matrices so
every later analysis is a computation on saved files:

  cm_std     7x7, ordinary argmax                       (rows reference, cols prediction)
  cm_sup     7x7, background logit suppressed
  cm_edge    7x7, ordinary argmax, 32-px border strip (the overlap zone with neighbours)
  cm_int     7x7, ordinary argmax, interior
  conf_hist  7 x 20 x 2, max-softmax histogram by predicted class and correctness
  cm_nirshuf / cm_nirlp / cm_nirmean   (4-channel runs only) NIR shuffled across tiles of
             the split (fixed permutation), low-passed to 2 m (4x4 mean, bilinear back,
             as in addendum G2), or replaced by the training-set mean
  names      tile names
  audit_*    (F6) for the 175 blinded-audit points that fall in the split: point id, and
             the predicted class (ordinary and background-suppressed) and max softmax there

Runs that already have an output file are skipped, so a later pass covers only new
checkpoints and no checkpoint is read twice. Images of a split are decoded once per input
mode and cached in memory; normalisation matches `get_val_transform` exactly.

Usage (Erawan, gpu-h100):
  python scripts/eval_unified.py --roots rerun_output rerun_output_ignbg ... \
      --roiet-roots rerun_output rerun_output_gh --out analysis/results/unified
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_rerun import build_model  # noqa: E402

P = Path("/project/gisnorth/paper1")
RES = P / "analysis" / "results"
RGB_ROOT = Path("/project/gisnorth/cmu_theos_ai_data_fixed")
NIR_ROOT = Path("/project/gisnorth/cmu_theos_ai_data_nir_only")
ROIET_ROOT = Path("/project/gisnorth/paper1_roiet/zscene")
SPATIAL = RES / "spatial_splits.json"
RANDOM = RES / "random_splits.json"
ROIET = RES / "roiet_splits.json"
MEAN3, STD3 = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
MEAN4, STD4 = MEAN3 + [0.5], STD3 + [0.25]
EDGE, NBINS, NC = 32, 20, 7
AUDIT_KEY = RES / "label_review_key.json"   # same 175 points in both audit rounds

_cache = {}


def mode_of(meta):
    if meta.get("nir_mode"):
        return meta["nir_mode"]            # nrg | lowpass
    return "rgbn" if meta.get("use_nir") else "rgb"


def lowpass(nir):
    h, w = nir.shape
    return cv2.resize(cv2.resize(nir, (w // 4, h // 4), interpolation=cv2.INTER_AREA),
                      (w, h), interpolation=cv2.INTER_LINEAR)


def load_split(splits_file, split, mode, root, only=None, labels_root=None):
    """uint8 images N,H,W,C in model channel order; labels N,H,W (remapped; 255 kept).

    `only` restricts the split to the given tile names (train tiles holding audit points).
    """
    key = (str(splits_file), split, mode, str(root), bool(only), str(labels_root))
    if key in _cache:
        return _cache[key]
    payload = json.loads(Path(splits_file).read_text())
    remap = {int(k): int(v) for k, v in payload.get("remap", {}).items()}
    names = payload["splits"][split]
    if only:
        names = [n for n in names if n in only]
    imgs, labs = [], []
    for n in names:
        rgb = cv2.cvtColor(cv2.imread(str(root / "tiles" / f"{n}.png")), cv2.COLOR_BGR2RGB)
        if mode != "rgb":
            nir = cv2.imread(str(root / "tiles" / f"{n}_nir.png"), cv2.IMREAD_GRAYSCALE)
            if nir is None:
                raise FileNotFoundError(f"{n}_nir.png under {root}")
        if mode == "rgb":
            img = rgb
        elif mode == "rgbn":
            img = np.dstack([rgb, nir])
        elif mode == "nrg":
            img = np.dstack([nir, rgb[..., 0], rgb[..., 1]])
        else:
            img = np.dstack([rgb, lowpass(nir)])
        lab = cv2.imread(str((labels_root or root) / "semantic_labels" / f"{n}.png"), cv2.IMREAD_GRAYSCALE)
        for s, d in remap.items():
            lab[lab == s] = d
        imgs.append(img)
        labs.append(lab)
    _cache[key] = (np.stack(imgs), np.stack(labs), names)
    return _cache[key]


def train_nir_mean(splits_file):
    key = ("nirmean", str(splits_file))
    if key not in _cache:
        names = json.loads(Path(splits_file).read_text())["splits"]["train"]
        _cache[key] = float(np.mean([cv2.imread(str(NIR_ROOT / "tiles" / f"{n}_nir.png"),
                                                cv2.IMREAD_GRAYSCALE).mean() for n in names]))
    return _cache[key]


def per_tile_cm(pred, t, mask=None):
    """pred, t: (B,H,W) long on device -> (B,7,7) int64 numpy."""
    out = []
    for i in range(pred.shape[0]):
        ok = t[i] < NC
        if mask is not None:
            ok = ok & mask
        idx = t[i][ok] * NC + pred[i][ok]
        out.append(torch.bincount(idx, minlength=NC * NC).view(NC, NC))
    return torch.stack(out).cpu().numpy()


@torch.no_grad()
def run_split(model, imgs, labs, mode, device, nir_variant=None, perm=None, nir_mean=None,
              full=True, bs=8, names=None, audit=None):
    mean, std = (MEAN4, STD4) if imgs.shape[-1] == 4 else (MEAN3, STD3)
    mean = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=device).view(1, -1, 1, 1)
    h, w = labs.shape[1:]
    edge = torch.ones(h, w, dtype=torch.bool, device=device)
    edge[EDGE:-EDGE, EDGE:-EDGE] = False
    res = {k: [] for k in (["cm_std", "cm_sup", "cm_edge", "cm_int"] if full else ["cm"])}
    hist = torch.zeros(NC * NBINS * 2, dtype=torch.long, device=device)
    aud = {"audit_id": [], "audit_pred_std": [], "audit_pred_sup": [], "audit_conf": []}
    for i in range(0, len(imgs), bs):
        x = imgs[i:i + bs].copy()
        if nir_variant == "shuf":
            x[..., 3] = imgs[perm[i:i + bs], ..., 3]
        elif nir_variant == "lp":
            x[..., 3] = np.stack([lowpass(a) for a in x[..., 3]])
        elif nir_variant == "mean":
            x[..., 3] = np.uint8(round(nir_mean))
        x = torch.from_numpy(x).to(device).permute(0, 3, 1, 2).float() / 255.0
        x = (x - mean) / std
        t = torch.from_numpy(labs[i:i + bs]).to(device).long()
        logits = model(x)
        pred = logits.argmax(1)
        if not full:
            res["cm"].append(per_tile_cm(pred, t))
            continue
        lg = logits.clone()
        lg[:, 0] = float("-inf")
        res["cm_std"].append(per_tile_cm(pred, t))
        res["cm_sup"].append(per_tile_cm(lg.argmax(1), t))
        res["cm_edge"].append(per_tile_cm(pred, t, edge))
        res["cm_int"].append(per_tile_cm(pred, t, ~edge))
        conf = torch.softmax(logits.float(), 1).amax(1)
        if audit and names is not None:
            pred_sup = lg.argmax(1)
            for j, n in enumerate(names[i:i + bs]):
                for pid, x0, y0 in audit.get(n, []):
                    aud["audit_id"].append(pid)
                    aud["audit_pred_std"].append(int(pred[j, y0, x0]))
                    aud["audit_pred_sup"].append(int(pred_sup[j, y0, x0]))
                    aud["audit_conf"].append(float(conf[j, y0, x0]))
        ok = t < NC
        b = (conf[ok] * NBINS).long().clamp(max=NBINS - 1)
        hist += torch.bincount(pred[ok] * NBINS * 2 + b * 2 + (pred[ok] == t[ok]).long(),
                               minlength=NC * NBINS * 2)
    out = {k: np.concatenate(v) for k, v in res.items()}
    if full:
        out["conf_hist"] = hist.view(NC, NBINS, 2).cpu().numpy()
        if aud["audit_id"]:
            out.update({k: np.array(v) for k, v in aud.items()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--roiet-roots", nargs="*", default=[],
                    help="roots whose runs are also scored on Roi Et zscene")
    ap.add_argument("--out", default=str(RES / "unified"))
    ap.add_argument("--limit", type=int, default=None, help="dev: first N runs per root")
    ap.add_argument("--splits", nargs="+", default=["val", "test", "external"])
    ap.add_argument("--labels-root", default=None,
                    help="read Lamphun semantic_labels from this root for every run (Roi Et keeps its "
                         "own); used to re-score the NIR arm against the RGB reference")
    ap.add_argument("--only", default=None, help="only runs whose name contains this substring")
    ap.add_argument("--nir-only", action="store_true", help="only runs that read the NIR data root")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audit = {}
    for it in json.loads(AUDIT_KEY.read_text())["items"]:
        audit.setdefault(it["tile"], []).append((it["id"], it["x"], it["y"]))

    for root in args.roots:
        root_p = P / root
        runs = sorted(p.parent.parent for p in root_p.glob("*/logs/*_best_metrics.json"))
        runs = runs[:args.limit] if args.limit else runs
        if args.only:
            runs = [r for r in runs if args.only in r.name]
        out_dir = Path(args.out) / root
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"== {root}: {len(runs)} runs", flush=True)
        for run_dir in runs:
            dest = out_dir / f"{run_dir.name}.npz"
            if dest.exists():
                continue
            meta = json.loads(next(run_dir.glob("logs/*_best_metrics.json")).read_text())["run"]
            mode = mode_of(meta)
            if args.nir_only and mode == "rgb":
                continue
            in_ch = 4 if mode in ("rgbn", "lowpass") else 3
            model = build_model(meta["model"], meta["encoder"], in_ch, meta.get("num_classes", NC))
            state = torch.load(run_dir / "models" / f"{meta['model']}_best.pth",
                               map_location="cpu")["model_state_dict"]
            if next(iter(state)).startswith("module."):
                state = {k[7:]: v for k, v in state.items()}
            model.load_state_dict(state)
            model.to(device).eval()

            splits_file = RANDOM if "random" in str(meta.get("splits", "")) else SPATIAL
            data_root = RGB_ROOT if mode == "rgb" else NIR_ROOT
            save = {"meta": json.dumps({**meta, "input_mode": mode, "splits_file": str(splits_file),
                                        "labels_root": args.labels_root})}
            jobs = [(s, splits_file, data_root) for s in args.splits]
            if root in args.roiet_roots:
                jobs.append(("roiet_zscene", ROIET, ROIET_ROOT))
            if splits_file == SPATIAL:
                # F6: training tiles that hold audit points (label-error detection works
                # on training labels); audit outputs only are used from this entry
                jobs.append(("train_audit", SPATIAL, data_root))
            for name, sf, droot in jobs:
                split = {"roiet_zscene": "external", "train_audit": "train"}.get(name, name)
                lroot = (Path(args.labels_root) if args.labels_root and name != "roiet_zscene" else None)
                imgs, labs, names = load_split(sf, split, mode, droot,
                                               only=set(audit) if name == "train_audit" else None,
                                               labels_root=lroot)
                r = run_split(model, imgs, labs, mode, device, names=names,
                              audit=audit if name != "roiet_zscene" else None)
                for k, v in r.items():
                    save[f"{name}__{k}"] = v
                save[f"{name}__names"] = np.array(names)
                if in_ch == 4 and name not in ("roiet_zscene", "train_audit"):
                    perm = np.random.default_rng(0).permutation(len(imgs))
                    for var in ("shuf", "lp", "mean"):
                        save[f"{name}__cm_nir{var}"] = run_split(
                            model, imgs, labs, mode, device, nir_variant=var, perm=perm,
                            nir_mean=train_nir_mean(splits_file), full=False)["cm"]
            np.savez_compressed(dest, **save)
            cm = save[f"{jobs[0][0]}__cm_std"].sum(0)
            iou = np.diag(cm) / (cm.sum(0) + cm.sum(1) - np.diag(cm) + 1e-9)
            print(f"  {run_dir.name:<48} {mode:7s} {jobs[0][0]} mIoU {100 * iou.mean():.2f}",
                  flush=True)
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
