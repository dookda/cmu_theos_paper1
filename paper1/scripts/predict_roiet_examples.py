#!/usr/bin/env python3
"""Ensemble predictions on a few Roi Et tiles, for the Roi Et figure only.

Not an evaluation: the Roi Et scores were read once by addendum C. This recomputes, with
the same locked ensemble and the same deterministic inference, the prediction maps of a
handful of tiles so they can be shown. Tiles are chosen by a fixed rule before any
prediction is made: for each of urban, agriculture, vegetation and water, the Roi Et tile
whose labelled pixels contain the largest share of that class (each tile used once).

Writes  analysis/results/roiet_examples.npz   (names, labels, raw/zscene 8-bit RGB,
        background-suppressed ensemble prediction on raw and on zscene input)

Usage (Erawan):  python scripts/predict_roiet_examples.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_rerun import build_model  # noqa: E402

P = Path("/project/gisnorth/paper1")
RES = P / "analysis" / "results"
ROOT = Path("/project/gisnorth/paper1_roiet")
CLASSES = ["background", "vegetation", "water", "urban", "agriculture", "bare_soil", "road"]
PICK = ["urban", "agriculture", "vegetation", "water"]
MEAN4, STD4 = [0.485, 0.456, 0.406, 0.5], [0.229, 0.224, 0.225, 0.25]


def read(variant, name):
    rgb = cv2.cvtColor(cv2.imread(str(ROOT / variant / "tiles" / f"{name}.png")), cv2.COLOR_BGR2RGB)
    nir = cv2.imread(str(ROOT / variant / "tiles" / f"{name}_nir.png"), cv2.IMREAD_GRAYSCALE)
    return rgb, nir


def main():
    names = json.loads((RES / "roiet_splits.json").read_text())["splits"]["external"]
    labs = {}
    for n in names:
        lab = cv2.imread(str(ROOT / "raw" / "semantic_labels" / f"{n}.png"), cv2.IMREAD_GRAYSCALE)
        lab[lab == 7] = 0                       # grass folded into background, as in training
        labs[n] = lab
    chosen = []
    for c in PICK:
        k = CLASSES.index(c)
        share = {n: (l == k).sum() / max((l < 7).sum(), 1) for n, l in labs.items() if n not in chosen}
        chosen.append(max(share, key=share.get))
    print("chosen:", dict(zip(PICK, chosen)))

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    members = Counter(json.loads((RES / "ensemble_selection.json").read_text())["selected"])
    probs = {v: np.zeros((len(chosen), 7, 512, 512), np.float32) for v in ("raw", "zscene")}
    for run, w in members.items():
        meta = json.loads(next((P / "rerun_output" / run).glob("logs/*_best_metrics.json")).read_text())["run"]
        nin = 4 if meta.get("use_nir") else 3
        m = build_model(meta["model"], meta["encoder"], nin, 7)
        st = torch.load(P / "rerun_output" / run / "models" / f"{meta['model']}_best.pth",
                        map_location="cpu")["model_state_dict"]
        m.load_state_dict({k[7:] if k.startswith("module.") else k: v for k, v in st.items()})
        m.to(dev).eval()
        for v in probs:
            x = np.stack([np.dstack(read(v, n)) for n in chosen]).astype(np.float32) / 255.0
            x = (x - np.array(MEAN4)) / np.array(STD4)
            t = torch.from_numpy(x[..., :nin].transpose(0, 3, 1, 2)).float().to(dev)
            with torch.no_grad():
                probs[v] += w * torch.softmax(m(t), 1).cpu().numpy()
        print("  +", run, f"(x{w})", flush=True)
    out = {"names": np.array(chosen), "picked_for": np.array(PICK),
           "labels": np.stack([labs[n] for n in chosen])}
    for v in probs:
        p = probs[v]
        p[:, 0] = -1                            # background suppressed (six-class map)
        out[f"pred_{v}"] = p.argmax(1).astype(np.uint8)
        out[f"rgb_{v}"] = np.stack([read(v, n)[0] for n in chosen])
    np.savez_compressed(RES / "roiet_examples.npz", **out)
    print("wrote", RES / "roiet_examples.npz")


if __name__ == "__main__":
    main()
