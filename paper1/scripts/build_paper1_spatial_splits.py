#!/usr/bin/env python3
"""Build spatially valid train/val/test splits for the paper1 re-run.

Replaces the random tile-level split of `prepare_tiles.py`, which put 100% of the
val/test tiles in direct pixel contact with training tiles (tiles are cut with
step = 512 - 64 = 448 px, so every tile overlaps each of its 8 neighbours).

Design
------
The two THEOS-2 scenes of 2026-01-08 overlap by 23.4 km2, so "Scene 2" is not an
independent area on its own. Tiles are therefore partitioned by footprint:

  pool A (internal)  tiles inside Scene 1, plus tiles in the Scene1/Scene2 overlap
  pool B (external)  tiles inside Scene 2 only -> never used for training

Pool A is grouped into square blocks (default 2 km, matching paper2) and whole
blocks are allocated to train/val/test, stratified by landscape composition so
every split keeps all eight classes at a realistic share.

Buffering
---------
Whole-block allocation alone does not remove leakage: tiles straddling a block
boundary still overlap tiles of the neighbouring block (paper2's block splits
retain 32.4% val / 21.6% test tiles in pixel contact with train for exactly this
reason). A tile is therefore kept in val/test/external only if *all* of its
overlapping neighbours belong to the same split. Train keeps every tile: dropping
the evaluation-side tile is what guarantees no evaluation pixel was ever seen in
training, and it costs no training data.

Inputs (all local)
------------------
  paper2/analysis/results/tiles_metadata.json   tile grid, UTM 47N bounds (4,342)
  paper2/data/labels_fixed/*.png                8-class masks, 512x512 (2,642)
  theos/lamphoon/th2_2026*_47n.tif              scene footprints

Class scheme: the study runs on seven classes; see the REMAP comment below for why
grass is folded into background.

Output
------
  paper1/analysis/results/spatial_splits.json   splits, blocks, audit, provenance
  paper1/analysis/results/spatial_splits_summary.csv

Usage
-----
  python paper1/scripts/build_paper1_spatial_splits.py
  python paper1/scripts/build_paper1_spatial_splits.py --block-size 1500
"""
import argparse
import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

# Local PROJ data breaks rasterio's CRS lookup on this machine; force the bundled copy.
os.environ.pop("PROJ_DATA", None)
os.environ.pop("PROJ_LIB", None)

ROOT = Path(__file__).resolve().parent.parent.parent
META = ROOT / "paper2" / "analysis" / "results" / "tiles_metadata.json"
LABELS = ROOT / "paper2" / "data" / "labels_fixed"
SCENES = {
    "scene1": ROOT / "theos" / "lamphoon" / "th2_20260108_034820_psp_gcs_ort_47n.tif",
    "scene2": ROOT / "theos" / "lamphoon" / "th2_20260108_034822_psp_gcs_ort_47n.tif",
}
OUT = ROOT / "paper1" / "analysis" / "results"

# The masks on disk carry eight classes, but `grass` (index 7) is only annotated
# north of ~2,051,163 m N: it holds 1-8% of the pixels there and *exactly* zero
# across the 1,284 southern tiles, including the whole Scene1/Scene2 overlap zone.
# A landscape boundary cannot be that sharp, so this is a change of annotation
# scheme mid-scene, and grass is dropped to keep the label semantics consistent.
# It is merged into background rather than vegetation because paper1's own audit
# rule ("original label is background, ensemble predicts bare soil or grass")
# established background as the class that absorbed unannotated grass.
SRC_CLASSES = ["background", "vegetation", "water", "urban",
               "agriculture", "bare_soil", "road", "grass"]
REMAP = {7: 0}                       # grass -> background
CLASSES = SRC_CLASSES[:7]
# Classes that must stay represented in every split. background is excluded: it is a
# catch-all, not a land-cover class, and it dominates the mask anyway.
STRATIFY_ON = [i for i, c in enumerate(CLASSES) if c != "background"]
FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
MIN_CLASS_SHARE = 0.002   # every non-background class holds >=0.2% of each split
TILE_PX = 512
STEP_PX = 448             # prepare_tiles.py: step = tile_size - overlap


def scene_bounds():
    import rasterio
    out = {}
    for name, path in SCENES.items():
        with rasterio.open(path) as src:
            b = src.bounds
            out[name] = (b.left, b.bottom, b.right, b.top)
    return out


def contains(tile_bounds, scene):
    return (tile_bounds["left"] >= scene[0] and tile_bounds["bottom"] >= scene[1]
            and tile_bounds["right"] <= scene[2] and tile_bounds["top"] <= scene[3])


def load_tiles():
    """Tile geometry + class composition for every labelled tile."""
    meta = json.load(open(META))
    meta = meta["tiles"] if isinstance(meta, dict) and "tiles" in meta else meta
    scenes = scene_bounds()

    tiles = {}
    for m in meta:
        name = Path(m["filename"]).stem
        mask_path = LABELS / f"{name}.png"
        if not mask_path.exists():
            continue                       # unlabelled tile
        b = m["bounds"]
        in1, in2 = contains(b, scenes["scene1"]), contains(b, scenes["scene2"])
        pool = "A" if in1 else "B" if in2 else None
        if pool is None:
            continue                       # falls outside both footprints
        a = np.asarray(Image.open(mask_path))
        if a.ndim == 3:
            a = a[..., 0]
        counts = np.bincount(a.ravel(), minlength=len(SRC_CLASSES))[:len(SRC_CLASSES)]
        for src, dst in REMAP.items():
            counts[dst] += counts[src]
        counts = counts[:len(CLASSES)]
        tiles[name] = {
            "cx": (b["left"] + b["right"]) / 2,
            "cy": (b["bottom"] + b["top"]) / 2,
            "col": m["x_off"] // STEP_PX,
            "row": m["y_off"] // STEP_PX,
            "pool": pool,
            "overlap_zone": bool(in1 and in2),
            "counts": counts.astype(np.int64),
        }
    return tiles, scenes


def shares(names, tiles):
    tot = np.zeros(len(CLASSES), dtype=np.int64)
    for n in names:
        tot += tiles[n]["counts"]
    s = tot[STRATIFY_ON].sum()
    return (tot[STRATIFY_ON] / s) if s else tot[STRATIFY_ON].astype(float)


def landscape(names, tiles):
    """Coarse block label so allocation stratifies over landscape types."""
    sh = shares(names, tiles)
    by = dict(zip([CLASSES[i] for i in STRATIFY_ON], sh))
    if by["vegetation"] >= 0.50:
        return "vegetation"
    if by["urban"] + by["road"] >= 0.35:
        return "built-up"
    return "agriculture"


def build_blocks(names, tiles, block_size):
    x0 = min(tiles[n]["cx"] for n in names)
    y0 = min(tiles[n]["cy"] for n in names)
    blocks = defaultdict(list)
    for n in names:
        t = tiles[n]
        bid = (int((t["cx"] - x0) // block_size), int((t["cy"] - y0) // block_size))
        blocks[bid].append(n)
    return blocks


def allocate(blocks, tiles, seed):
    rng = random.Random(seed)
    strata = defaultdict(list)
    for bid, names in blocks.items():
        strata[landscape(names, tiles)].append(bid)
    assign = {}
    for _, bids in sorted(strata.items()):
        bids = sorted(bids)
        rng.shuffle(bids)
        n_tiles = sum(len(blocks[b]) for b in bids)
        targets = {s: f * n_tiles for s, f in FRACTIONS.items()}
        counts = {s: 0 for s in FRACTIONS}
        for b in bids:
            s = max(FRACTIONS, key=lambda k: targets[k] - counts[k])
            assign[b] = s
            counts[s] += len(blocks[b])
    return assign


def score(assign, blocks, tiles):
    """Worst per-class deviation from the pool-wide composition; None if a split
    loses a class."""
    overall = shares([n for ns in blocks.values() for n in ns], tiles)
    worst = 0.0
    for s in FRACTIONS:
        names = [n for b, ns in blocks.items() if assign[b] == s for n in ns]
        if not names:
            return None
        sh = shares(names, tiles)
        if sh.min() < MIN_CLASS_SHARE:
            return None
        worst = max(worst, float(np.abs(sh - overall).max()))
    return worst


def neighbours(name, tiles, index):
    t = tiles[name]
    out = []
    for dc in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if dc == dr == 0:
                continue
            other = index.get((t["col"] + dc, t["row"] + dr))
            if other:
                out.append(other)
    return out


def apply_buffer(split_of, tiles):
    """Carve a one-tile buffer between splits by dropping the lower-priority tile
    of every overlapping cross-split pair.

    Priority external > test > val > train. Training tiles are sacrificed first:
    they are the most redundant (and get augmented), whereas evaluation tiles are
    scarce and carry the statistical power of the study. Dropping either side of a
    pair removes the shared pixels equally well, so the choice is purely about
    which set can afford the loss.
    """
    priority = {"external": 3, "test": 2, "val": 1, "train": 0}
    index = {(t["col"], t["row"]): n for n, t in tiles.items() if n in split_of}
    drop = set()
    for name, s in split_of.items():
        for other in neighbours(name, tiles, index):
            o = split_of[other]
            if o == s:
                continue
            loser = name if priority[s] < priority[o] else other
            drop.add(loser)
    dropped = defaultdict(list)
    for n in drop:
        dropped[split_of[n]].append(n)
    kept = {n: s for n, s in split_of.items() if n not in drop}
    return kept, dropped


def audit(split_of, tiles):
    """Count evaluation tiles still in pixel contact with a training tile."""
    index = {(t["col"], t["row"]): n for n, t in tiles.items() if n in split_of}
    res = {}
    for s in ("val", "test", "external"):
        names = [n for n, v in split_of.items() if v == s]
        bad = [n for n in names
               if any(split_of[o] == "train" for o in neighbours(n, tiles, index))]
        res[s] = {"tiles": len(names), "touching_train": len(bad)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-size", type=float, default=2000.0)
    ap.add_argument("--seed-trials", type=int, default=500)
    ap.add_argument("--out-suffix", default="",
                    help="ต่อท้ายชื่อไฟล์ผล ใช้สำรวจขนาดบล็อกโดยไม่ทับของที่ใช้อยู่")
    args = ap.parse_args()

    tiles, scenes = load_tiles()
    pool_a = [n for n, t in tiles.items() if t["pool"] == "A"]
    pool_b = [n for n, t in tiles.items() if t["pool"] == "B"]
    n_overlap = sum(1 for n in pool_a if tiles[n]["overlap_zone"])
    print(f"labelled tiles: {len(tiles)}")
    print(f"  pool A (internal): {len(pool_a)}  "
          f"[scene1-only {len(pool_a) - n_overlap}, overlap zone {n_overlap}]")
    print(f"  pool B (external, scene2-only): {len(pool_b)}")

    blocks = build_blocks(pool_a, tiles, args.block_size)
    print(f"blocks: {len(blocks)} @ {args.block_size:.0f} m")

    best = None
    for seed_i in range(args.seed_trials):
        a = allocate(blocks, tiles, seed_i)
        s = score(a, blocks, tiles)
        if s is not None and (best is None or s < best[0]):
            best = (s, seed_i, a)
    if best is None:
        raise SystemExit("no allocation kept every class in every split; "
                         "try a different --block-size")
    dev, seed, assign = best
    print(f"chosen seed {seed}, worst class-share deviation {dev:.4f}")

    split_of = {}
    for bid, names in blocks.items():
        for n in names:
            split_of[n] = assign[bid]
    for n in pool_b:
        split_of[n] = "external"

    before = audit(split_of, tiles)
    split_of, dropped = apply_buffer(split_of, tiles)
    after = audit(split_of, tiles)

    print("\nbuffer (tiles dropped because they overlap another split):")
    for s in ("train", "val", "test", "external"):
        print(f"  {s}: -{len(dropped[s])}")
    print("\nleakage audit — evaluation tiles in pixel contact with a train tile:")
    for s in ("val", "test", "external"):
        b, a = before[s], after[s]
        print(f"  {s:9s} before {b['touching_train']:4d}/{b['tiles']:4d}"
              f"   after {a['touching_train']:4d}/{a['tiles']:4d}")

    out_splits = defaultdict(list)
    for n, s in split_of.items():
        out_splits[s].append(n)
    for s in out_splits:
        out_splits[s].sort()

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "site": "lamphun",
        "block_size_m": args.block_size,
        "seed": seed,
        "worst_class_share_deviation": dev,
        "tile_px": TILE_PX,
        "step_px": STEP_PX,
        "classes": CLASSES,
        "source_classes": SRC_CLASSES,
        "remap": {str(k): v for k, v in REMAP.items()},
        "scene_bounds_utm47n": {k: list(v) for k, v in scenes.items()},
        "pools": {
            "A_internal": len(pool_a),
            "A_overlap_zone": n_overlap,
            "B_external_scene2_only": len(pool_b),
        },
        "buffer_dropped": {k: sorted(v) for k, v in dropped.items()},
        "leakage_audit": {"before_buffer": before, "after_buffer": after},
        "splits": dict(out_splits),
        "blocks": {f"{b[0]}_{b[1]}": {"split": assign[b], "tiles": sorted(ns)}
                   for b, ns in sorted(blocks.items())},
        # Scene 2 sits outside pool A's block lattice, so it gets its own blocks on
        # the same 2 km grid. They define no split — every external tile is test —
        # but the block bootstrap needs a spatially valid resampling unit there too.
        "blocks_external": {
            f"x{b[0]}_{b[1]}": {"split": "external", "tiles": sorted(ns)}
            for b, ns in sorted(build_blocks(
                [n for n in pool_b if n in split_of], tiles, args.block_size).items())},
        "provenance": {
            "tiles_metadata": str(META.relative_to(ROOT)),
            "labels": str(LABELS.relative_to(ROOT)),
        },
    }
    (OUT / f"spatial_splits{args.out_suffix}.json").write_text(json.dumps(payload, indent=1))

    with open(OUT / f"spatial_splits_summary{args.out_suffix}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "n_tiles"] + [CLASSES[i] for i in STRATIFY_ON])
        for s in ("train", "val", "test", "external"):
            names = out_splits.get(s, [])
            if not names:
                continue
            w.writerow([s, len(names)] + [f"{v:.4f}" for v in shares(names, tiles)])
    print(f"\nwrote {OUT / f'spatial_splits{args.out_suffix}.json'}")
    print(f"wrote {OUT / f'spatial_splits_summary{args.out_suffix}.csv'}")


if __name__ == "__main__":
    main()
