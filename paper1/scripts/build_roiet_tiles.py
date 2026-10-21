#!/usr/bin/env python3
"""Cut Roi Et into Paper 1 tiles for the cross-region probe (addendum C).

See `paper1/configs/addendum_prereg_2026-10.md` section C. Two input variants, each a
dataset tree that `eval_rerun.py` reads unchanged:

  raw     Roi Et DN rendered with the Lamphun tile transform as is
  zscene  Roi Et DN first matched per band to the mean/SD of the Lamphun training
          tiles, then rendered with the same transform

Lamphun tile transform (recovered from the training tiles by regression, r = 1.000):
RGB from theos/lamphoon/render.qml (R 100-541, G 116-395, B 164-404); NIR global
p2/p98 stretch 83-790 from make_nir_tiles.py. Bands are R, G, B, NIR in both scenes
(Roi Et vegetation NDVI +0.31 with band 4 as NIR).

Labels: the Roi Et label raster codes unlabelled pixels as 0, the same as background, so
the labelled area is taken from the footprints of the 445 server masks. Pixels outside
it, and pixels with DN 0 or 65535 in any band, are written as 255 and not scored. Tiles
with < 50% labelled pixels are dropped.

The split file names the set `external` so `eval_rerun.py --split external` scores it
per 1,500 m block with no code change.

Usage
-----
  python paper1/scripts/build_roiet_tiles.py --out /path/to/roiet_p1
"""
import argparse
import json
import os
from pathlib import Path

for _k in ("PROJ_DATA", "PROJ_LIB", "GDAL_DATA", "GDAL_DRIVER_PATH"):
    os.environ.pop(_k, None)

import cv2
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform as warp_xy
from rasterio.windows import Window, from_bounds

ROOT = Path(__file__).resolve().parent.parent.parent
RES = ROOT / "paper1" / "analysis" / "results"
RE = ROOT / "paper2" / "data" / "roi_et"
IMG = RE / "roiet_utm48n_0p5m.tif"
LAB = RE / "roiet_labels_utm48n_0p5m.tif"
MASKS = RE / "labels_server" / "semantic"
META = RE / "tiles_metadata.json"
LAMPHUN_META = ROOT / "paper2" / "analysis" / "results" / "tiles_metadata.json"
LAMPHUN_SCENES = [ROOT / "theos" / "lamphoon" / f
                  for f in ("th2_20260108_034820_psp_gcs_ort_47n.tif",
                            "th2_20260108_034822_psp_gcs_ort_47n.tif")]

STRETCH = [(100, 541), (116, 395), (164, 404), (83, 790)]  # R, G, B, NIR
TILE, STEP, BLOCK_M = 512, 448, 1500.0
CRS = "EPSG:32648"


def render(dn, lo, hi):
    return np.clip((dn.astype(np.float32) - lo) / (hi - lo + 1e-8) * 255, 0, 255).astype(np.uint8)


def footprint(shape, transform):
    shapes = []
    meta = {m["filename"]: m for m in json.loads(META.read_text())}
    names = sorted(p.name for p in MASKS.glob("*.png"))
    for n in names:
        b = meta[n]["bounds"]
        xs, ys = warp_xy("EPSG:4326", CRS,
                         [b["left"], b["right"], b["right"], b["left"]],
                         [b["bottom"], b["bottom"], b["top"], b["top"]])
        shapes.append(({"type": "Polygon",
                        "coordinates": [list(zip(xs, ys)) + [(xs[0], ys[0])]]}, 1))
    return rasterize(shapes, out_shape=shape, transform=transform, dtype="uint8"), len(names)


def valid_dn(a):
    return (a > 0).all(0) & (a < 65535).all(0)


def lamphun_moments():
    """Per-band mean/SD over the spatial-split training tiles (every 4th pixel)."""
    train = set(json.loads((RES / "spatial_splits.json").read_text())["splits"]["train"])
    meta = [m for m in json.loads(LAMPHUN_META.read_text())
            if Path(m["filename"]).stem in train]
    scenes = [rasterio.open(p) for p in LAMPHUN_SCENES]
    s = np.zeros(4); ss = np.zeros(4); n = 0
    for m in meta:
        b = m["bounds"]
        for sc in scenes:
            w = from_bounds(b["left"], b["bottom"], b["right"], b["top"], transform=sc.transform)
            a = sc.read(window=w, boundless=True, fill_value=0,
                        out_shape=(4, TILE, TILE))[:, ::4, ::4].reshape(4, -1).astype(np.float64)
            ok = valid_dn(a)
            if ok.mean() > 0.99:
                break
        a = a[:, ok]
        s += a.sum(1); ss += (a ** 2).sum(1); n += a.shape[1]
    mu = s / n
    return mu, np.sqrt(ss / n - mu ** 2), len(meta)


def roiet_moments(src):
    s = np.zeros(4); ss = np.zeros(4); n = 0
    for y in range(0, src.height, 2048):
        a = src.read(window=Window(0, y, src.width, min(2048, src.height - y)))
        a = a[:, ::4, ::4].reshape(4, -1).astype(np.float64)
        a = a[:, valid_dn(a)]
        s += a.sum(1); ss += (a ** 2).sum(1); n += a.shape[1]
    mu = s / n
    return mu, np.sqrt(ss / n - mu ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    src, lab = rasterio.open(IMG), rasterio.open(LAB)
    assert src.transform == lab.transform and src.shape == lab.shape
    fp, n_masks = footprint(src.shape, src.transform)

    mu_l, sd_l, n_train = lamphun_moments()
    mu_r, sd_r = roiet_moments(src)
    print("Lamphun train mean", mu_l.round(1), "sd", sd_l.round(1), f"({n_train} tiles)")
    print("Roi Et   mean", mu_r.round(1), "sd", sd_r.round(1))

    variants = ("raw", "zscene")
    for v in variants:
        (args.out / v / "tiles").mkdir(parents=True, exist_ok=True)
        (args.out / v / "semantic_labels").mkdir(parents=True, exist_ok=True)

    names, centers, labelled = [], {}, []
    for y in range(0, src.height - TILE + 1, STEP):
        for x in range(0, src.width - TILE + 1, STEP):
            w = Window(x, y, TILE, TILE)
            f = fp[y:y + TILE, x:x + TILE]
            if f.mean() < 0.5:
                continue
            a = src.read(window=w)
            ok = valid_dn(a) & (f == 1)
            if ok.mean() < 0.5:
                continue
            lb = lab.read(1, window=w)
            lb = np.where(ok, lb, 255).astype(np.uint8)
            name = f"roiet_{y // STEP:03d}_{x // STEP:03d}"
            for v in variants:
                d = a.astype(np.float64)
                if v == "zscene":
                    d = (d - mu_r[:, None, None]) / sd_r[:, None, None] * sd_l[:, None, None] \
                        + mu_l[:, None, None]
                rgb = np.dstack([render(d[i], *STRETCH[i]) for i in range(3)])
                cv2.imwrite(str(args.out / v / "tiles" / f"{name}.png"), rgb[:, :, ::-1])
                cv2.imwrite(str(args.out / v / "tiles" / f"{name}_nir.png"),
                            render(d[3], *STRETCH[3]))
                cv2.imwrite(str(args.out / v / "semantic_labels" / f"{name}.png"), lb)
            names.append(name)
            labelled.append(float(ok.mean()))
            cx, cy = src.transform * (x + TILE / 2, y + TILE / 2)
            centers[name] = (cx, cy)

    blocks = {}
    for n, (cx, cy) in centers.items():
        k = f"r{int(cx // BLOCK_M)}_{int(cy // BLOCK_M)}"
        blocks.setdefault(k, {"split": "external", "tiles": []})["tiles"].append(n)
    spatial = json.loads((RES / "spatial_splits.json").read_text())
    payload = {
        "design": "Roi Et cross-region probe (addendum C); set named 'external' for eval_rerun.py",
        "classes": spatial["classes"], "remap": spatial["remap"],
        "tile_px": TILE, "step_px": STEP, "block_size_m": BLOCK_M, "crs": CRS,
        "n_server_masks": n_masks, "min_labelled_fraction": 0.5,
        "labelled_fraction": {"mean": float(np.mean(labelled)), "min": float(np.min(labelled))},
        "stretch_rgbn": STRETCH,
        "zscene": {"lamphun_train_mean": mu_l.tolist(), "lamphun_train_sd": sd_l.tolist(),
                   "roiet_mean": mu_r.tolist(), "roiet_sd": sd_r.tolist()},
        "splits": {"external": sorted(names)},
        "blocks_external": blocks,
    }
    (RES / "roiet_splits.json").write_text(json.dumps(payload, indent=1))
    print(f"{len(names)} tiles, {len(blocks)} blocks -> {args.out}, {RES / 'roiet_splits.json'}")


if __name__ == "__main__":
    main()
