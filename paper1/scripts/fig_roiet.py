#!/usr/bin/env python3
"""Roi Et transfer probe figure (addendum C).

(a) Thailand with Lamphun (training area) and Roi Et (transfer probe)
(b) Roi Et scene, rendered after per-band moment matching to the Lamphun training tiles
    and the Lamphun stretch; labelled tiles and the 1,500 m evaluation blocks outlined,
    example tiles lettered
(c) Four example tiles chosen by a fixed rule (largest labelled share of urban,
    agriculture, vegetation, water): input as delivered (raw DN, Lamphun stretch), input
    after moment matching, reference label, and the six-class prediction of the locked
    ensemble on each input. Percentages: agreement with the reference on labelled
    non-background pixels.

Inputs:  paper2/data/roi_et/roiet_utm48n_0p5m.tif, analysis/results/roiet_splits.json,
         analysis/results/roiet_examples.npz (predict_roiet_examples.py on Erawan),
         analysis/shapes/external/geoBoundaries-THA-ADM1_simplified.geojson
Output:  analysis/figure/fig_roiet.png

Needs geopandas:  ~/miniforge3/bin/python paper1/scripts/fig_roiet.py
"""
import json
import os
import sys
from pathlib import Path

for _k in ("PROJ_DATA", "PROJ_LIB", "GDAL_DATA", "GDAL_DRIVER_PATH"):
    os.environ.pop(_k, None)

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe              # noqa: E402
import matplotlib.pyplot as plt                  # noqa: E402
import numpy as np                               # noqa: E402
import rasterio                                  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from palette import CLASS as CLASS_COLOR         # noqa: E402

RES = ROOT / "analysis" / "results"
IMG = ROOT.parent / "paper2" / "data" / "roi_et" / "roiet_utm48n_0p5m.tif"
THA = ROOT / "analysis" / "shapes" / "external" / "geoBoundaries-THA-ADM1_simplified.geojson"
OUT = ROOT / "analysis" / "figure" / "fig_roiet.png"
CLASSES = ["background", "vegetation", "water", "urban", "agriculture", "bare_soil", "road"]
HALO = [pe.withStroke(linewidth=2.2, foreground="white")]
STEP, TILE = 448, 512


def lut():
    t = np.full((256, 3), 1.0)                   # unlabelled (255): white, hatched on top
    for i, c in enumerate(CLASSES):
        h = CLASS_COLOR[c].lstrip("#")
        t[i] = [int(h[j:j + 2], 16) / 255 for j in (0, 2, 4)]
    return t


def scene_thumb(sp, ds=16):
    z = sp["zscene"]
    st = sp["stretch_rgbn"]
    with rasterio.open(IMG) as s:
        h, w = s.height // ds, s.width // ds
        a = s.read([1, 2, 3], out_shape=(3, h, w),
                   resampling=rasterio.enums.Resampling.nearest).astype(float)
        tr = s.transform
    bad = (a <= 0).any(0) | (a >= 65535).any(0)
    rgb = np.zeros((h, w, 3))
    for i in range(3):
        v = (a[i] - z["roiet_mean"][i]) / z["roiet_sd"][i] * z["lamphun_train_sd"][i] \
            + z["lamphun_train_mean"][i]
        rgb[..., i] = np.clip((v - st[i][0]) / (st[i][1] - st[i][0]), 0, 1)
    rgba = np.dstack([rgb, (~bad).astype(float)])
    ext = (tr.c, tr.c + tr.a * w * ds, tr.f + tr.e * h * ds, tr.f)
    return rgba, ext, tr


def tile_xy(name, tr):
    r, c = (int(x) for x in name.split("_")[1:3])
    x0, y0 = tr * (c * STEP, r * STEP)
    return x0, y0 - TILE * 0.5                   # lower-left corner (north-up raster)


def main():
    sp = json.loads((RES / "roiet_splits.json").read_text())
    ex = np.load(RES / "roiet_examples.npz")
    names = [str(n) for n in ex["names"]]
    L = lut()

    fig = plt.figure(figsize=(7.16, 9.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 2.05], hspace=0.10)
    top = gs[0].subgridspec(1, 2, width_ratios=[0.62, 1.38], wspace=0.10)
    ax_a, ax_b = fig.add_subplot(top[0]), fig.add_subplot(top[1])

    # (a) Thailand
    tha = gpd.read_file(THA)
    tha.plot(ax=ax_a, facecolor="#e6e6e6", edgecolor="#9a9a9a", lw=0.25)
    for key, col, lab, dx in (("Lamphun", "#c0392b", "Lamphun\n(training)", 2.3),
                              ("Roi Et", "#0072b2", "Roi Et\n(transfer probe)", 0.6)):
        g = tha[tha.shapeName.str.contains(key)]
        g.plot(ax=ax_a, facecolor=col, edgecolor="black", lw=0.4)
        c = g.geometry.iloc[0].centroid
        ax_a.annotate(lab, xy=(c.x, c.y), xytext=(c.x + dx, c.y + 1.4), fontsize=6.5,
                      ha="left", path_effects=HALO, arrowprops=dict(arrowstyle="-", lw=0.6))
    ax_a.set_xlim(97.2, 105.8); ax_a.set_ylim(5.5, 20.6); ax_a.set_aspect("equal")
    ax_a.set_xticks([98, 100, 102, 104]); ax_a.set_yticks([8, 12, 16, 20])
    ax_a.set_xticklabels([f"{v}°E" for v in [98, 100, 102, 104]], fontsize=6)
    ax_a.set_yticklabels([f"{v}°N" for v in [8, 12, 16, 20]], fontsize=6)
    ax_a.tick_params(length=2, pad=1.5)
    ax_a.set_title("(a) Thailand", fontsize=8, loc="left")

    # (b) Roi Et scene with labelled tiles, blocks and examples
    rgba, ext, tr = scene_thumb(sp)
    ax_b.imshow(rgba, extent=ext, interpolation="bilinear")
    for n in sp["splits"]["external"]:
        x0, y0 = tile_xy(n, tr)
        ax_b.add_patch(Rectangle((x0, y0), TILE * 0.5, TILE * 0.5, fill=False,
                                 ec="#ffd23f", lw=0.25))
    bm = sp["block_size_m"]
    for k in sp["blocks_external"]:
        bx, by = (int(v) for v in k[1:].split("_"))
        ax_b.add_patch(Rectangle((bx * bm, by * bm), bm, bm, fill=False, ec="white",
                                 lw=0.9, ls=(0, (3, 2))))
    for i, n in enumerate(names):
        x0, y0 = tile_xy(n, tr)
        ax_b.add_patch(Rectangle((x0, y0), TILE * 0.5, TILE * 0.5, fill=False,
                                 ec="#d55e00", lw=1.4))
        ax_b.text(x0 + 300, y0 + 300, "abcd"[i], fontsize=8, fontweight="bold",
                  color="#d55e00", path_effects=HALO)
    ax_b.set_xlim(ext[0], ext[1]); ax_b.set_ylim(ext[2], ext[3]); ax_b.set_aspect("equal")
    xt = np.arange(np.ceil(ext[0] / 2000) * 2000, ext[1], 2000)
    yt = np.arange(np.ceil(ext[2] / 2000) * 2000, ext[3], 2000)
    ax_b.set_xticks(xt); ax_b.set_yticks(yt)
    ax_b.set_xticklabels([f"{v / 1000:.0f}" for v in xt], fontsize=6)
    ax_b.set_yticklabels([f"{v / 1000:.0f}" for v in yt], fontsize=6)
    ax_b.set_xlabel("Easting, UTM 48N (km)", fontsize=6.5, labelpad=1)
    ax_b.set_ylabel("Northing (km)", fontsize=6.5, labelpad=1)
    ax_b.tick_params(length=2, pad=1.5)
    ax_b.set_title("(b) Roi Et, 13 Jan 2025, moment matched: labelled tiles (yellow),\n"
                   "1.5 km evaluation blocks (dashed), example tiles a–d", fontsize=7, loc="left")

    # (c) example tiles
    cols = ["Input as delivered", "After moment matching", "Reference",
            "Prediction, as delivered", "Prediction, moment matched"]
    sub = gs[1].subgridspec(len(names), 5, wspace=0.04, hspace=0.06)
    lab = ex["labels"]
    for i in range(len(names)):
        ok = (lab[i] > 0) & (lab[i] < 7)
        panels = [ex["rgb_raw"][i] / 255, ex["rgb_zscene"][i] / 255, L[lab[i]],
                  L[ex["pred_raw"][i]], L[ex["pred_zscene"][i]]]
        for j, im in enumerate(panels):
            ax = fig.add_subplot(sub[i, j])
            ax.imshow(im, interpolation="nearest")
            if j == 2:
                un = (lab[i] == 255).astype(float)
                if un.any():
                    ax.contourf(un, levels=[0.5, 1.5], colors="none", hatches=["////"])
            ax.set_xticks([]); ax.set_yticks([])
            if j >= 3:
                p = ex["pred_raw" if j == 3 else "pred_zscene"][i]
                agr = 100 * (p[ok] == lab[i][ok]).mean() if ok.any() else float("nan")
                ax.text(8, 40, f"{agr:.0f}%", fontsize=7, path_effects=HALO)
            if i == 0:
                ax.set_title(cols[j], fontsize=6.5)
            if j == 0:
                ax.set_ylabel(f"({'abcd'[i]}) {str(ex['picked_for'][i])}", fontsize=7)
    handles = [Patch(facecolor=CLASS_COLOR[c], edgecolor="black", lw=0.4,
                     label=c.replace("_", " ")) for c in CLASSES]
    handles.append(Patch(facecolor="white", edgecolor="black", lw=0.4, hatch="////",
                         label="unlabelled"))
    fig.legend(handles=handles, loc="lower center", ncol=8, fontsize=6.3, frameon=False,
               bbox_to_anchor=(0.5, 0.005), handlelength=1.2, columnspacing=0.9)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.955, bottom=0.04)
    fig.savefig(OUT, dpi=600)
    print("wrote", OUT, "tiles:", dict(zip(ex["picked_for"], names)))


if __name__ == "__main__":
    main()
