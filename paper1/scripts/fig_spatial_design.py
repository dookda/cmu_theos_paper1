#!/usr/bin/env python3
"""Figure: the spatial-block split that replaces the random tile split.

This is the figure that has to convince a reviewer the protocol is sound, so it
shows the things a reader would otherwise have to take on trust: where the two
scene footprints actually overlap, which blocks went to which split, the buffer
of dropped tiles between splits, and the external scene held out whole.

Journal style per the project convention: English labels, black and white,
title inside the figure, no caption text.

Usage
-----
  python paper1/scripts/fig_spatial_design.py
  python paper1/scripts/fig_spatial_design.py --out /tmp/fig.png --dpi 600
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.lines import Line2D                   # noqa: E402
from matplotlib.patches import Rectangle              # noqa: E402

os.environ.pop("PROJ_DATA", None)
os.environ.pop("PROJ_LIB", None)

ROOT = Path(__file__).resolve().parent.parent.parent
SPLITS = ROOT / "paper1" / "analysis" / "results" / "spatial_splits.json"
META = ROOT / "paper2" / "analysis" / "results" / "tiles_metadata.json"
OUT = ROOT / "paper1" / "analysis" / "figure" / "fig_spatial_design.png"

# Greyscale + hatching so the panel survives black-and-white printing.
STYLE = {
    "train":         {"fc": "0.80", "hatch": "",     "label": "Train"},
    "val":           {"fc": "0.55", "hatch": "///",  "label": "Validation"},
    "test":          {"fc": "0.30", "hatch": "",     "label": "Test"},
    "external_near": {"fc": "0.05", "hatch": "",     "label": "External, near"},
    "external_far":  {"fc": "0.05", "hatch": "xxx",  "label": "External, far"},
}
DROPPED = {"fc": "white", "ec": "0.6", "label": "Dropped as buffer"}
DIST = Path(__file__).resolve().parent.parent / "analysis" / "results" / \
    "external_distance.json"


def load():
    sp = json.loads(SPLITS.read_text())
    meta = json.loads(META.read_text())
    meta = meta["tiles"] if isinstance(meta, dict) and "tiles" in meta else meta
    bounds = {Path(m["filename"]).stem: m["bounds"] for m in meta}
    split_of = {t: s for s, tiles in sp["splits"].items() for t in tiles}
    # แยก external ตามระยะถึง train — Scene 2 ถูก label เป็นหย่อม ๆ บางหย่อมชิดพื้นที่
    # train บางหย่อมห่างเกือบ 10 กม. รวมเป็นก้อนเดียวจะกลบผลที่น่าสนใจที่สุด
    if DIST.exists():
        dist = json.loads(DIST.read_text())["splits"]["external"]
        for t in dist["near_tiles"]:
            split_of[t] = "external_near"
        for t in dist["far_tiles"]:
            split_of[t] = "external_far"
    dropped = {t for v in sp["buffer_dropped"].values() for t in v}
    return sp, bounds, split_of, dropped


def draw_tiles(ax, bounds, split_of, dropped):
    for name, b in bounds.items():
        s = split_of.get(name)
        if s is None and name not in dropped:
            continue
        w, h = b["right"] - b["left"], b["top"] - b["bottom"]
        if s is None:
            ax.add_patch(Rectangle((b["left"], b["bottom"]), w, h,
                                   facecolor=DROPPED["fc"], edgecolor=DROPPED["ec"],
                                   linewidth=0.15, zorder=2))
        else:
            st = STYLE[s]
            ax.add_patch(Rectangle((b["left"], b["bottom"]), w, h,
                                   facecolor=st["fc"], edgecolor="none",
                                   hatch=st["hatch"] or None, linewidth=0.0, zorder=1))


def draw_blocks(ax, sp, bounds):
    for key in ("blocks", "blocks_external"):
        for info in sp.get(key, {}).values():
            tiles = [bounds[t] for t in info["tiles"] if t in bounds]
            if not tiles:
                continue
            x0 = min(b["left"] for b in tiles)
            x1 = max(b["right"] for b in tiles)
            y0 = min(b["bottom"] for b in tiles)
            y1 = max(b["top"] for b in tiles)
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   edgecolor="black", linewidth=0.5, zorder=3))


def draw_scenes(ax, sp):
    for name, (l, b, r, t) in sp["scene_bounds_utm47n"].items():
        ax.add_patch(Rectangle((l, b), r - l, t - b, fill=False, edgecolor="black",
                               linewidth=1.3, linestyle=(0, (6, 3)), zorder=4))
        ax.text(l + 200, t - 700, "Scene 1" if name == "scene1" else "Scene 2",
                fontsize=8, style="italic", zorder=5)


def scale_bar(ax, x, y, length=2000):
    ax.plot([x, x + length], [y, y], color="black", linewidth=2.5,
            solid_capstyle="butt", zorder=6)
    for xi in (x, x + length):
        ax.plot([xi, xi], [y - 180, y + 180], color="black", linewidth=1.2, zorder=6)
    ax.text(x + length / 2, y + 320, f"{length / 1000:.0f} km", ha="center",
            fontsize=7.5, zorder=6)


def north_arrow(ax, x, y, size=1200):
    ax.annotate("", xy=(x, y + size), xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>", color="black", linewidth=1.3),
                zorder=6)
    ax.text(x, y + size + 250, "N", ha="center", fontsize=9, fontweight="bold",
            zorder=6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--dpi", type=int, default=400)
    args = ap.parse_args()

    sp, bounds, split_of, dropped = load()
    fig, ax = plt.subplots(figsize=(7.0, 8.6))

    draw_tiles(ax, bounds, split_of, dropped)
    draw_blocks(ax, sp, bounds)
    draw_scenes(ax, sp)

    used = [bounds[t] for t in list(split_of) + list(dropped) if t in bounds]
    x0 = min(b["left"] for b in used) - 600
    x1 = max(b["right"] for b in used) + 600
    y0 = min(b["bottom"] for b in used) - 900
    y1 = max(b["top"] for b in used) + 900
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (m, UTM 47N)", fontsize=8)
    ax.set_ylabel("Northing (m, UTM 47N)", fontsize=8)
    ax.tick_params(labelsize=7)

    n = {}
    for t, s in split_of.items():
        n[s] = n.get(s, 0) + 1
    ax.set_title(
        f"Spatial-block split of the THEOS-2 Lamphun tiles\n"
        f"{sp['block_size_m']:.0f} m blocks, one-tile buffer between splits; "
        f"no evaluation tile shares pixels with training data",
        fontsize=10, fontweight="bold", pad=12)

    handles = [Rectangle((0, 0), 1, 1, facecolor=STYLE[s]["fc"],
                         hatch=STYLE[s]["hatch"] or None, edgecolor="none",
                         label=f"{STYLE[s]['label']} (n={n.get(s, 0)})")
               for s in ("train", "val", "test", "external_near", "external_far")]
    handles.append(Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="0.6",
                             label=f"{DROPPED['label']} (n={len(dropped)})"))
    handles.append(Line2D([0], [0], color="black", linewidth=1.3,
                          linestyle=(0, (6, 3)), label="Scene footprint"))
    handles.append(Line2D([0], [0], color="black", linewidth=0.5,
                          label=f"{sp['block_size_m']:.0f} m block"))
    # ใต้แผนที่ ไม่ทับข้อมูล — แผนที่มีที่ว่างตรงกลางอยู่แล้วจากช่องโหว่ของการ label
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.09),
              ncol=3, fontsize=7, framealpha=1.0, edgecolor="black", borderpad=0.6)

    scale_bar(ax, x0 + 900, y0 + 1400)
    north_arrow(ax, x1 - 1300, y0 + 1200)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {out}")
    print(f"  tiles drawn: {len(split_of)} assigned + {len(dropped)} buffer")
    print(f"  blocks: {len(sp['blocks'])} internal + {len(sp['blocks_external'])} external")


if __name__ == "__main__":
    main()
