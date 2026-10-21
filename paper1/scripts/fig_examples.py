#!/usr/bin/env python3
"""ตัวอย่างไทล์จริงจากชุดทดสอบ: ภาพ THEOS-2, label อ้างอิง, ผล ensemble และจุดที่ไม่ตรงกัน

ตัวเลขใน paper บอกขนาดของผล แต่ผู้อ่านไม่เห็นว่าภาพ 0.5 ม. หน้าตาอย่างไร
และความคลาดเคลื่อนเกิดตรงไหน รูปนี้ให้เห็นทั้งสองอย่างบนพื้นที่ที่โมเดลไม่เคยเห็น

การเลือกไทล์เป็นกฎตายตัว ไม่ได้เลือกด้วยตา: ต่อหนึ่งคลาสหลัก เลือกไทล์ชุด test
ที่คลาสนั้นมีสัดส่วนใน label มากที่สุด (เฉพาะไทล์ที่มีข้อมูลภาพ >= 99%)
จึงรวมทั้งไทล์ที่โมเดลทำได้ดีและไม่ดี

แหล่งข้อมูล
  theos/lamphoon/lamphoon_merged.tif        ภาพ (แบนด์ 1-3 = R,G,B; stretch ตาม render.qml)
  paper2/data/labels_fixed/tile_*.png        label อ้างอิง (7 = grass ยุบเป็น background)
  paper2/analysis/results/tiles_metadata.json  ขอบเขตของไทล์
  analysis/results/spatial_splits.json       รายชื่อไทล์ชุด test
  outputs/map_ensemble_lamphun.tif           ผล ensemble ที่เลือกจาก validation (scene 1)

ใช้งาน
------
  python paper1/scripts/fig_examples.py
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe                # noqa: E402
import matplotlib.pyplot as plt                    # noqa: E402
import numpy as np                                 # noqa: E402
import rasterio                                    # noqa: E402
from matplotlib.patches import Patch               # noqa: E402
from PIL import Image                              # noqa: E402
from rasterio.windows import from_bounds           # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palette import CLASS as CLASS_COLOR           # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parent
IMG = REPO / "theos" / "lamphoon" / "lamphoon_merged.tif"
LABELS = REPO / "paper2" / "data" / "labels_fixed"
META = REPO / "paper2" / "analysis" / "results" / "tiles_metadata.json"
SPLITS = ROOT / "analysis" / "results" / "spatial_splits.json"
MAP = ROOT / "outputs" / "map_ensemble_lamphun.tif"
OUT = ROOT / "analysis" / "figure" / "fig_examples.png"

CLASSES = ["background", "vegetation", "water", "urban", "agriculture",
           "bare_soil", "road"]
MAP2 = ROOT / "outputs" / "map_ensemble_lamphun_scene2.tif"
# แผนที่หกคลาส (ไม่ทำนาย background) — ถ้ามีครบสองฉากจะใช้ชุดนี้ ให้ตรงกับแผนที่หลักของบทความ
MAP6 = ROOT / "outputs" / "map_ensemble6_lamphun.tif"
MAP62 = ROOT / "outputs" / "map_ensemble6_lamphun_scene2.tif"
SHOW = ["urban", "agriculture", "vegetation", "water", "bare_soil"]
# เมื่อมีแผนที่ฉากที่ 2 แสดงทั้งสองชุดที่กันไว้ คลาสละหนึ่งไทล์ต่อชุด น้ำและดินเปล่ามาจากชุดทดสอบภายใน
# เพราะชุดภายนอกมีสองคลาสนี้น้อยมาก (ดินเปล่า 0.28% ของพิกเซล)
GROUPS = [("test", "Scene 1 test", ["urban", "water", "bare_soil"]),
          ("external", "Scene 2 held out", ["urban", "agriculture", "vegetation"])]
# stretch ของ render.qml ชุดเดียวกับไทล์ที่ใช้ฝึก
STRETCH = [(100, 541), (116, 395), (164, 404)]
MISMATCH = "#d6006e"

# ภาพ THEOS-2 ในรูปอ่านจากไฟล์ภาพย่อยจริงที่ใช้ฝึกและประเมินแบบจำลอง
# (cmu_theos_ai_data_fixed/tiles บน Erawan สำเนาไว้ที่ analysis/figure/source_tiles)
# ไม่ได้สร้างหรือแต่งภาพขึ้นใหม่ ถ้าไม่มีไฟล์ สคริปต์หยุดแทนที่จะใช้ภาพจากแหล่งอื่น
SOURCE_TILES = ROOT / "analysis" / "figure" / "source_tiles"


def tile_rgb(tid):
    p = SOURCE_TILES / f"{tid}.png"
    if not p.exists():
        raise SystemExit(f"ไม่พบไฟล์ภาพย่อยจริง {p} — คัดลอกจาก "
                         f"/project/gisnorth/cmu_theos_ai_data_fixed/tiles/{tid}.png ก่อน")
    return np.asarray(Image.open(p).convert("RGB")).astype(float) / 255



def lut():
    t = np.ones((256, 3))
    for i, c in enumerate(CLASSES):
        h = CLASS_COLOR[c]
        t[i] = [int(h[k:k + 2], 16) / 255 for k in (1, 3, 5)]
    return t


def load_label(name):
    a = np.array(Image.open(LABELS / name)).astype(np.uint8)
    a[a == 7] = 0
    return a


def pick_tiles(test, meta, map_bounds, show=None):
    """ไทล์ test ที่ class นั้นมีสัดส่วนมากที่สุด ไม่ซ้ำไทล์กัน"""
    l, b, r, t = map_bounds
    cand = []
    for tid in test:
        m = meta.get(tid + ".png")
        if not m or m["valid_ratio"] < 0.99:
            continue
        bb = m["bounds"]
        if bb["left"] < l or bb["right"] > r or bb["bottom"] < b or bb["top"] > t:
            continue
        lab = load_label(tid + ".png")
        share = np.bincount(lab.ravel(), minlength=len(CLASSES)) / lab.size
        cand.append((tid, share))
    used, out = set(), []
    for c in (show or SHOW):
        k = CLASSES.index(c)
        for tid, share in sorted(cand, key=lambda x: -x[1][k]):
            if tid not in used:
                used.add(tid)
                out.append((c, tid, share[k]))
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    meta = {m["filename"]: m for m in json.loads(META.read_text())}
    splits = json.loads(SPLITS.read_text())["splits"]
    global MAP, MAP2
    six = MAP6.exists() and MAP62.exists()
    if six:
        MAP, MAP2 = MAP6, MAP62
    two = MAP2.exists()
    picks = []          # (class, tile, share, split, scene label, map path)
    if two:
        for split, lab_, show in GROUPS:
            mpath = MAP if split == "test" else MAP2
            with rasterio.open(mpath) as mp:
                picks += [(c, tid, sh, split, lab_, mpath)
                          for c, tid, sh in pick_tiles(splits[split], meta, tuple(mp.bounds), show)]
    else:
        with rasterio.open(MAP) as mp:
            picks = [(c, tid, sh, "test", "Scene 1 test", MAP)
                     for c, tid, sh in pick_tiles(splits["test"], meta, tuple(mp.bounds))]
    if not picks:
        return print("ข้าม: ไม่พบไทล์ที่อยู่ในขอบเขตแผนที่")

    colors = lut()
    n = len(picks)
    fig, axes = plt.subplots(n, 4, figsize=(7.2, (1.72 if two else 1.85) * n + 0.6))
    heads = ["THEOS-2 image (R, G, B)", "Reference label",
             "Ensemble prediction" + (" (six classes)" if six else ""), "Disagreement"]
    rows = []
    with rasterio.open(MAP) as mp1, rasterio.open(MAP2 if two else MAP) as mp2:
        for i, (cls, tid, share, split, scene, mpath) in enumerate(picks):
            mp = mp1 if mpath == MAP else mp2
            bb = meta[tid + ".png"]["bounds"]
            rgb = tile_rgb(tid)
            pw = from_bounds(bb["left"], bb["bottom"], bb["right"], bb["top"],
                             mp.transform).round_offsets().round_lengths()
            pred = np.asarray(mp.read(1, window=pw))
            lab = load_label(tid + ".png")
            hh, ww = (min(x, y) for x, y in zip(lab.shape, pred.shape))
            lab, pred, rgb = lab[:hh, :ww], pred[:hh, :ww], rgb[:hh, :ww]
            valid = pred != 255
            # แผนที่หกคลาสไม่มี background ให้เทียบ: คิดความสอดคล้องเฉพาะพิกเซลที่ label เป็นคลาสจริง
            scored = valid & (lab != 0) if six else valid
            agree = (lab == pred)[scored].mean()
            rows.append({"tile": tid, "split": split, "class": cls,
                         "class_share": round(float(share), 4),
                         "pixel_agreement": round(float(agree), 4)})

            pred_v = np.where(valid, pred, 0)
            dis = np.ones((hh, ww, 3)) * 0.92
            g = rgb.mean(axis=2, keepdims=True) * 0.55 + 0.4
            dis[:] = g
            mis = (lab != pred) & scored
            dis[mis] = [int(MISMATCH[k:k + 2], 16) / 255 for k in (1, 3, 5)]
            ref = colors[lab]
            if six:
                # พื้นที่ที่ไม่มี label ไม่ใช่คลาส: แสดงภาพจริงแบบจางแทนการระบายสี แล้วทับลายขีด
                ref = np.where((lab == 0)[..., None], np.broadcast_to(g, ref.shape), ref)
            panels = [rgb, ref, colors[pred_v], dis]
            for j, im in enumerate(panels):
                ax = axes[i, j]
                ax.imshow(im, interpolation="nearest")
                if six and j == 1 and (lab == 0).any():
                    matplotlib.rcParams["hatch.linewidth"] = 0.35
                    ax.contourf((lab == 0).astype(float), levels=[0.5, 1.5], colors="none",
                                hatches=["////"])
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_linewidth(0.6)
                if i == 0:
                    ax.set_title(heads[j], fontsize=8)
            axes[i, 0].set_ylabel((f"({chr(97 + i)}) {scene}\n" if two else f"({chr(97 + i)}) ")
                                  + f"{cls.replace('_', ' ')}, {100 * share:.0f}% of tile",
                                  fontsize=7 if two else 7.5)
            axes[i, 3].text(0.97, 0.04, f"agree {100 * agree:.0f}%" + (" of labelled" if six else ""),
                            transform=axes[i, 3].transAxes, ha="right", va="bottom",
                            fontsize=7, bbox=dict(fc="white", ec="none", pad=1.2))
            # มาตราส่วน 50 ม. = 100 px
            halo = [pe.withStroke(linewidth=2.2, foreground="white")]
            axes[i, 0].plot([12, 112], [hh - 18, hh - 18], color="white", lw=2.2,
                            path_effects=[pe.withStroke(linewidth=3.6, foreground="black")])
            axes[i, 0].text(62, hh - 26, "50 m", color="black", ha="center",
                            va="bottom", fontsize=6.5, path_effects=halo)

    handles = [Patch(facecolor=CLASS_COLOR[c], edgecolor="black", lw=0.4,
                     label=c.replace("_", " ")) for c in CLASSES if not (six and c == "background")]
    if six:
        handles.insert(0, Patch(facecolor="0.85", edgecolor="0.25", lw=0.4, hatch="////",
                                label="no reference label"))
    handles.append(Patch(facecolor=MISMATCH, edgecolor="black", lw=0.4,
                         label="label ≠ prediction"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.08, right=0.995, top=0.965, bottom=0.075,
                        wspace=0.03, hspace=0.05)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=a.dpi, facecolor="white")
    rec = ROOT / "analysis" / "results" / "fig_examples_tiles.json"
    rec.write_text(json.dumps({"rule": "per held-out set and class, the tile with the largest "
                               "reference share of that class (valid_ratio >= 0.99, inside "
                               "the ensemble map); water and bare soil are drawn from the "
                               "internal test split because the held-out scene has very "
                               "little of either",
                               "prediction": ("six-class ensemble map (background excluded from "
                                              "the argmax); agreement over reference pixels "
                                              "that are not background") if six else
                                             "seven-class ensemble map; agreement over all pixels",
                               "tiles": rows}, indent=2))
    print(f"wrote {out}")
    for r in rows:
        print(f"  {r}")


if __name__ == "__main__":
    main()
