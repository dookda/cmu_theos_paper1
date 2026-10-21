#!/usr/bin/env python3
"""รูปแผนที่ผลลัพธ์เต็มฉากของ ensemble — ขาว-ดำ มี north arrow และ scale bar

เจ็ดคลาสในภาพขาว-ดำต้องแยกออกจากกันเมื่อพิมพ์ จึงใช้ระดับเทาที่เว้นระยะเท่ากัน
เรียงตามความสว่างที่สื่อความหมาย (น้ำเข้มสุด ดินเปล่าสว่างสุด) แทนการไล่ตามเลขคลาส
no-data เป็นสีขาวและไม่ปรากฏในคำอธิบาย

ใช้งาน
------
  python paper1/scripts/fig_map.py
  python paper1/scripts/fig_map.py --downsample 6 --dpi 500
"""
import argparse, os, sys
from pathlib import Path

os.environ.pop("PROJ_DATA", None); os.environ.pop("PROJ_LIB", None)
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.colors import ListedColormap, BoundaryNorm   # noqa: E402
from matplotlib.patches import Patch                  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
# ระดับเทาเรียงตามความหมาย: น้ำเข้ม -> พืช -> เมือง -> เกษตร -> ดินเปล่าสว่าง
# เดิมเป็นเฉดเทา ซึ่งแยกเจ็ดคลาสด้วยตาแทบไม่ได้ ใช้ชุดสีกลางแทน
sys.path.insert(0, str(Path(__file__).resolve().parent))
from palette import CLASS as CLASS_COLOR                       # noqa: E402
ORDER = ["water", "vegetation", "urban", "road", "agriculture", "background", "bare_soil"]
CLASSES = ["background", "vegetation", "water", "urban", "agriculture", "bare_soil", "road"]


def scale_bar(ax, x, y, length_m, res_m, label):
    L = length_m / res_m
    ax.plot([x, x + L], [y, y], color="black", lw=3, solid_capstyle="butt", zorder=6)
    for xi in (x, x + L):
        ax.plot([xi, xi], [y - L * 0.035, y + L * 0.035], color="black", lw=1.6, zorder=6)
    ax.text(x + L / 2, y - L * 0.10, label, ha="center", va="bottom",
            fontsize=8.5, zorder=6)


def north_arrow(ax, x, y, size):
    ax.annotate("", xy=(x, y - size), xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5), zorder=6)
    ax.text(x, y - size * 1.28, "N", ha="center", va="top",
            fontsize=11, fontweight="bold", zorder=6)


def read_map(path, d):
    """แผนที่ลดความละเอียดด้วย mode พร้อมขอบเขต (left, bottom, right, top)"""
    with rasterio.open(path) as s:
        arr = s.read(1, out_shape=(s.height // d, s.width // d),
                     resampling=rasterio.enums.Resampling.mode)
        return arr, tuple(s.bounds), s.res[0] * d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=str(ROOT / "outputs" / "map_ensemble_lamphun.tif"))
    ap.add_argument("--map2", default=str(ROOT / "outputs" / "map_ensemble_lamphun_scene2.tif"),
                    help="แผนที่ฉากที่ 2 (ถ้ามี จะต่อเป็นภาพเดียวกับฉากที่ 1)")
    ap.add_argument("--downsample", type=int, default=4)
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--out", default=str(ROOT / "analysis" / "figure" / "fig_map.png"))
    a = ap.parse_args()

    d = a.downsample
    maps = [read_map(a.map, d)]
    if Path(a.map2).exists():
        maps.append(read_map(a.map2, d))
    res = maps[0][2]
    L = min(m[1][0] for m in maps); B = min(m[1][1] for m in maps)
    Rt = max(m[1][2] for m in maps); T = max(m[1][3] for m in maps)
    H, W = int(round((T - B) / res)), int(round((Rt - L) / res))
    arr = np.full((H, W), 255, np.uint8)
    boxes = []
    # ฉากที่ 1 วางก่อน ฉากที่ 2 เติมเฉพาะส่วนที่ฉากที่ 1 ไม่มีข้อมูล:
    # ในเขตทับซ้อนจึงแสดงผลของฉากที่ 1 ซึ่งเป็นฉากที่ใช้ฝึก
    for m, b, _ in maps:
        r0, c0 = int(round((T - b[3]) / res)), int(round((b[0] - L) / res))
        h, w = min(m.shape[0], H - r0), min(m.shape[1], W - c0)
        sub = arr[r0:r0 + h, c0:c0 + w]
        fill = sub == 255
        sub[fill] = m[:h, :w][fill]
        boxes.append((c0, r0, w, h))
    print(f"แผนที่ {W} x {H} px ที่ {res:.1f} ม./px จาก {len(maps)} ฉาก")

    # แปลง index คลาส -> สี RGB ตามชุดสีกลาง พิกเซลนอกคลาสให้เป็นขาว
    def _hex2rgb(h):
        return tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))

    lut = np.ones((256, 3), dtype=float)
    for i, c in enumerate(CLASSES):
        lut[i] = _hex2rgb(CLASS_COLOR[c])
    img = lut[arr]

    two = len(maps) == 2
    fw = 5.6 if two else 7.2
    fig, ax = plt.subplots(figsize=(fw, fw * H / W))
    ax.imshow(img, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)

    if two:
        from matplotlib.patches import Rectangle
        for (c0, r0, w, h), col, lab, va, yy in zip(
                boxes, ("#d55e00", "#0072b2"),
                ("Scene 1: training, validation and internal test",
                 "Scene 2: held out south of the overlap with Scene 1"),
                ("bottom", "top"), (0, 1)):
            ax.add_patch(Rectangle((c0, r0), w, h, fill=False, ec=col, lw=1.3, zorder=5))
            ax.text(c0 + w / 2, r0 - H * 0.006 if yy == 0 else r0 + h + H * 0.006, lab,
                    ha="center", va=va, fontsize=8, color="black", zorder=6)
    scale_bar(ax, W * 0.05, H * (0.035 if two else 0.955), 2000, res, "2 km")
    north_arrow(ax, W * 0.955, H * (0.08 if two else 0.10), H * (0.04 if two else 0.055))
    ax.set_xlim(-W * 0.01, W * 1.01); ax.set_ylim(H * 1.035, -H * 0.035)

    handles = [Patch(facecolor=CLASS_COLOR[c], edgecolor="black", lw=0.5,
                     label=c.replace("_", " ")) for c in ORDER]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01),
              ncol=4, fontsize=8.5, frameon=True, edgecolor="black", handlelength=1.6)
    ax.set_title(("Land cover of both Lamphun scenes" if two else
                  "Land cover of the Lamphun scene") + ", THEOS-2 8 January 2026\n"
                 "seven-member ensemble selected on validation alone",
                 fontsize=10.5, pad=10)

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=a.dpi, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
