#!/usr/bin/env python3
"""แผนที่พื้นที่ศึกษา: ตำแหน่งในประเทศไทย, อำเภอของลำพูน, และภาพ THEOS-2 สีจริง

(a) ประเทศไทยรายจังหวัด เน้นลำพูน
(b) จังหวัดลำพูนรายอำเภอ พร้อมกรอบภาพ THEOS-2 สองฉาก
(c) ภาพ THEOS-2 สีจริง (แบนด์ 1-3 = R,G,B; stretch ตาม render.qml ชุดเดียวกับไทล์ฝึก)
    ซ้อนขอบเขตตำบลและอำเภอ

แหล่งข้อมูล
  theos/lamphoon/lamphoon_merged.tif                    ภาพ 0.5 ม. (ลดความละเอียดเพื่อแสดง)
  Landuse_lamphoon_2563/admin_boundary/{amp,tambon}_ลำพูน.shp  ขอบเขตอำเภอ/ตำบล (UTM 47N)
  analysis/shapes/scenes.geojson                        กรอบภาพสองฉาก
  analysis/shapes/external/geoBoundaries-THA-ADM1_simplified.geojson
      ขอบเขตจังหวัด geoBoundaries gbOpen (OpenStreetMap, ODbL 1.0)

ต้องใช้ geopandas: ~/miniforge3/bin/python paper1/scripts/fig_study_area.py
"""
import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe                # noqa: E402
import matplotlib.pyplot as plt                    # noqa: E402
import numpy as np                                 # noqa: E402
import rasterio                                    # noqa: E402
from matplotlib.lines import Line2D                # noqa: E402
from matplotlib.patches import Patch               # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parent
IMG = REPO / "theos" / "lamphoon" / "lamphoon_merged.tif"
ADM = ROOT / "Landuse_lamphoon_2563" / "admin_boundary"
SCENES = ROOT / "analysis" / "shapes" / "scenes.geojson"
THA = ROOT / "analysis" / "shapes" / "external" / "geoBoundaries-THA-ADM1_simplified.geojson"
OUT = ROOT / "analysis" / "figure" / "fig_study_area.png"
UTM = "EPSG:32647"
STRETCH = [(100, 541), (116, 395), (164, 404)]
SCENE_C = ["#d55e00", "#0072b2"]
HALO = [pe.withStroke(linewidth=2.4, foreground="white")]
# ชื่อตำบลในอำเภอเมืองลำพูน ถอดเป็นอักษรโรมันตามหลักราชบัณฑิตยสถาน (RTGS)
TAMBON_EN = {"ต.ในเมือง": "Nai Mueang", "ต.เหมืองง่า": "Mueang Nga", "ต.อุโมงค์": "Umong",
             "ต.หนองช้างคืน": "Nong Chang Khuen", "ต.ประตูป่า": "Pratu Pa",
             "ต.ริมปิง": "Rim Ping", "ต.ต้นธง": "Ton Thong", "ต.บ้านแป้น": "Ban Paen",
             "ต.เหมืองจี้": "Mueang Chi", "ต.ป่าสัก": "Pa Sak", "ต.เวียงยอง": "Wiang Yong",
             "ต.บ้านกลาง": "Ban Klang", "ต.มะเขือแจ้": "Makhuea Chae",
             "ต.ศรีบัวบาน": "Si Bua Ban", "ต.หนองหนาม": "Nong Nam"}
DIST_OFFSET = {"Mae Tha": (2000, -2000)}
LEADER = {"Pa Sang": (466000, 2053000)}   # อำเภอเล็ก ใช้เส้นชี้ไปยังที่ว่าง


def true_color(downsample):
    with rasterio.open(IMG) as s:
        h, w = s.height // downsample, s.width // downsample
        a = s.read([1, 2, 3], out_shape=(3, h, w),
                   resampling=rasterio.enums.Resampling.average).astype(float)
        b = s.bounds
    valid = a.sum(axis=0) > 0
    for k, (lo, hi) in enumerate(STRETCH):
        a[k] = np.clip((a[k] - lo) / (hi - lo), 0, 1)
    rgba = np.dstack([a[0], a[1], a[2], valid.astype(float)])
    return rgba, (b.left, b.right, b.bottom, b.top)


def scale_bar(ax, x, y, km, label):
    m = km * 1000
    ax.plot([x, x + m], [y, y], color="black", lw=3, solid_capstyle="butt", zorder=8)
    ax.plot([x, x + m / 2], [y, y], color="white", lw=1.4, solid_capstyle="butt", zorder=9)
    ax.text(x + m / 2, y + m * 0.08, label, ha="center", va="bottom", fontsize=7,
            path_effects=HALO, zorder=9)


def north(ax, x, y, size):
    ax.annotate("", xy=(x, y), xytext=(x, y - size),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.3), zorder=9)
    ax.text(x, y + size * 0.1, "N", ha="center", va="bottom", fontsize=8,
            fontweight="bold", path_effects=HALO, zorder=9)


def utm_ticks(ax, step):
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    xs = np.arange(np.ceil(x0 / step) * step, x1, step)
    ys = np.arange(np.ceil(y0 / step) * step, y1, step)
    ax.set_xticks(xs); ax.set_yticks(ys)
    ax.set_xticklabels([f"{v / 1000:.0f}" for v in xs], fontsize=6.5)
    ax.set_yticklabels([f"{v / 1000:.0f}" for v in ys], fontsize=6.5, rotation=90,
                       va="center")
    ax.tick_params(length=2, pad=1.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--downsample", type=int, default=8)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    tha = gpd.read_file(THA)
    lpn = tha[tha.shapeName.str.contains("Lamphun")]
    amp = gpd.read_file(ADM / "amp_ลำพูน.shp").to_crs(UTM)
    tam = gpd.read_file(ADM / "tambon_ลำพูน.shp", encoding="cp874").to_crs(UTM)
    sc = gpd.read_file(SCENES).to_crs(UTM).sort_values("scene")
    rgba, ext = true_color(a.downsample)

    fig = plt.figure(figsize=(7.16, 6.3))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.55], height_ratios=[1, 1.15],
                          wspace=0.12, hspace=0.16)
    ax_a = fig.add_subplot(gs[0, 0]); ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[:, 1])

    # (a) Thailand
    tha.plot(ax=ax_a, facecolor="#e6e6e6", edgecolor="#9a9a9a", lw=0.25)
    lpn.plot(ax=ax_a, facecolor="#c0392b", edgecolor="#7b241c", lw=0.5)
    c = lpn.geometry.iloc[0].centroid
    ax_a.annotate("Lamphun", xy=(c.x, c.y), xytext=(c.x + 2.2, c.y + 0.6), fontsize=7,
                  arrowprops=dict(arrowstyle="-", lw=0.6), path_effects=HALO)
    ax_a.text(100.6, 13.75, "Bangkok", fontsize=6, ha="left", path_effects=HALO)
    ax_a.plot(100.52, 13.75, "o", ms=2.2, color="black")
    ax_a.set_xlim(97.2, 105.8); ax_a.set_ylim(5.5, 20.6); ax_a.set_aspect("equal")
    ax_a.set_xticks([98, 100, 102, 104]); ax_a.set_yticks([8, 12, 16, 20])
    ax_a.set_xticklabels([f"{v}°E" for v in [98, 100, 102, 104]], fontsize=6.5)
    ax_a.set_yticklabels([f"{v}°N" for v in [8, 12, 16, 20]], fontsize=6.5)
    ax_a.tick_params(length=2, pad=1.5)
    ax_a.set_title("(a) Thailand", fontsize=8, loc="left")

    # (b) Lamphun districts and scene footprints
    amp.plot(ax=ax_b, facecolor="#f4f4f4", edgecolor="#555555", lw=0.5)
    amp[amp.AMPHOE_E.str.contains("Muang")].plot(ax=ax_b, facecolor="#f6dcd8",
                                                  edgecolor="#555555", lw=0.5)
    for _, r in amp.iterrows():
        p = r.geometry.representative_point()
        nm = r.AMPHOE_E.replace("Amphoe ", "").replace("Muang", "Mueang")
        if nm in LEADER:
            ax_b.annotate(nm, xy=(p.x, p.y), xytext=LEADER[nm], fontsize=5.2,
                          ha="center", va="center", path_effects=HALO,
                          arrowprops=dict(arrowstyle="-", lw=0.5, color="#333333"))
            continue
        dx, dy = DIST_OFFSET.get(nm, (0, 0))
        ax_b.text(p.x + dx, p.y + dy, nm.replace(" ", "\n", 1 if len(nm) > 12 else 0),
                  fontsize=5.2, ha="center", va="center", path_effects=HALO)
    for (_, r), col in zip(sc.iterrows(), SCENE_C):
        gpd.GeoSeries([r.geometry], crs=UTM).boundary.plot(ax=ax_b, color=col, lw=1.0)
    ax_b.set_aspect("equal"); utm_ticks(ax_b, 20000)
    ax_b.set_title("(b) Lamphun Province, districts", fontsize=8, loc="left")
    scale_bar(ax_b, ax_b.get_xlim()[0] + 3000, ax_b.get_ylim()[0] + 4000, 20, "20 km")

    # (c) THEOS-2 true colour with subdistrict and district boundaries
    ax_c.imshow(rgba, extent=ext, interpolation="bilinear", zorder=1)
    tam.boundary.plot(ax=ax_c, color="#ffd23f", lw=0.55, zorder=3)
    amp.boundary.plot(ax=ax_c, color="#222222", lw=1.3, zorder=4,
                      path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])
    wy = tam[tam.TAM_NAM_T == "ต.เวียงยอง"]
    wy.boundary.plot(ax=ax_c, color="#ff2d55", lw=1.6, zorder=4.5)
    for _, r in tam.iterrows():
        en = TAMBON_EN.get(r.TAM_NAM_T)
        p = r.geometry.representative_point()
        if en and ext[0] < p.x < ext[1] and ext[2] < p.y < ext[3]:
            ax_c.text(p.x, p.y, en, fontsize=5.6, ha="center", va="center", zorder=6,
                      fontweight="bold" if en == "Wiang Yong" else "normal",
                      color="black", path_effects=HALO)
    for (_, r), col, name in zip(sc.iterrows(), SCENE_C, ["Scene 1", "Scene 2"]):
        gpd.GeoSeries([r.geometry], crs=UTM).boundary.plot(ax=ax_c, color=col, lw=1.2,
                                                           zorder=5)
    pad = 600
    ax_c.set_xlim(ext[0] - pad, ext[1] + pad); ax_c.set_ylim(ext[2] - pad, ext[3] + pad)
    ax_c.set_aspect("equal"); utm_ticks(ax_c, 4000)
    ax_c.set_facecolor("#f4f4f4")
    ax_c.set_title("(c) THEOS-2 true colour, 8 January 2026", fontsize=8, loc="left")
    x0, x1 = ax_c.get_xlim(); y0, y1 = ax_c.get_ylim()
    scale_bar(ax_c, x0 + 800, y0 + 900, 4, "4 km")
    north(ax_c, x0 + 1100, y1 - 900, 1800)
    ax_c.set_xlabel("Easting (km), UTM zone 47N", fontsize=6.5, labelpad=2)
    ax_b.set_xlabel("Easting (km)", fontsize=6.5, labelpad=2)

    handles = [Patch(facecolor="#c0392b", edgecolor="#7b241c", label="Lamphun Province"),
               Patch(facecolor="#f6dcd8", edgecolor="#555555", label="Mueang Lamphun District"),
               Line2D([], [], color="#555555", lw=1.4, label="District boundary"),
               Line2D([], [], color="#ffd23f", lw=1.2, label="Subdistrict boundary"),
               Line2D([], [], color="#ff2d55", lw=1.6, label="Wiang Yong Subdistrict"),
               Line2D([], [], color=SCENE_C[0], lw=1.4, label="THEOS-2 Scene 1 extent"),
               Line2D([], [], color=SCENE_C[1], lw=1.4, label="THEOS-2 Scene 2 extent")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6.5, frameon=False,
               bbox_to_anchor=(0.5, -0.005))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.96, bottom=0.11)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=a.dpi, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
