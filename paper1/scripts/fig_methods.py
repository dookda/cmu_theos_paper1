#!/usr/bin/env python3
"""เปรียบเทียบผลทำนายของทุกวิธีบนภาพย่อยที่กันไว้สี่ภาพ (รูปเสริม)

แถวคือวิธี (เก้าสถาปัตยกรรม, encoder ที่ฝึกล่วงหน้าบนภาพมุมสูงสองแบบ, ฝึกโดยไม่มีพื้นหลัง,
ensemble) คอลัมน์คือภาพย่อย สองแถวแรกเป็นภาพ THEOS-2 และ label อ้างอิง ทุกวิธีใช้เงื่อนไข
RGB + augmentation, seed 0 และแสดงแบบไม่ให้ทำนายพื้นหลัง (หกคลาส) ให้ตรงกับแผนที่หลัก
พื้นที่ที่ label ไม่ได้กำหนดคลาสแสดงเป็นภาพจางทับลายขีด เหมือน fig_examples.py

ผลทำนายมาจาก analysis/results/method_tiles.npz (predict_tiles.py บน Erawan)
ภาพ THEOS-2 ตัดจาก theos/lamphoon/lamphoon_merged.tif ด้วย stretch ของ render.qml

ใช้งาน
------
  python paper1/scripts/fig_methods.py
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                    # noqa: E402
import numpy as np                                 # noqa: E402
import rasterio                                    # noqa: E402
from PIL import Image                              # noqa: E402
from matplotlib.patches import Patch               # noqa: E402
from rasterio.windows import from_bounds           # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palette import CLASS as CLASS_COLOR           # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parent
NPZ = ROOT / "analysis" / "results" / "method_tiles.npz"
META = REPO / "paper2" / "analysis" / "results" / "tiles_metadata.json"
SPLITS = ROOT / "analysis" / "results" / "spatial_splits.json"
IMG = REPO / "theos" / "lamphoon" / "lamphoon_merged.tif"
OUT = ROOT / "analysis" / "figure" / "fig_methods.png"
CLASSES = ["background", "vegetation", "water", "urban", "agriculture", "bare_soil", "road"]
STRETCH = [(100, 541), (116, 395), (164, 404)]
ROWS = [("unet_mobilenet__b1_rgb_aug__s0", "U-Net MobileNetV2"),
        ("unet_resnet50__b1_rgb_aug__s0", "U-Net ResNet50"),
        ("unet_densenet121__b1_rgb_aug__s0", "U-Net DenseNet121"),
        ("unet_effb3__b1_rgb_aug__s0", "U-Net EffNet-B3"),
        ("unetpp_effb3__b1_rgb_aug__s0", "U-Net++ EffNet-B3"),
        ("deeplabv3_r101__b1_rgb_aug__s0", "DeepLabV3+ ResNet101"),
        ("hrnet_w48__b1_rgb_aug__s0", "HRNet-W48"),
        ("segformer_b2__b1_rgb_aug__s0", "SegFormer-B2"),
        ("upernet_convnextt__b1_rgb_aug__s0", "UPerNet ConvNeXt-T"),
        ("satlas_swinb__b1_rgb_aug__s0", "SatlasPretrain Swin-B"),
        ("dinov3_vitl_sat__b1_rgb_aug__s0", "DINOv3-SAT ViT-L"),
        ("segformer_b2__b1_rgb_aug_ignbg__s0", "SegFormer-B2,\ntrained without\nbackground"),
        ("ensemble", "Ensemble\n(7 members)")]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=600)
    a = ap.parse_args()
    if not NPZ.exists():
        return print("ข้าม fig_methods: ยังไม่มี method_tiles.npz")
    d = np.load(NPZ)
    tiles = [str(t) for t in d["tiles"]]
    labels = d["labels"]
    meta = {m["filename"]: m for m in json.loads(META.read_text())}
    sp = json.loads(SPLITS.read_text())["splits"]
    where = {t: ("Scene 1 test" if t in sp["test"] else "Scene 2 held out") for t in tiles}

    lut = np.array([[int(CLASS_COLOR[c][i:i + 2], 16) / 255 for i in (1, 3, 5)] for c in CLASSES])
    rows = [r for r in ROWS if f"{r[0]}__sup" in d.files]
    short = {"U-Net MobileNetV2": "U-Net\nMobileNetV2", "U-Net ResNet50": "U-Net\nResNet50",
             "U-Net DenseNet121": "U-Net\nDenseNet121", "U-Net EffNet-B3": "U-Net\nEffNet-B3",
             "U-Net++ EffNet-B3": "U-Net++\nEffNet-B3", "DeepLabV3+ ResNet101": "DeepLabV3+\nResNet101",
             "HRNet-W48": "HRNet-W48\n", "SegFormer-B2": "SegFormer-B2\n",
             "UPerNet ConvNeXt-T": "UPerNet\nConvNeXt-T", "SatlasPretrain Swin-B": "SatlasPretrain\nSwin-B",
             "DINOv3-SAT ViT-L": "DINOv3-SAT\nViT-L",
             "SegFormer-B2,\ntrained without\nbackground": "SegFormer-B2,\nno background",
             "Ensemble\n(7 members)": "Ensemble\n(7 members)"}
    # แต่ละภาพย่อยเป็นบล็อก 3 แถว x 5 ช่อง (ภาพ, label, 13 วิธี) วางบล็อกเป็น 2 x 2
    panels = [("THEOS-2\n(R, G, B)", None), ("Reference\nlabel", None)] + \
             [(short.get(lab, lab), key) for key, lab in rows]
    PC, PR = 5, 3
    fig = plt.figure(figsize=(7.16, 6.0))
    # แถวเว้นระหว่างบล็อกบนและล่าง ให้หัวบล็อกล่างไม่ชนแถวสุดท้ายของบล็อกบน
    gs = fig.add_gridspec(2 * PR + 1, 2 * PC + 1, left=0.005, right=0.995, top=0.935, bottom=0.07,
                          wspace=0.08, hspace=0.5, height_ratios=[1] * PR + [0.18] + [1] * PR,
                          width_ratios=[1] * PC + [0.12] + [1] * PC)
    agree = {}
    matplotlib.rcParams["hatch.linewidth"] = 0.3
    if True:
        for j, t in enumerate(tiles):
            br, bc = divmod(j, 2)
            rgb = tile_rgb(t)
            lab = labels[j]
            h, w = min(rgb.shape[0], lab.shape[0]), min(rgb.shape[1], lab.shape[1])
            rgb, lab = rgb[:h, :w], lab[:h, :w]
            grey = rgb.mean(axis=2, keepdims=True) * 0.55 + 0.4
            for k, (title, key) in enumerate(panels):
                r, c = divmod(k, PC)
                ax = fig.add_subplot(gs[br * (PR + 1) + r, bc * (PC + 1) + c])
                if k == 0:
                    ax.imshow(rgb)
                elif k == 1:
                    ax.imshow(np.where((lab == 0)[..., None], np.broadcast_to(grey, rgb.shape),
                                       lut[lab]))
                    if (lab == 0).any():
                        ax.contourf((lab == 0).astype(float), levels=[0.5, 1.5], colors="none",
                                    hatches=["////"])
                else:
                    pr = d[f"{key}__sup"][j][:h, :w]
                    ax.imshow(lut[pr])
                    a_ = float((pr == lab)[lab != 0].mean())
                    agree.setdefault(key, []).append(a_)
                    ax.text(0.96, 0.05, f"{100 * a_:.0f}%", ha="right", va="bottom", fontsize=5,
                            color="black", transform=ax.transAxes,
                            bbox=dict(fc="white", ec="none", pad=0.5))
                ax.set_title(title, fontsize=5.6, pad=1.5, color="black", linespacing=0.95)
                ax.set_xticks([]); ax.set_yticks([])
                for sp_ in ax.spines.values():
                    sp_.set_linewidth(0.4)
            x0 = gs[br * (PR + 1), bc * (PC + 1)].get_position(fig).x0
            y1 = gs[br * (PR + 1), bc * (PC + 1)].get_position(fig).y1
            fig.text(x0, y1 + 0.032, f"({chr(97 + j)}) {where[t]}, {t.replace('tile_', 'tile ')}",
                     fontsize=7, fontweight="bold", color="black", ha="left", va="bottom")
    handles = [Patch(facecolor="0.85", edgecolor="0.25", lw=0.4, hatch="////",
                     label="no reference label")]
    handles += [Patch(facecolor=CLASS_COLOR[c], edgecolor="black", lw=0.4,
                      label=c.replace("_", " ")) for c in CLASSES[1:]]
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=6.5, frameon=False,
               bbox_to_anchor=(0.5, 0.0), handlelength=1.4, columnspacing=1.0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=a.dpi, facecolor="white")
    (ROOT / "analysis" / "results" / "method_tiles_agreement.json").write_text(json.dumps(
        {"note": "share of labelled (non-background) reference pixels where the six-class "
                 "prediction equals the label, per tile; descriptive, four tiles only",
         "tiles": tiles, "agreement": agree}, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
