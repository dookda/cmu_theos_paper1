#!/usr/bin/env python3
"""ตารางแผนที่ผลลัพธ์เต็มซีนของทุกโมเดล — แถวคือเงื่อนไขการฝึก คอลัมน์คือสถาปัตยกรรม

ตัวเลขบอกว่าอะไรดีกว่า แต่ไม่บอกว่าผลต่างหน้าตาเป็นอย่างไรบนพื้นที่จริง
ตารางนี้ให้ผู้อ่านเห็นเองว่าอิทธิพลขนาดหนึ่งถึงสองจุดที่รายงานไว้
แทบมองไม่ออกด้วยตา ซึ่งเป็นข้อสังเกตที่ตัวเลขอย่างเดียวสื่อไม่ได้

thumbnail สร้างบนคลัสเตอร์ด้วย scene_thumbnail.py แล้วดึงมาเฉพาะ PNG

ใช้งาน
------
  python paper1/scripts/fig_scene_grid.py
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                    # noqa: E402
from matplotlib.patches import Patch               # noqa: E402
from PIL import Image                              # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palette import CLASS                          # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
THUMBS = ROOT / "analysis" / "scene_grid"
RES = ROOT / "analysis" / "results"
OUT = ROOT / "analysis" / "figure" / "fig_scene_grid.png"

ARCH = {"segformer_b2": "SegFormer-B2", "upernet_convnextt": "UPerNet\nConvNeXt-T",
        "hrnet_w48": "HRNet-W48", "unetpp_effb3": "U-Net++\nEffNet-B3",
        "unet_effb3": "U-Net\nEffNet-B3", "unet_resnet50": "U-Net\nResNet50",
        "unet_densenet121": "U-Net\nDenseNet121", "unet_mobilenet": "U-Net\nMobileNetV2",
        "deeplabv3_r101": "DeepLabV3+\nResNet101"}
COND = {"b0_rgb_plain": "RGB\nplain", "b1_rgb_aug": "RGB\n+aug",
        "b2_rgb_wt": "RGB\n+weights", "b3_rgb_aug_wt": "RGB\n+aug+weights",
        "b4_rgbnir_plain": "RGB-NIR\nplain", "b5_rgbnir_aug": "RGB-NIR\n+aug",
        "b6_rgbnir_wt": "RGB-NIR\n+weights", "b7_rgbnir_aug_wt": "RGB-NIR\n+aug+weights"}
CLASSES = ["background", "vegetation", "water", "urban", "agriculture",
           "bare_soil", "road"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    pngs = {p.stem: p for p in THUMBS.glob("*.png")}
    if not pngs:
        return print(f"ข้าม: ไม่มี thumbnail ใน {THUMBS}")

    miou = {}
    ev = RES / "rerun_eval_test.json"
    if ev.exists():
        miou = {r["run"]: r["mean_iou"] for r in json.loads(ev.read_text())["results"]}

    # จัดเป็นตาราง: คอลัมน์ = สถาปัตยกรรมที่ครบทุกเงื่อนไข, แถว = เงื่อนไข
    cells = {}
    for run in pngs:
        arch, cond, _ = run.split("__")
        cells.setdefault(arch, {})[cond] = run
    full = [x for x in ARCH if x in cells and len(cells[x]) > 1]
    part = [x for x in ARCH if x in cells and len(cells[x]) == 1]
    conds = [c for c in COND if any(c in v for v in cells.values())]

    ncol = max(len(full), len(part))
    nrow = len(conds) + (1 if part else 0)
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.55 * ncol, 1.72 * nrow + 0.9),
                             squeeze=False)
    for r, cond in enumerate(conds):
        for c in range(ncol):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4); sp.set_color("0.6")
            run = cells.get(full[c], {}).get(cond) if c < len(full) else None
            if run and run in pngs:
                ax.imshow(Image.open(pngs[run]))
                if run in miou:
                    ax.set_xlabel(f"{100 * miou[run]:.1f}", fontsize=6.5, labelpad=1.5)
            else:
                ax.set_facecolor("0.97")
            if c == 0:
                ax.set_ylabel(COND[cond], fontsize=7, rotation=0, ha="right",
                              va="center", labelpad=26)
            if r == 0 and c < len(full):
                ax.set_title(ARCH[full[c]], fontsize=7.5, pad=4)

    if part:
        r = len(conds)
        for c in range(ncol):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4); sp.set_color("0.6")
            if c < len(part):
                run = next(iter(cells[part[c]].values()))
                ax.imshow(Image.open(pngs[run]))
                ax.set_title(ARCH[part[c]], fontsize=7.5, pad=4)
                if run in miou:
                    ax.set_xlabel(f"{100 * miou[run]:.1f}", fontsize=6.5, labelpad=1.5)
            else:
                ax.set_facecolor("0.97"); ax.axis("off")
            if c == 0:
                ax.set_ylabel("RGB plain\n(baselines)", fontsize=7, rotation=0,
                              ha="right", va="center", labelpad=26)

    fig.legend([Patch(facecolor=CLASS[c], edgecolor="black", lw=0.4) for c in CLASSES],
               [c.replace("_", " ") for c in CLASSES],
               loc="lower center", ncol=7, fontsize=7.5, frameon=True,
               edgecolor="black", bbox_to_anchor=(0.5, 0.004))
    fig.suptitle(f"Full-scene land cover from every trained model at one seed\n"
                 f"{len(pngs)} models; the number under each panel is test mIoU (%)",
                 fontsize=10.5, y=0.995)
    fig.tight_layout(rect=(0, 0.045, 1, 0.975))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {a.out}  ({nrow} แถว x {ncol} คอลัมน์, {len(pngs)} ภาพ)")


if __name__ == "__main__":
    main()
