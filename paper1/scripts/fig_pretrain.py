#!/usr/bin/env python3
"""เทียบ encoder ที่ฝึกล่วงหน้าบนภาพมุมสูง (addendum E) กับเก้าสถาปัตยกรรมของแฟกทอเรียล

รูปเปรียบเทียบโมเดลรูปอื่นสร้างจากแฟกทอเรียล 360 รอบ จึงไม่มี SatlasPretrain และ
DINOv3-SAT ซึ่งฝึกเฉพาะเงื่อนไข RGB + augmentation ห้า seed รูปนี้เทียบในเงื่อนไขเดียวกัน
(b1_rgb_aug) เพื่อให้ความต่างที่เห็นเป็นของ encoder ไม่ใช่ของเงื่อนไขการฝึก

แหล่งข้อมูล
  analysis/results/rerun_eval_{test,external}.json        เก้าสถาปัตยกรรม (ใช้เฉพาะ b1_rgb_aug)
  analysis/results/foundation_eval_{test,external}.json   Satlas, DINOv3

ใช้งาน
------
  python paper1/scripts/fig_pretrain.py
"""
import argparse
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

from fig_results import NICE, load, save  # noqa: E402

COND = "b1_rgb_aug"
FOUNDATION = {"satlas_swinb": "SatlasPretrain Swin-B",
              "dinov3_vitl_sat": "DINOv3-SAT ViT-L"}
ACCENT = "#0072b2"


def collect(split):
    g = defaultdict(list)
    for name in (f"rerun_eval_{split}.json", f"foundation_eval_{split}.json"):
        ev = load(name)
        if not ev:
            return None
        for r in ev["results"]:
            arch, cond = r["run"].split("__")[:2]
            if cond == COND:
                g[arch].append(100 * r["mean_iou"])
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=600)
    a = ap.parse_args()
    data = {s: collect(s) for s in ("test", "external")}
    if not all(data.values()) or not all(k in data["external"] for k in FOUNDATION):
        return print("ข้าม fig_pretrain: ยังไม่มีผล foundation")
    archs = sorted(data["external"], key=lambda k: st.mean(data["external"][k]))
    fig, ax = plt.subplots(figsize=(7.0, 0.40 * len(archs) + 2.1))
    for i, k in enumerate(archs):
        if k in FOUNDATION:
            ax.axhspan(i - 0.5, i + 0.5, color=ACCENT, alpha=0.08, lw=0)
    for split, mk, fc, lab, dy in (("test", "o", "black", "Internal test", 0.14),
                                   ("external", "s", "white", "Held-out scene", -0.14)):
        for i, k in enumerate(archs):
            v = data[split][k]
            ax.plot(v, [i + dy] * len(v), "|", color="0.55", ms=7, mew=1, zorder=2)
            ax.plot(st.mean(v), i + dy, mk, ms=6, mfc=fc, mec="black", mew=1.1, zorder=3,
                    label=lab if i == 0 else None)
    ax.set_yticks(range(len(archs)))
    ax.set_yticklabels([FOUNDATION.get(k, NICE.get(k, k)) for k in archs], fontsize=8.5)
    for tick, k in zip(ax.get_yticklabels(), archs):
        if k in FOUNDATION:
            tick.set_fontweight("bold")          # ตัวหนังสือสีดำทั้งรูป; แถบพื้นสีบอกกลุ่มแทน
    ax.set_ylim(-0.6, len(archs) - 0.4)
    ax.set_xlabel("mIoU (%), RGB with augmentation; ticks are the five seeds", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(axis="x", color="0.9", lw=0.6); ax.set_axisbelow(True)
    ax.legend(fontsize=8.5, frameon=True, edgecolor="black",
              loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax.set_title("Encoders pretrained on overhead imagery against the factorial's architectures\n"
                 "same training condition; shaded rows are pretrained on overhead imagery",
                 fontsize=10, pad=10)
    save(fig, "fig_pretrain.png", a.dpi)


if __name__ == "__main__":
    main()
