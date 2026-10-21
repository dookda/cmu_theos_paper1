#!/usr/bin/env python3
"""รูปผลลัพธ์ของ paper1 re-run — มี title ในภาพ ไม่มี caption

ใช้สีจาก palette.py ตั้งแต่ 2026-08-24 โดยยังคง marker ต่างกันควบคู่เสมอ
ภาพที่มีแค่สองสามกลุ่ม (forest plot) ยังเป็นขาว-ดำ เพราะรูปทรงพอแล้ว

สามรูป อ่านตัวเลขจากไฟล์ผลทั้งหมด ไม่มี hardcode:
  fig_effects  ผลของสามปัจจัยต่อสถาปัตยกรรม พร้อม 95% CI
  fig_blocks   block bootstrap ราย contrast บน test และ external
  fig_perclass IoU รายคลาสของ ensemble บนสอง held-out split

ทำไมเป็น forest plot ไม่ใช่ bar chart: ปริมาณที่งานนี้รายงานคือ *ช่วง* ไม่ใช่จุด
แท่งกราฟชวนให้อ่านความสูงว่าเป็นข้อเท็จจริง ส่วนจุด+เส้นบังคับให้เห็นความไม่แน่นอน
"""
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import matplotlib.patheffects as _pe      # noqa: E402

# ตัวหนังสือทุกจุดในรูปเป็นสีดำ; บนพื้นเข้มใช้ขอบขาวรอบตัวอักษรแทนการสลับเป็นสีขาว
HALO = [_pe.withStroke(linewidth=1.8, foreground="white")]

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from palette import CLASS as CLASS_COLOR, SEQUENTIAL, marker as _mk, series as _sr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "analysis" / "results"
FIG = ROOT / "analysis" / "figure"
NICE = {"unetpp_effb3": "U-Net++ EffNet-B3", "hrnet_w48": "HRNet-W48",
        "unet_effb3": "U-Net EffNet-B3", "segformer_b2": "SegFormer-B2",
        "upernet_convnextt": "UPerNet ConvNeXt-T", "unet_mobilenet": "U-Net MobileNetV2",
        "unet_resnet50": "U-Net ResNet50", "unet_densenet121": "U-Net DenseNet121",
        "deeplabv3_r101": "DeepLabV3+ ResNet101"}
ARM = {"b1_rgb_aug": "Augmentation", "b4_rgbnir_plain": "NIR band",
       "b2_rgb_wt": "Class weighting"}


def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def fig_effects(dpi):
    eff = load("rerun_effects_mean_iou.json")
    if not eff:
        return print("ข้าม fig_effects: ไม่มีไฟล์ผล")
    facs = ["augment", "bands (NIR)", "weights"]
    titles = {"augment": "Augmentation", "bands (NIR)": "NIR band",
              "weights": "Class weighting"}
    archs = sorted({r["architecture"] for r in eff if r["effect"] in facs},
                   key=lambda a: -next(x["mean"] for x in eff
                                       if x["architecture"] == a and x["effect"] == "augment"))
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.1), sharey=True)
    for ax, f in zip(axes, facs):
        for i, a in enumerate(archs):
            r = next((x for x in eff if x["architecture"] == a and x["effect"] == f), None)
            if not r:
                continue
            sig = not (r["ci_lo"] <= 0 <= r["ci_hi"])
            ax.plot([100 * r["ci_lo"], 100 * r["ci_hi"]], [i, i],
                    color="black", lw=1.1, solid_capstyle="butt", zorder=2)
            ax.plot(100 * r["mean"], i, "o", ms=5.5, zorder=3,
                    mfc="black" if sig else "white", mec="black", mew=1.1)
        ax.axvline(0, color="0.55", lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.set_title(titles[f], fontsize=9.5)
        ax.set_xlabel("Effect on mIoU (points)", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(axis="x", color="0.9", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(archs)))
    axes[0].set_yticklabels([NICE.get(a, a) for a in archs], fontsize=8)
    axes[0].set_ylim(-0.6, len(archs) - 0.4)
    fig.suptitle("Effect of each training factor on validation mIoU, five seeds\n"
                 "filled marker = 95% CI excludes zero", fontsize=10.5, y=1.06)
    save(fig, "fig_effects.png", dpi)


def fig_blocks(dpi):
    rows = []
    for split in ("test", "external"):
        bb = load(f"block_bootstrap_{split}_mean_iou.json")
        if not bb:
            continue
        for c in bb:
            arm = next((v for k, v in ARM.items() if k in c["arm_a"]), c["arm_a"])
            rows.append((split, arm, c))
    if not rows:
        return print("ข้าม fig_blocks: ไม่มีไฟล์ผล")
    fig, ax = plt.subplots(figsize=(7.4, 0.34 * len(rows) + 1.5))
    for i, (split, arm, c) in enumerate(rows):
        y = len(rows) - 1 - i
        sig = not (c["ci95"][0] <= 0 <= c["ci95"][1])
        ax.plot([100 * c["ci95"][0], 100 * c["ci95"][1]], [y, y], color="black", lw=1.2)
        ax.plot(100 * c["mean_diff"], y, "o", ms=6,
                mfc="black" if sig else "white", mec="black", mew=1.2)
        ax.text(ax.get_xlim()[1], y, f"  {c['block_wins']}/{c['n_blocks']}",
                va="center", fontsize=8, family="monospace")
    ax.axvline(0, color="0.55", lw=0.8, ls=(0, (4, 3)))
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{a}  ({s})" for s, a, _ in rows][::-1], fontsize=8.5)
    ax.set_xlabel("Difference from plain RGB baseline (mIoU points)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(axis="x", color="0.9", lw=0.6); ax.set_axisbelow(True)
    ax.margins(x=0.16)
    ax.set_title("Paired spatial-block bootstrap against the plain baseline\n"
                 "filled marker = 95% CI excludes zero; right column = blocks improved",
                 fontsize=10.5, pad=10)
    save(fig, "fig_blocks.png", dpi)


def fig_perclass(dpi):
    """IoU รายคลาสของ ensemble บนสองชุดที่กันไว้ — เจ็ดคลาส (ตัวชี้วัดหลัก) เทียบกับหกคลาส

    สามค่าต่อคลาส: (1) คะแนนเจ็ดคลาสตามปกติ (2) ผลทำนายเดิมคิดคะแนนเฉพาะหกคลาสจริง
    บนพิกเซลอ้างอิงที่ไม่ใช่พื้นหลัง (3) ไม่ให้ทำนายพื้นหลัง (addendum J) ค่าที่ (2) แยก
    ผลของการเปลี่ยนวิธีคิดคะแนนออกจากผลของการเปลี่ยนคำทำนาย
    """
    ev = {"test": load("ensemble_eval_test.json"), "external": load("ensemble_eval_external.json")}
    j = load("ensemble_bg_results.json") or {}
    if not all(ev.values()):
        return print("ข้าม fig_perclass: ไม่มีไฟล์ผล")
    if not all(j.get(k, {}).get("status") == "done" for k in ev):
        return print("ข้าม fig_perclass: ยังไม่มี ensemble_bg_results.json (addendum J)")
    order = sorted((c for c in ev["external"]["per_class_iou"] if c != "background"),
                   key=lambda c: -j["external"]["sup"]["per_class_iou6"][c])
    rows = ["mean"] + order[::-1] + ["background"]       # ล่างขึ้นบน; พื้นหลังอยู่บนสุด
    series = [("Seven classes (headline metric)", "o", "white"),
              ("Six classes, same prediction", "s", "0.62"),
              ("Six classes, background not predicted", "D", "black")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9), sharey=True)
    for ax, (split, title) in zip(axes, (("test", "Internal test"), ("external", "Held-out scene"))):
        e, b = ev[split], j[split]
        for y, c in enumerate(rows):
            if c == "mean":
                v = [100 * e["mean_iou"], b["std"]["miou6"], b["sup"]["miou6"]]
            elif c == "background":
                v = [100 * e["per_class_iou"][c], None, None]
            else:
                v = [100 * e["per_class_iou"][c], b["std"]["per_class_iou6"][c],
                     b["sup"]["per_class_iou6"][c]]
            xs = [x for x in v if x is not None]
            ax.plot([min(xs), max(xs)], [y, y], color="0.55", lw=1.0, zorder=1)
            for x, (lab, mk, fc) in zip(v, series):
                if x is not None:
                    ax.plot(x, y, mk, ms=5.6, mfc=fc, mec="black", mew=0.9, zorder=3,
                            label=lab if y == 0 else None)
            ax.text(max(xs) + 2.0, y, f"{max(xs):.0f}", va="center", fontsize=7, color="black")
        ax.axhline(0.5, color="0.6", lw=0.6, ls="--")
        ax.set_xlim(20, 104); ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_xlabel("IoU (%)", fontsize=9); ax.tick_params(labelsize=8)
        ax.grid(axis="x", color="0.9", lw=0.6); ax.set_axisbelow(True)
        ax.set_title(title, fontsize=9.5)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels(["mean (mIoU)" if c == "mean" else c.replace("_", " ") for c in rows],
                            fontsize=8.5)
    axes[0].get_yticklabels()[0].set_fontweight("bold")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=8,
               frameon=True, edgecolor="black", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Per-class IoU of the validation-selected ensemble "
                 f"({ev['test']['n_members']} members)", fontsize=10.5, y=0.99)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_perclass.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {FIG / 'fig_perclass.png'}")



def fig_tiling(dpi):
    """แผนภาพการตัดภาพย่อยแบบซ้อนทับ และเหตุที่การสุ่มแบ่งทำให้รั่ว

    เป็นรูปกลไก ไม่ใช่รูปผลลัพธ์ บทความอ้างเรขาคณิตนี้หลายจุดจึงควรเห็นได้
    ค่าทุกค่าอ่านจาก spatial_splits.json
    """
    import matplotlib.patches as mp
    sp = load("spatial_splits.json")
    if not sp:
        return print("ข้าม fig_tiling: ไม่มี spatial_splits.json")
    T, ST = sp["tile_px"], sp["step_px"]
    OV = T - ST

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.0),
                                 gridspec_kw={"width_ratios": [1.15, 1]})

    # ---------- ซ้าย: ภาพย่อยสองใบที่ติดกัน ----------
    h = T * 0.62                                  # วาดเตี้ยลงเพื่อให้เห็นแถบซ้อนชัด
    a1.add_patch(mp.Rectangle((0, 0), T, h, fill=False, lw=1.6, ec="black"))
    a1.add_patch(mp.Rectangle((ST, 0), T, h, fill=False, lw=1.6, ec="black",
                              ls=(0, (5, 3))))
    a1.add_patch(mp.Rectangle((ST, 0), OV, h, fc="0.62", ec="none"))

    def dim(ax, x0, x1, y, label, dy=-26):
        ax.annotate("", xy=(x0, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="<|-|>", lw=1.0, color="black",
                                    mutation_scale=9, shrinkA=0, shrinkB=0))
        for xx in (x0, x1):
            ax.plot([xx, xx], [y - 12, y + 12], lw=0.9, color="black")
        ax.text((x0 + x1) / 2, y + dy, label, ha="center", va="top", fontsize=8.5)

    dim(a1, 0, T, -60, f"tile {T} px")
    dim(a1, 0, ST, -150, f"stride {ST} px")
    dim(a1, ST, ST + OV, h + 95, f"overlap {OV} px", dy=30)
    a1.set_xlim(-70, ST + T + 70)
    a1.set_ylim(-215, h + 195)
    a1.set_aspect("equal"); a1.axis("off")
    a1.set_title("How the tiles are cut", fontsize=10.5, pad=6)

    # ---------- ขวา: ภาพย่อยประเมินกับเพื่อนบ้านแปดใบ ----------
    n_int = sum(len(sp["splits"][k]) for k in ("train", "val", "test"))
    share = len(sp["splits"]["train"]) / n_int
    f = OV / T                                     # สัดส่วนแถบซ้อนของจริง
    # เพื่อนบ้านแปดใบ วาดที่ระยะก้าว 1 หน่วย ภาพย่อยกว้าง 1 หน่วย
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == dc == 0:
                continue
            a2.add_patch(mp.Rectangle((dc - 0.5, dr - 0.5), 1, 1,
                                      fc="0.78", ec="black", lw=0.9))
    # แถบที่ภาพย่อยกลางกินเข้าไปในเพื่อนบ้าน = ขอบนอกของมันเอง
    a2.add_patch(mp.Rectangle((-0.5 - f, -0.5 - f), 1 + 2 * f, 1 + 2 * f,
                              fc="none", ec="black", lw=1.4, ls=(0, (4, 2.5))))
    a2.add_patch(mp.Rectangle((-0.5, -0.5), 1, 1, fc="white", ec="black", lw=2.0))
    a2.text(0, 0, "evaluation\ntile", ha="center", va="center", fontsize=8.5)
    a2.text(0, -1.72, f"its {OV} px margin (dashed) falls inside\n"
                      f"every one of the eight neighbours",
            ha="center", va="top", fontsize=8.5)
    a2.set_xlim(-1.85, 1.85); a2.set_ylim(-2.45, 2.15)
    a2.set_aspect("equal"); a2.axis("off")
    p_none = (1 - share) ** 8
    a2.set_title(f"Why a random tile split leaks\n"
                 f"at a {100 * share:.0f}% training share, P(no training "
                 f"neighbour) \u2248 1 in {1 / p_none:,.0f}",
                 fontsize=10.5, pad=6)
    # legend วางนอกกริด
    a2.add_patch(mp.Rectangle((-1.80, 1.72), 0.22, 0.22, fc="0.78", ec="black", lw=0.9))
    a2.text(-1.52, 1.83, "neighbouring tile", va="center", fontsize=8)
    a2.add_patch(mp.Rectangle((0.28, 1.72), 0.22, 0.22, fc="white", ec="black", lw=1.4))
    a2.text(0.56, 1.83, "tile being scored", va="center", fontsize=8)
    save(fig, "fig_tiling.png", dpi)


def fig_confusion(dpi):
    """เมทริกซ์ความสับสนของ ensemble บนสอง held-out split

    IoU รายคลาสบอกว่าคลาสไหนยาก แต่ไม่บอกว่าความผิดพลาดวิ่งไปทางไหน
    ซึ่งเป็นสิ่งที่ข้อสรุปเรื่องรอยต่อ land use / land cover ต้องใช้
    """
    import numpy as np
    et, ex = load("ensemble_eval_test.json"), load("ensemble_eval_external.json")
    if not et or "confusion_matrix" not in et:
        return print("ข้าม fig_confusion: ยังไม่มี confusion_matrix ในไฟล์ผล")
    panels = [(et, "Internal test split")]
    if ex and "confusion_matrix" in ex:
        panels.append((ex, "External scene"))

    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 4.9))
    if len(panels) == 1:
        axes = [axes]
    for ax, (d, title) in zip(axes, panels):
        cm = np.array(d["confusion_matrix"], dtype=float)
        names = [c.replace("_", " ") for c in d["class_names"]]
        rs = cm.sum(axis=1, keepdims=True)
        row = np.divide(cm, rs, out=np.zeros_like(cm), where=rs > 0)
        ax.imshow(row, cmap=SEQUENTIAL, vmin=0, vmax=1)
        for i in range(len(names)):
            for j in range(len(names)):
                v = row[i, j]
                if v < 0.005:
                    continue
                ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center",
                        fontsize=7.5, color="black", path_effects=HALO if v > 0.45 else None)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("predicted", fontsize=9)
        ax.set_ylabel("reference label", fontsize=9)
        ax.set_title(f"{title}\nrow-normalised %, "
                     f"mIoU {100 * d['mean_iou']:.2f}%", fontsize=10)
        ax.set_xticks(np.arange(-0.5, len(names), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(names), 1), minor=True)
        ax.grid(which="minor", color="0.85", lw=0.5)
        ax.tick_params(which="minor", length=0)
    save(fig, "fig_confusion.png", dpi)


def fig_models(dpi):
    """ผลของทุกโมเดลบนสอง held-out split เทียบกันในภาพเดียว

    เทียบเคียงกับ fig3 ของสตั๊ดดี้ 10 ม.: จุดหนึ่งจุดคือหนึ่งการฝึก
    แกน x คือชุดทดสอบภายใน แกน y คือฉากที่กันไว้ เส้นทแยงคือไม่มีการเสื่อม
    จุดที่อยู่ใกล้เส้นแปลว่าถ่ายทอดข้ามฉากได้ ไม่ใช่จำพื้นที่ฝึก
    """
    import numpy as np
    from collections import defaultdict
    t = load("rerun_eval_test.json")
    x = load("rerun_eval_external.json")
    ens_t, ens_x = load("ensemble_eval_test.json"), load("ensemble_eval_external.json")
    if not t or not x:
        return print("ข้าม fig_models: ไม่มีไฟล์ผล")

    def key(r):
        return r["model"] + "_" + r["encoder"]

    xt = {r["run"]: r["mean_iou"] for r in t["results"]}
    xx = {r["run"]: r["mean_iou"] for r in x["results"]}
    arch_of = {r["run"]: key(r) for r in t["results"]}
    runs = sorted(set(xt) & set(xx))
    by = defaultdict(list)
    for r in runs:
        by[arch_of[r]].append((xt[r], xx[r]))

    order = sorted(by, key=lambda k: -np.mean([a for a, _ in by[k]]))


    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    _all = [c for v in by.values() for pair in v for c in pair]
    if ens_t and ens_x:                       # ensemble อยู่นอกช่วงของ run เดี่ยว
        _all += [ens_t["mean_iou"], ens_x["mean_iou"]]
    lo, hi = 100 * min(_all) - 1.0, 100 * max(_all) + 1.4
    ax.plot([lo, hi], [lo, hi], ls=(0, (5, 4)), color="0.5", lw=1.1, zorder=0)
    ax.text(lo + 0.35, lo + 0.12, "no loss across the scene boundary",
            fontsize=7.5, color="black", rotation=45,
            rotation_mode="anchor", ha="left", va="bottom")
    for i, k in enumerate(order):
        v = np.array(by[k]) * 100
        ax.scatter(v[:, 0], v[:, 1], s=28, marker=_mk(i),
                   facecolor=_sr(i), edgecolor="black", lw=0.6, alpha=0.9,
                   label=f"{NICE.get(_short(k), k)} (n={len(v)})", zorder=3)
    if ens_t and ens_x:
        ax.scatter([100 * ens_t["mean_iou"]], [100 * ens_x["mean_iou"]],
                   s=155, marker="*", facecolor="black", edgecolor="black",
                   lw=0.8, zorder=4)
        ax.annotate(f"ensemble\n({ens_t['n_members']} members)",
                    xy=(100 * ens_t["mean_iou"], 100 * ens_x["mean_iou"]),
                    xytext=(-14, -30), textcoords="offset points",
                    fontsize=8, ha="right",
                    arrowprops=dict(arrowstyle="-", lw=0.8, color="black"))
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("mIoU on the internal test split (test set, %)", fontsize=9.5)
    ax.set_ylabel("mIoU on the wholly held-out second scene\n(external set, %)", fontsize=9.5)
    ax.grid(color="0.9", lw=0.6); ax.set_axisbelow(True)
    ax.legend(fontsize=7.0, frameon=True, edgecolor="black", loc="lower right",
              handletextpad=0.4, borderpad=0.4, labelspacing=0.35)
    ax.set_title(f"Every trained model on both held-out splits\n"
                 f"{len(runs)} runs across {len(order)} architectures; "
                 f"one point per run, read once per split", fontsize=10.5, pad=10)
    save(fig, "fig_models.png", dpi)


def _short(k):
    """ชื่อย่อที่ใช้ใน NICE — ไฟล์ผลเก็บ model_encoder ส่วน NICE ใช้ชื่อสั้น"""
    m = {"segformer_mit_b2": "segformer_b2", "upernet_tu-convnext_tiny": "upernet_convnextt",
         "hrnet_tu-hrnet_w48": "hrnet_w48", "unetpp_efficientnet-b3": "unetpp_effb3",
         "unet_efficientnet-b3": "unet_effb3", "unet_mobilenet_v2": "unet_mobilenet",
         "unet_resnet50": "unet_resnet50", "unet_densenet121": "unet_densenet121",
         "deeplabv3_resnet101": "deeplabv3_r101"}
    return m.get(k, k)


def _ink(bg):
    """สีตัวอักษรที่อ่านออกบนพื้นสีนั้น — เลือกจากความสว่าง ไม่ใช่กำหนดตายตัว

    ขาวบนส้ม #e69f00 (luminance 0.64) อ่านแทบไม่ออก การกำหนดสีตัวอักษรไว้ล่วงหน้า
    จึงพังทันทีที่เปลี่ยนชุดสี
    """
    if not bg.startswith("#"):
        return "white" if float(bg) < 0.55 else "black"
    r, g, b = (int(bg[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.55 else "black"


def save(fig, name, dpi):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {FIG / name}")




def fig_arch(dpi):
    """ค่าเฉลี่ยรายสถาปัตยกรรมบนสอง held-out split — แทนตารางสองตาราง"""
    import statistics as st
    from collections import defaultdict
    data = {}
    for split in ("test", "external"):
        ev = load(f"rerun_eval_{split}.json")
        if not ev:
            return print("ข้าม fig_arch")
        g = defaultdict(list)
        for r in ev["results"]:
            g[r["run"].split("__")[0]].append(r["mean_iou"])
        data[split] = {k: (st.mean(v), st.pstdev(v), len(v)) for k, v in g.items()}
    archs = sorted(data["test"], key=lambda a: -data["test"][a][0])
    y = range(len(archs))
    fig, ax = plt.subplots(figsize=(7.0, 0.42 * len(archs) + 2.0))
    for split, mk, fc, lab in (("test", "o", "black", "Internal test"),
                               ("external", "s", "white", "External (Scene 2)")):
        m = [data[split][a][0] for a in archs]
        s = [data[split][a][1] for a in archs]
        ax.errorbar([100 * v for v in m], list(y), xerr=[100 * v for v in s],
                    fmt=mk, ms=5.5, mfc=fc, mec="black", mew=1.1,
                    ecolor="0.45", elinewidth=1, capsize=2.5, ls="none", label=lab)
    ax.set_yticks(list(y))
    ax.set_yticklabels([NICE.get(a, a) for a in archs], fontsize=8.5)
    ax.set_ylim(-0.6, len(archs) - 0.4)
    ax.set_xlabel("mIoU (%), mean over all conditions and seeds", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(axis="x", color="0.9", lw=0.6); ax.set_axisbelow(True)
    ax.legend(fontsize=8.5, frameon=True, edgecolor="black",
              loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    ax.set_title("Held-out accuracy by architecture\n"
                 "bars span one standard deviation across conditions and seeds",
                 fontsize=10.5, pad=10)
    save(fig, "fig_arch.png", dpi)


def fig_audit(dpi):
    """สองแผงของการตรวจสอบขอบเขตอ้างอิง — แทนตารางเมทริกซ์และตารางองค์ประกอบ"""
    import numpy as np
    lr = (load("label_review_result_round2.json")
          or load("label_review_result.json"))
    g1 = load("from_paper3/g1_adjudication_pilot.json")
    if not (lr and g1):
        return print("ข้าม fig_audit")
    cls = lr["classes"]
    conf = np.array(lr["confusion_label_vs_reviewer"], float)
    row = conf.sum(1, keepdims=True)
    frac = np.divide(conf, row, out=np.zeros_like(conf), where=row > 0)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.9),
                                 gridspec_kw={"width_ratios": [1.35, 1]})
    im = a1.imshow(frac, cmap=SEQUENTIAL, vmin=0, vmax=1)
    a1.set_xticks(range(len(cls))); a1.set_yticks(range(len(cls)))
    a1.set_xticklabels([c.replace("_", " ") for c in cls], rotation=45,
                       ha="right", fontsize=7.5)
    a1.set_yticklabels([c.replace("_", " ") for c in cls], fontsize=7.5)
    a1.set_xlabel("independent judgement", fontsize=8.5)
    a1.set_ylabel("reference label", fontsize=8.5)
    for i in range(len(cls)):
        for j in range(len(cls)):
            if conf[i, j]:
                a1.text(j, i, f"{int(conf[i, j])}", ha="center", va="center",
                        fontsize=7, color="black",
                        path_effects=HALO if frac[i, j] > 0.45 else None)
    a1.set_title("Where the reference and an independent\njudgement disagree "
                 "(row-normalised)", fontsize=9.5)
    fig.colorbar(im, ax=a1, fraction=0.045, pad=0.03).ax.tick_params(labelsize=7)

    dec = g1["disagreement_decomposition"]; tot = sum(dec.values())
    # สีสื่อความหมาย: เขียว = การเปลี่ยนแปลงจริง (สิ่งที่อยากวัด)
    # ส้ม/เทา = ปัญหาของแหล่งอื่นหรือตัดสินไม่ได้ แดง = ความผิดของเฉลยชุดนี้เอง
    order = [("Land actually changed", "real_change", "#2c7a4b"),
             ("2020 layer wrong / definitions", "t0_or_definition", "#e69f00"),
             ("Reviewers did not agree", "undecided", "#9e9e9e"),
             ("Reference here is wrong", "t1_label_error", "#c0392b")]
    left = 0
    for lab, key, col in order:
        w = 100 * dec[key] / tot
        a2.barh(0, w, left=left, color=col, edgecolor="black", lw=0.7)
        if w > 8:
            a2.text(left + w / 2, 0, f"{w:.1f}%", ha="center", va="center",
                    fontsize=8.5, color="black", path_effects=HALO)
        left += w
    a2.set_xlim(0, 100); a2.set_ylim(-0.6, 0.6)
    a2.set_yticks([]); a2.set_xlabel("share of adjudicated disagreement (%)", fontsize=8.5)
    a2.tick_params(labelsize=8)
    a2.legend([plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="black", lw=0.7)
               for _, _, c in order], [l for l, _, _ in order],
              fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.28),
              ncol=2, frameon=True, edgecolor="black")
    a2.set_title(f"What the disagreement is made of\n"
                 f"{g1['n_items']} parcels, two reviewers, "
                 f"kappa {g1['min_kappa_q2']:.2f}", fontsize=9.5)
    save(fig, "fig_audit.png", dpi)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=400)
    a = ap.parse_args()
    fig_tiling(a.dpi)
    fig_effects(a.dpi); fig_blocks(a.dpi); fig_perclass(a.dpi)
    fig_arch(a.dpi); fig_audit(a.dpi); fig_confusion(a.dpi); fig_models(a.dpi)
