#!/usr/bin/env python3
"""แผนภาพขั้นตอนการดำเนินการศึกษาของ paper1

ภาพเดิม (fig_4_1_methodology_flowchart.png) เป็นของการรันรอบแรกซึ่งใช้ไม่ได้แล้ว
ตัวนี้วาดโปรโตคอลที่ใช้จริง ตัวเลขอ่านจากไฟล์ผล ไม่ได้พิมพ์ลงไป

ใช้งาน
------
  python paper1/scripts/fig_workflow.py
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch    # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "analysis" / "results"
OUT = ROOT / "analysis" / "figure" / "fig_workflow.png"
MD = ROOT / "reports" / "workflow_flowchart.md"


def load(n):
    p = RES / n
    return json.loads(p.read_text()) if p.exists() else None


# รูปทรงของกล่องสื่อความหมายตามธรรมเนียมผังงาน (ISO 5807 ที่วารสาร RS ใช้กัน):
#   สี่เหลี่ยมด้านขนาน = ข้อมูลเข้า/ออก   สี่เหลี่ยม = การดำเนินการ
#   หกเหลี่ยม = จุดควบคุม/ประตู           แคปซูล = ข้อสรุป
SHAPE = {"IMG": "data", "REF": "data", "COUNT": "data",
         "LOCK": "gate", "BOUND": "verdict"}
# ระยะของงานวิจัย ใช้เป็นแถบ (subgraph) เพื่อให้เห็นโครงของระเบียบวิธีก่อนเห็นรายละเอียด
STAGES = [("P1", "Stage 1 - Data", ["IMG", "REF", "TILE"]),
          ("P2", "Stage 2 - Spatial partition", ["SPLIT", "COUNT"]),
          ("P3", "Stage 3 - Factorial experiment", ["FACT"]),
          ("P4", "Stage 4 - Selection and held-out reads", ["SEL", "LOCK"]),
          ("P5", "Stage 5 - Analysis", ["ANALYSIS"]),
          ("P6", "Stage 6 - Reference audit", ["AUDIT"])]


def _node(key, text):
    """กล่องหนึ่งใบ พร้อมรูปทรงตามบทบาทของมัน"""
    kind = SHAPE.get(key, "process")
    if kind == "data":
        return f'    {key}[/"{text}"/]'
    if kind == "gate":
        return f'    {key}{{{{"{text}"}}}}'
    if kind == "verdict":
        return f'    {key}(["{text}"])'
    return f'    {key}["{text}"]'


def mermaid_flat(nodes):
    """ผังชุดเดียวกันแบบที่ draw.io นำเข้าได้

    ตัวนำเข้า mermaid ของ draw.io รองรับไวยากรณ์ชุดเล็กกว่าที่ GitHub/VS Code รองรับ
    เจอ `subgraph` แล้วตอบว่า "Unsupported diagram type: subgraph" จึงต้องมีอีกชุด
    ที่ไม่มี subgraph ไม่มี classDef/style และใช้เฉพาะกล่องสี่เหลี่ยม
    ข้อมูลระยะไม่ได้หายไป แต่ย้ายไปเป็นป้าย S1..S6 หน้าข้อความของแต่ละกล่องแทน
    """
    # ป้ายระยะเป็นข้อความล้วน ไม่ใช้ entity อย่าง &nbsp; เพราะตัวนำเข้าของ draw.io
    # ไม่ได้แปลง HTML entity ให้ทุกกรณี
    stage_of = {k: title.split(" - ")[0] for _, title, keys in STAGES for k in keys}
    lines = ["flowchart TD"]
    for key, text in nodes:
        tag = stage_of.get(key)
        label = f"[{tag}] {text}" if tag else text
        lines.append(f'    {key}["{label}"]')
    lines += [
        "    IMG --> TILE",
        "    REF --> TILE",
        "    TILE --> SPLIT",
        "    SPLIT --> COUNT",
        "    COUNT --> FACT",
        "    FACT --> SEL",
        "    FACT --> LOCK",
        "    SEL --> LOCK",
        "    SEL --> ANALYSIS",
        "    LOCK --> ANALYSIS",
        "    ANALYSIS --> AUDIT",
        "    AUDIT --> BOUND",
    ]
    return "\n".join(lines)


def mermaid(nodes, grey):
    """ผังเดียวกับภาพ แต่เป็น mermaid สำหรับอ่านใน .md

    ตัวเลขส่งเข้ามาจากผู้เรียกซึ่งอ่านจากไฟล์ผลชุดเดียวกับภาพ ไม่ได้พิมพ์ซ้ำที่นี่
    จัดกลุ่มเป็นระยะตามธรรมเนียมผังระเบียบวิธี ผู้อ่านจะได้เห็นโครงหกระยะก่อน
    แล้วค่อยอ่านรายละเอียดในกล่อง แทนที่จะเจอกล่องยาวสิบเอ็ดใบเรียงกันเฉย ๆ
    """
    txt = dict(nodes)
    placed = {k for _, _, ks in STAGES for k in ks}
    lines = ["flowchart TD"]
    for sid, title, keys in STAGES:
        lines.append(f'    subgraph {sid}["{title}"]')
        lines += ["    " + _node(k, txt[k]).lstrip() for k in keys if k in txt]
        lines.append("    end")
    for key, text in nodes:
        if key not in placed:
            lines.append(_node(key, text))
    lines += [
        "    IMG --> TILE",
        "    REF --> TILE",
        "    TILE --> SPLIT",
        "    SPLIT --> COUNT",
        "    COUNT --> FACT",
        "    FACT --> SEL",
        "    FACT --> LOCK",
        "    SEL --> LOCK",
        "    SEL --> ANALYSIS",
        "    LOCK --> ANALYSIS",
        "    ANALYSIS --> AUDIT",
        "    AUDIT --> BOUND",
        "    classDef keystep fill:#e8e8e8,stroke:#000,stroke-width:2px;",
        "    classDef verdict fill:#d8d8d8,stroke:#000,stroke-width:2px;",
        f"    class {','.join(grey)} keystep;",
        "    class BOUND verdict;",
    ] + [
        f"    style {s[0]} fill:none,stroke:#999,stroke-width:1px,"
        f"stroke-dasharray:4 3" for s in STAGES
    ] + [
    ]
    return "\n".join(lines)


MMD = ROOT / "analysis" / "figure" / "fig_workflow.mmd"


def mermaid_figure(f):
    """ผังเจ็ดระยะแบบ mermaid — ข้อความและตัวเลขชุดเดียวกับภาพ matplotlib

    render เป็น PNG ด้วย paper1/scripts/render_mermaid.sh (mermaid-cli) แล้วใช้เป็นรูปในบทความ
    ตัวหนังสือทั้งหมดเป็นสีดำ กล่องเทาคือจุดควบคุมสามจุด
    """
    def node(k, title, body=""):
        lab = f"<b>{title}</b>" + (f"<br/>{body}" if body else "")
        return f'        {k}["{lab}"]'
    L = ["%%{init: {'theme': 'base', 'flowchart': {'curve': 'linear', 'nodeSpacing': 28, "
         "'rankSpacing': 34, 'htmlLabels': true, 'padding': 10, 'wrappingWidth': 900}, "
         "'themeCSS': '.nodeLabel, .cluster-label, .label, span, p, b {font-family: Arial, Helvetica, "
         "sans-serif !important; color: #000000 !important; fill: #000000 !important;}', "
         "'themeVariables': {"
         "'fontFamily': 'Arial, Helvetica, sans-serif', 'fontSize': '15px', "
         "'primaryColor': '#ffffff', 'primaryTextColor': '#000000', 'primaryBorderColor': '#000000', "
         "'lineColor': '#000000', 'clusterBkg': '#ffffff', 'clusterBorder': '#9a9a9a', "
         "'titleColor': '#000000', 'edgeLabelBackground': '#ffffff'}}}%%",
         "flowchart TB"]
    st = [
        ("S1", "1 &nbsp;Data", "LR", [
            node("IMG", "THEOS-2 imagery", f"two scenes, 0.5 m, R G B NIR<br/>8 January 2026"),
            node("REF", "Reference masks", "drawn on the same imagery"),
            node("TILE", "Tiling", f"{f['tile']} px tiles, {f['step']} px stride<br/>"
                                   f"{f['tile'] - f['step']} px overlap per side")]),
        ("S2", "2 &nbsp;Spatial partition", "LR", [
            node("SPLIT", "Spatial split", f"{f['block']:.0f} m blocks, one-tile buffer "
                                            f"({f['nb']} tiles dropped)<br/>Scene 2 held out as the external set"),
            node("COUNT", "Split sizes", f"train {f['n']['train']:,} · validation {f['n']['val']:,} · "
                                         f"test {f['n']['test']:,} · external {f['n']['external']:,}<br/>"
                                         f"{f['leak']}")]),
        ("S3", "3 &nbsp;Factorial experiment", "LR", [
            node("FACT", f"{f['ncond']} conditions × {f['narch']} architectures × {f['nseed']} seeds "
                         f"= {f['ntask']} runs",
                 "input bands (RGB, RGB + NIR) × augmentation (off, on) × class weighting (off, on)"
                 "<br/>each effect estimated once per seed in every architecture")]),
        ("S4", "4 &nbsp;Selection and held-out reads", "LR", [
            node("SEL", "Ensemble selection", "on the validation split only<br/>greedy forward selection"),
            node("LOCK", "Held-out sets locked", "test read once, external read once<br/>"
                                                 "after every choice is frozen")]),
        ("S5", "5 &nbsp;Analysis", "LR", [
            node("EFF", "Factor effects", "95% intervals over matched seeds"),
            node("UNC", "Spatial uncertainty", "paired bootstrap over spatial blocks<br/>"
                                               "per-class IoU, confusion, full-scene map")]),
        ("S6", "6 &nbsp;Reference audit", "LR", [
            node("AUDIT", "Audit of the reference labels",
                 "blinded point sample read by two reviewers" + (f" (kappa {f['kappa']:.2f})" if f['kappa'] else "")
                 + "<br/>parcel adjudication against the 2020 national land-use layer")]),
        ("S7", "7 &nbsp;Pre-registered additions", "LR", [
            node("A1", "Random-split<br/>control"), node("A2", "Transfer to a<br/>second province"),
            node("A3", "Training without<br/>background"),
            node("A4", "Encoders pretrained<br/>on overhead imagery"),
            node("A5", "Near-infrared<br/>input tests")]),
    ]
    for sid, title, d, nodes in st:
        L.append(f'    subgraph {sid}["<b>{title}</b>"]')
        L.append(f"        direction {d}")
        L += nodes
        L.append("    end")
    L += ["    IMG --> REF --> TILE", "    SPLIT --> COUNT", "    SEL --> LOCK",
          "    S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7",
          "    A1 ~~~ A2 ~~~ A3 ~~~ A4 ~~~ A5", "    EFF ~~~ UNC",
          "    classDef box fill:#ffffff,stroke:#000000,stroke-width:1px,color:#000000;",
          "    classDef key fill:#dedede,stroke:#000000,stroke-width:2.2px,color:#000000;",
          "    class IMG,REF,TILE,COUNT,FACT,SEL,EFF,UNC,A1,A2,A3,A4,A5 box;",
          "    class SPLIT,LOCK,AUDIT key;"]
    L += [f"    style {sid} fill:#ffffff,stroke:#9a9a9a,stroke-width:1px,color:#000000"
          for sid, *_ in st]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--md", default=str(MD))
    a = ap.parse_args()

    sp = load("spatial_splits.json") or {}
    tl = load("rerun_tasklist.json") or {}
    lr = load("label_review_result_round2.json") or {}
    n = {k: len(v) for k, v in sp.get("splits", {}).items()}
    af = sp.get("leakage_audit", {}).get("after_buffer", {})
    nb = sum(len(v) for v in (sp.get("buffer_dropped") or {}).values())
    ntask = len(tl.get("tasks", []))
    narch = len({t["model"] + "_" + t["encoder"] for t in tl.get("tasks", [])})
    nseed = len({t["seed"] for t in tl.get("tasks", [])})
    ncond = len({t["condition"] for t in tl.get("tasks", [])})
    # จำนวนที่วิเคราะห์จริง อ่านจากไฟล์เดียวกับตารางในบทความ ไม่ใช่จากแผน
    # ถ้าแฟกทอเรียลยังรันไม่ครบ ภาพต้องบอกตามนั้น ไม่ใช่โฆษณาตัวเลขแผน
    cells = load("rerun_cells_mean_iou.json") or []
    ndone = sum(c.get("n", 0) for c in cells)

    _tt = "  ".join(f"{k} {v:,}" for k, v in
                    (("train", n.get("train", 0)), ("val", n.get("val", 0)),
                     ("test", n.get("test", 0)), ("external", n.get("external", 0))))
    _leak = ", ".join(f"{k} {v['touching_train']}/{v['tiles']}"
                      for k, v in af.items()) if af else ""
    _k = (lr.get("inter_reviewer") or {})
    _kv = list(_k.values())[0]["cohens_kappa"] if _k else None

    # ข้อความของทุกกล่อง เขียนที่เดียว ใช้ทั้งภาพและผัง mermaid
    txt = {
        "IMG": "THEOS-2, two scenes 0.5 m\nR,G,B,NIR — 8 Jan 2026",
        "REF": ("Reference masks\ndrawn on the same image\n"
                "in a purpose-built annotation tool"),
        "TILE": (f"Tiles {sp.get('tile_px', 512)} px at a {sp.get('step_px', 448)} px "
                 f"stride — {sp.get('tile_px', 512) - sp.get('step_px', 448)} px "
                 f"overlap on every side\ncontext at tile edges, more examples "
                 f"— and dependence between neighbours"),
        "SPLIT": (f"SPATIAL SPLIT — the step the study exists to get right\n"
                  f"whole {sp.get('block_size_m', 1500):.0f} m blocks assigned at "
                  f"random (seed {sp.get('seed', '')}), stratified by landscape\n"
                  f"one-tile buffer carved between splits — {nb} tiles given up\n"
                  f"scene 2 held out entirely as the external set"),
        "COUNT": f"{_tt}\nleakage audit after buffering — {_leak}",
        "FACT": (f"FACTORIAL — {ncond} conditions x {narch} architectures x "
                 f"{nseed} seeds = {ntask} runs"
                 + (f"  ({ndone} analysed here)" if 0 < ndone < ntask else "") + "\n"
                 f"bands (RGB / RGB-NIR) x augmentation (off / on) x "
                 f"class weighting (off / on)\n"
                 f"every factor estimated once per seed in every architecture,\n"
                 f"so an effect is a distribution rather than one difference"),
        "SEL": ("Selection — validation only\ngreedy forward ensemble;\n"
                "size is an outcome, not a setting"),
        "LOCK": ("Held-out sets LOCKED\ntest read once, external once;\n"
                 "unlock needs a recorded reason"),
        "ANALYSIS": ("ANALYSIS — intervals, not point differences\n"
                     "factor effects with 95% CI over matched seeds; "
                     "paired block bootstrap (10,000 resamples)\n"
                     "on the spatial block, because tiles inside a block share "
                     "landscape\nper-class IoU, confusion, full-scene map and its "
                     "inference cost"),
        "AUDIT": ("REFERENCE AUDIT — complementary assessments\n"
                  "blinded point sample, two reviewers"
                  + (f" (kappa {_kv:.2f})" if _kv else "") + "\n"
                  "parcel adjudication against the 2020 national land-use layer\n"
                  "cell-level comparison of the two products"),
        "BOUND": ("Interpret model performance alongside\n"
                  "reference-label uncertainty"),
    }

    short = {
        "IMG": "THEOS-2 imagery\ntwo scenes, 0.5 m, RGB+NIR",
        "REF": "Reference masks\npurpose-built annotation tool",
        "TILE": (f"Tiling\n{sp.get('tile_px', 512)} px, {sp.get('step_px', 448)} px "
                 f"stride"),
        "SPLIT": (f"SPATIAL SPLIT\n{sp.get('block_size_m', 1500):.0f} m blocks "
                  f"+ one-tile buffer"),
        "COUNT": (f"train {n.get('train', 0):,} · val {n.get('val', 0):,} · "
                  f"test {n.get('test', 0):,} · external {n.get('external', 0):,}\n"
                  f"leakage 0 after buffering"),
        "FACT": (f"FACTORIAL\n{ncond} × {narch} × {nseed} = {ntask} runs"),
        "SEL": "Ensemble selection\nvalidation only",
        "LOCK": "Held-out sets LOCKED\ntest once, external once",
        "ANALYSIS": "ANALYSIS\neffect CIs + block bootstrap",
        "AUDIT": "REFERENCE AUDIT\ncomplementary assessments",
        "BOUND": "Interpret performance alongside\nreference-label uncertainty",
    }

    # ---- ภาพ: เจ็ดระยะ แต่ละกล่องมีหัวข้อตัวหนาและข้อเท็จจริงหนึ่งถึงสองบรรทัด ----
    # รายละเอียดเต็มของแต่ละขั้นอยู่ในเนื้อความและใน workflow_flowchart.md
    # กล่องพื้นเทาคือจุดควบคุมสามจุด: การแบ่งเชิงพื้นที่ การล็อกชุดที่กันไว้ การตรวจข้อมูลอ้างอิง
    KEY = "0.88"
    tile, step = sp.get("tile_px", 512), sp.get("step_px", 448)
    fig = plt.figure(figsize=(7.2, 7.5), dpi=a.dpi)
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    X0, X1 = 17.5, 99.0                      # ขอบซ้าย-ขวาของพื้นที่กล่อง

    def box(x, y, w, h, title, body="", fc="white", lw=0.9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.9",
                                    facecolor=fc, edgecolor="black", lw=lw))
        if body:
            ax.text(x + w / 2, y + h - 1.55, title, ha="center", va="top",
                    fontsize=7.6, fontweight="bold")
            ax.text(x + w / 2, y + h - 4.2, body, ha="center", va="top", fontsize=6.7,
                    linespacing=1.25)
        else:
            ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                    fontsize=7.6, fontweight="bold")

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=8, color="black", lw=0.9,
                                     shrinkA=0, shrinkB=0))

    def stage(num, title, y, h):
        """ป้ายระยะทางซ้าย และเส้นคั่นบาง ๆ ใต้ระยะ"""
        ax.text(1.0, y + h / 2, str(num), ha="left", va="center", fontsize=13,
                fontweight="bold", color="black")
        ax.text(4.6, y + h / 2, title, ha="left", va="center", fontsize=7.4,
                color="black", linespacing=1.15)

    W = X1 - X0
    half = (W - 3.0) / 2

    # 1 Data
    y = 88.5; h = 9.0
    stage(1, "Data", y, h)
    w3 = (W - 8.0) / 3
    box(X0, y, w3, h, "THEOS-2 imagery", "two scenes, 0.5 m, R G B NIR\n8 January 2026")
    box(X0 + w3 + 4, y, w3, h, "Reference masks", "drawn on the same imagery")
    box(X0 + 2 * (w3 + 4), y, w3, h, "Tiling",
        f"{tile} px tiles, {step} px stride\n{tile - step} px overlap per side")
    arrow(X0 + w3 + 0.3, y + h / 2, X0 + w3 + 3.7, y + h / 2)
    arrow(X0 + 2 * w3 + 4.3, y + h / 2, X0 + 2 * w3 + 7.7, y + h / 2)

    # 2 Spatial partition
    y2 = 73.5
    stage(2, "Spatial\npartition", y2, h)
    box(X0, y2, half, h, "Spatial split",
        f"{sp.get('block_size_m', 1500):.0f} m blocks, one-tile buffer ({nb} tiles dropped)\n"
        f"Scene 2 held out as the external set", fc=KEY, lw=1.4)
    box(X0 + half + 3, y2, half, h, "Split sizes",
        f"train {n.get('train', 0):,} · validation {n.get('val', 0):,} · "
        f"test {n.get('test', 0):,} · external {n.get('external', 0):,}\n"
        + ("evaluation tiles touching training: 0" if af and not any(
            v["touching_train"] for v in af.values()) else f"leakage audit: {_leak}"))
    # จาก Tiling ลงมาเข้าตรงกลางของระยะที่ 2 เหมือนลูกศรระหว่างระยะอื่น
    xt = X0 + 2 * w3 + 8 + w3 / 2
    ax.plot([xt, xt, X0 + half / 2], [y - 0.3, y2 + h + 3.2, y2 + h + 3.2], color="black", lw=0.9)
    arrow(X0 + half / 2, y2 + h + 3.2, X0 + half / 2, y2 + h + 0.3)
    arrow(X0 + half + 0.3, y2 + h / 2, X0 + half + 2.7, y2 + h / 2)

    # 3 Factorial experiment
    y3 = 58.5
    stage(3, "Factorial\nexperiment", y3, h)
    box(X0, y3, W, h,
        f"{ncond} conditions × {narch} architectures × {nseed} seeds = {ntask} runs"
        + (f" ({ndone} analysed)" if 0 < ndone < ntask else ""),
        "input bands (RGB, RGB + NIR) × augmentation (off, on) × class weighting (off, on)\n"
        "each effect estimated once per seed in every architecture")
    arrow(X0 + half + 3 + half / 2, y2 - 0.3, X0 + half + 3 + half / 2, y3 + h + 0.3)

    # 4 Selection and held-out reads
    y4 = 43.5
    stage(4, "Selection and\nheld-out reads", y4, h)
    box(X0, y4, half, h, "Ensemble selection", "on the validation split only\ngreedy forward selection")
    box(X0 + half + 3, y4, half, h, "Held-out sets locked",
        "test read once, external read once\nafter every choice is frozen", fc=KEY, lw=1.4)
    arrow(X0 + half / 2, y3 - 0.3, X0 + half / 2, y4 + h + 0.3)
    arrow(X0 + half + 0.3, y4 + h / 2, X0 + half + 2.7, y4 + h / 2)

    # 5 Analysis
    y5 = 28.5
    stage(5, "Analysis", y5, h)
    box(X0, y5, half, h, "Factor effects", "95% intervals over matched seeds")
    box(X0 + half + 3, y5, half, h, "Spatial uncertainty",
        "paired bootstrap over spatial blocks\nper-class IoU, confusion, full-scene map")
    # ผลของปัจจัยมาจากรอบการฝึก ส่วนความไม่แน่นอนเชิงพื้นที่มาจากชุดที่กันไว้
    arrow(X0 + half / 2, y4 - 0.3, X0 + half / 2, y5 + h + 0.3)
    arrow(X0 + half + 3 + half / 2, y4 - 0.3, X0 + half + 3 + half / 2, y5 + h + 0.3)

    # 6 Reference audit
    y6 = 13.5
    stage(6, "Reference\naudit", y6, h)
    box(X0, y6, W, h, "Audit of the reference labels",
        "blinded point sample read by two reviewers"
        + (f" (kappa {_kv:.2f})" if _kv else "")
        + "\nparcel adjudication against the 2020 national land-use layer", fc=KEY, lw=1.4)
    arrow(X0 + W / 2, y5 - 0.3, X0 + W / 2, y6 + h + 0.3)

    # 7 Pre-registered additions
    y7 = 0.8; h7 = 8.2
    stage(7, "Pre-registered\nadditions", y7, h7)
    items = [("Random-split\ncontrol", ""), ("Transfer to a\nsecond province", ""),
             ("Training without\nbackground", ""), ("Encoders pretrained\non overhead imagery", ""),
             ("Near-infrared\ninput tests", "")]
    wi = (W - 4 * 1.6) / len(items)
    for i, (ti, _) in enumerate(items):
        box(X0 + i * (wi + 1.6), y7, wi, h7, "")
        ax.text(X0 + i * (wi + 1.6) + wi / 2, y7 + h7 / 2, ti, ha="center", va="center",
                fontsize=6.9, linespacing=1.2)
    arrow(X0 + W / 2, y6 - 0.3, X0 + W / 2, y7 + h7 + 0.3)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {a.out}")

    MMD.write_text(mermaid_figure({
        "tile": tile, "step": step, "block": sp.get("block_size_m", 1500), "nb": nb, "n": n,
        "leak": ("evaluation tiles touching training: 0" if af and not any(
            v["touching_train"] for v in af.values()) else f"leakage audit: {_leak}"),
        "ncond": ncond, "narch": narch, "nseed": nseed, "ntask": ntask, "kappa": _kv}) + "\n")
    print(f"wrote {MMD}")

    order = ["IMG", "REF", "TILE", "SPLIT", "COUNT", "FACT", "SEL", "LOCK",
             "ANALYSIS", "AUDIT", "BOUND"]
    nodes = [(k, short[k].replace("\n", "<br/>")) for k in order]
    # คำอธิบายใต้ผัง ถือข้อความเต็มชุดเดียวกับภาพ เรียงตามระยะ
    import re as _re
    name = {"IMG": "Imagery", "REF": "Reference masks", "TILE": "Tiling",
            "SPLIT": "Spatial split", "COUNT": "Split sizes and leakage audit",
            "FACT": "Factorial design", "SEL": "Ensemble selection",
            "LOCK": "Held-out lock", "ANALYSIS": "Analysis",
            "AUDIT": "Reference audit", "BOUND": "Verdict"}

    def _body(k):
        """ข้อความเต็มของกล่อง ตัดหัวเรื่องตัวพิมพ์ใหญ่ที่ซ้ำกับชื่อรายการออก"""
        s = " ".join(txt[k].split("\n"))
        return _re.sub(r"^[A-Z][A-Z ]{3,}\u2014 ", "", s)

    key_lines = []
    for sid, title, keys in STAGES:
        key_lines.append(f"**{title}**\n")
        for k in keys:
            key_lines.append(f"- **{name[k]}** — {_body(k)}")
        key_lines.append("")
    key_lines.append(f"**{name['BOUND']}** — {_body('BOUND')}")
    chart_key = "\n".join(key_lines)
    md = Path(a.md)
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(
        "# How the study was carried out\n\n"
        "<!-- สร้างโดย paper1/scripts/fig_workflow.py — อย่าแก้ด้วยมือ\n"
        "     ข้อความและตัวเลขชุดเดียวกับ analysis/figure/fig_workflow.png\n"
        "     ซึ่งอ่านจาก analysis/results/ รันสคริปต์ใหม่เมื่อผลเปลี่ยน -->\n\n"
        "```mermaid\n" + MMD.read_text() + "```\n\n"
        "<details><summary>ผังชุดเดียวกันสำหรับนำเข้า draw.io "
        "(ตัวนำเข้าของ draw.io ไม่รองรับ <code>subgraph</code> — "
        "ระยะย้ายไปเป็นป้าย S1..S6 หน้าข้อความแทน)</summary>\n\n"
        "```mermaid\n" + mermaid_flat(nodes) + "\n```\n\n</details>\n\n"
        "## รายละเอียดของแต่ละกล่อง\n\n" + chart_key + "\n\n"
        "**การอ่านผัง** แถบประคือระยะของงานวิจัย เรียงหนึ่งถึงหก "
        "รูปทรงของกล่องบอกบทบาท: สี่เหลี่ยมด้านขนาน = ข้อมูลเข้าหรือออก, "
        "สี่เหลี่ยม = การดำเนินการ, หกเหลี่ยม = จุดควบคุมที่ห้ามข้าม, "
        "แคปซูล = ข้อสรุปของงาน "
        "กล่องพื้นเทาคือขั้นที่ทำให้ตัวเลขที่รายงานมีความหมายตามที่บอก "
        "ได้แก่ การแบ่งข้อมูลเชิงพื้นที่ การล็อกชุดที่กันไว้ "
        "และการตรวจสอบขอบเขตอ้างอิง\n")
    print(f"wrote {md}")


if __name__ == "__main__":
    main()
