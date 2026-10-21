#!/usr/bin/env python3
"""สร้างหน้าตรวจ label รอบใหม่จาก key เดิม พร้อมฝังคู่มือนิยามคลาส

ใช้ "จุดเดิม" ไม่ใช่จุดใหม่ เพราะสิ่งที่ต้องการวัดคือผลของคู่มือ ถ้าเปลี่ยนชุดจุดด้วย
จะแยกไม่ออกว่า agreement ที่ดีขึ้นมาจากคู่มือหรือมาจากจุดที่ง่ายกว่า

ตัวสุ่มถูกแก้ไปแล้วหลังรอบแรก (เพิ่มการกรอง no-data) การเรียก build_label_review.py
ด้วย seed เดิมจึงได้จุดคนละชุด — สคริปต์นี้อ่าน key ที่บันทึกไว้แทน

ใช้งาน
------
  python paper1/scripts/rebuild_label_review.py --reviewers r1 r2
"""
import argparse, base64, io, json, re, html as H
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
TILES = ROOT.parent / "paper2" / "data" / "tiles_falsecolor"
KEY = ROOT / "analysis" / "results" / "label_review_key.json"
MANUAL = ROOT / "reports" / "annotation_manual_th.md"
TH = {"background": "พื้นหลัง / อื่น ๆ", "vegetation": "ไม้ยืนต้น / พุ่มไม้",
      "water": "แหล่งน้ำ", "urban": "อาคาร สิ่งปลูกสร้าง",
      "agriculture": "พื้นที่เกษตร แปลงปลูก", "bare_soil": "ดินเปล่า",
      "road": "ถนน พื้นผิวจราจร"}


def md2html(t):
    out = []
    for line in t.split("\n"):
        s = line.rstrip()
        if s.startswith("### "):  out.append(f"<h4>{H.escape(s[4:])}</h4>")
        elif s.startswith("## "): out.append(f"<h3>{H.escape(s[3:])}</h3>")
        elif s.startswith("> "):  out.append(f"<blockquote>{H.escape(s[2:])}</blockquote>")
        elif s.startswith("|"):   out.append(s)
        elif s.startswith("---"): out.append("<hr>")
        elif s.strip():           out.append(f"<p>{H.escape(s)}</p>")
    h = "\n".join(out)
    def tbl(m):
        rows = [r for r in m.group(0).strip().split("\n") if r.strip()]
        rows = [r for r in rows if not re.match(r"^\|[\s:|-]+\|$", r)]
        c = [[x.strip() for x in r.strip("|").split("|")] for r in rows]
        head = "".join(f"<th>{H.escape(x)}</th>" for x in c[0])
        body = "".join("<tr>" + "".join(f"<td>{H.escape(x)}</td>" for x in r) + "</tr>"
                       for r in c[1:])
        return f"<table><tr>{head}</tr>{body}</table>"
    return re.sub(r"(?:^\|.*\n?)+", tbl, h, flags=re.M)


def b64(img, size, q=82):
    buf = io.BytesIO()
    img.resize((size, size), Image.LANCZOS).convert("RGB").save(buf, "JPEG", quality=q)
    return base64.b64encode(buf.getvalue()).decode()


def chips(item, ctx_px, zoom_px):
    t = np.asarray(Image.open(TILES / f"{item['tile']}.png").convert("RGB"))
    x, y, h, z = item["x"], item["y"], ctx_px // 2, zoom_px // 2
    ctx = Image.fromarray(t[y - h:y + h, x - h:x + h])
    ImageDraw.Draw(ctx).rectangle([h - z, h - z, h + z, h + z],
                                  outline=(255, 255, 0), width=3)
    return b64(ctx, 320), b64(Image.fromarray(t[y - z:y + z, x - z:x + z]), 320, 88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default=str(KEY))
    ap.add_argument("--reviewers", nargs="+", default=["r1", "r2"])
    ap.add_argument("--set-id", default=None)
    ap.add_argument("--out-dir", default=str(ROOT / "reports" / "label_review_round2"))
    a = ap.parse_args()

    key = json.loads(Path(a.key).read_text())
    set_id = a.set_id or key["set_id"] + "m"        # m = with manual
    classes = key["classes"]
    print(f"จุดเดิม {len(key['items'])} จุด | set_id ใหม่ = {set_id}")

    items = []
    for it in key["items"]:
        c, z = chips(it, key["context_px"], key["zoom_px"])
        items.append({"id": it["id"], "ctx": c, "zoom": z})

    manual = MANUAL.read_text(encoding="utf-8")
    core = manual[manual.index("## 0."):manual.index("## 5.")]
    panel = ('<details id="manual" open><summary><b>คู่มือการจำแนกคลาส — เปิดค้างไว้ได้ '
             '(กด M ซ่อน/แสดง)</b></summary><div class="manual">'
             + md2html(core) + "</div></details>")

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    src = (ROOT / "reports" / "label_review.html").read_text(encoding="utf-8")
    css = """
#manual{margin:14px 0;border:1px solid #2f3540;border-radius:8px;background:#171b22;padding:10px 14px}
#manual summary{cursor:pointer;font-size:14px}
.manual{max-height:44vh;overflow:auto;font-size:13px;line-height:1.65;padding-right:8px}
.manual h3{font-size:14px;margin:12px 0 4px}.manual h4{font-size:13px;margin:10px 0 3px;color:#9aa0a6}
.manual table{border-collapse:collapse;margin:6px 0;width:100%}
.manual th,.manual td{border:1px solid #2f3540;padding:3px 7px;text-align:left;font-size:12px}
.manual blockquote{margin:6px 0;padding:5px 10px;border-left:3px solid #3fb950;color:#9aa0a6}
.manual hr{border:0;border-top:1px solid #2f3540;margin:10px 0}
"""
    for rv in a.reviewers:
        s = src
        s = re.sub(r"const ITEMS = \[.*?\], CLASSES",
                   "const ITEMS = " + json.dumps(items) + ", CLASSES", s, flags=re.S)
        s = re.sub(r'SETID = "[^"]*"', f'SETID = "{set_id}"', s)
        s = s.replace("</style>", css + "</style>", 1)
        m = re.search(r'<div id="app">', s)
        s = s[:m.start()] + panel + "\n" + s[m.start():]
        s = s.replace('$("#rev").value = localStorage.getItem(KEY+"_rev") || "";',
                      f'$("#rev").value = localStorage.getItem(KEY+"_rev") || "{rv}";')
        s = s.replace("</body>",
                      "<script>document.addEventListener('keydown',e=>{"
                      "if((e.key==='m'||e.key==='M')&&!/INPUT|TEXTAREA/.test("
                      "document.activeElement.tagName)){const d="
                      "document.getElementById('manual');if(d)d.open=!d.open;}});"
                      "</script></body>", 1)
        p = out / f"review_{rv}.html"
        p.write_text(s, encoding="utf-8")
        print(f"  wrote {p} ({p.stat().st_size/1e6:.1f} MB)")
    (out / "key.json").write_text(json.dumps({**key, "set_id": set_id}, indent=1))
    print(f"  wrote {out / 'key.json'}  (เฉลยเดิม จุดเดิม)")


if __name__ == "__main__":
    main()
