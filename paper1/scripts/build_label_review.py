#!/usr/bin/env python3
"""สร้างชุดตรวจสอบคุณภาพ label แบบสุ่มปิดตา — ออกมาเป็นไฟล์ HTML ไฟล์เดียวที่เปิดตรวจได้เลย

ทำไมต้องสุ่มปิดตาแทนที่จะให้โมเดลเสนอจุดแก้
-------------------------------------------
รอบแรกใช้โมเดลเสนอว่าตรงไหนน่าจะ label ผิด แล้วให้คนกดรับ/ไม่รับ วิธีนั้นวัด
"คนเห็นด้วยกับโมเดลแค่ไหน" ไม่ได้วัด "label ผิดแค่ไหน" และมีแนวโน้มดันให้ label ใหม่
เข้าหาคำทำนายของโมเดลเดิม (confirmation bias) ที่สำคัญคือมันมองไม่เห็น label ที่ผิด
ในจุดที่โมเดลไม่ได้ทัก จึงประมาณ false negative ไม่ได้เลย

สคริปต์นี้สุ่มจุดขึ้นมาเองจากทั้งพื้นที่ โดยไม่ใช้โมเดล ผู้ตรวจไม่เห็น label ปัจจุบัน
เห็นแต่ภาพ แล้วตอบว่าตรงกลางกรอบคือคลาสอะไร เอาคำตอบไปเทียบกับ label ทีหลัง จะได้

  * อัตราความผิดพลาดของ label แยกรายคลาส แบบไม่เอนเอียง
  * เมทริกซ์ว่าคลาสไหนถูก label เป็นคลาสไหน
  * ถ้ามีผู้ตรวจมากกว่าหนึ่งคน จะได้ค่าความสอดคล้องระหว่างผู้ตรวจด้วย

สุ่มแบบแบ่งชั้นตามคลาสของ label ปัจจุบัน เพื่อให้คลาสที่มีน้อย (ดินเปล่า 0.9% ของพิกเซล)
ได้จำนวนตัวอย่างพอจะประมาณค่าได้ การถ่วงกลับตามสัดส่วนจริงทำตอนวิเคราะห์

ใช้งาน
------
  python paper1/scripts/build_label_review.py                    # 25 จุด/คลาส
  python paper1/scripts/build_label_review.py --per-class 40 --seed 7
"""
import argparse
import base64
import io
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent.parent
SPLITS = ROOT / "paper1" / "analysis" / "results" / "spatial_splits.json"
TILES = ROOT / "paper2" / "data" / "tiles_falsecolor"      # ชื่อโฟลเดอร์ชวนเข้าใจผิด:
LABELS = ROOT / "paper2" / "data" / "labels_fixed"          # ลำพูนเป็น true colour ที่ยืด contrast
OUT_HTML = ROOT / "paper1" / "reports" / "label_review.html"
OUT_KEY = ROOT / "paper1" / "analysis" / "results" / "label_review_key.json"

CONTEXT_PX = 256      # กรอบบริบท 256 px = 128 ม.
ZOOM_PX = 64          # กรอบซูม 64 px = 32 ม. คือสิ่งที่ถามจริง
MARGIN = CONTEXT_PX // 2


def to_b64(img, size, quality=80):
    img = img.resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


MAX_NODATA = 0.10     # สัดส่วน no-data สูงสุดที่ยอมให้มีในกรอบที่ถาม


def sample_points(classes, per_class, seed, splits):
    """สุ่มจุดแบบแบ่งชั้นตามคลาสของ label ปัจจุบัน

    ข้ามจุดที่กรอบคำถามมี no-data มากเกินไป รอบแรกไม่ได้กรองตรงนี้ ทำให้ 20 จาก 175 จุด
    ตกบนพื้นที่ดำสนิท และคลาส water โดนหนักสุดเพราะน้ำเข้มกับ no-data แยกกันไม่ออก
    ด้วยตาในภาพที่ยืด contrast แล้ว
    """
    rng = random.Random(seed)
    names = sorted({t for v in splits["splits"].values() for t in v})
    rng.shuffle(names)
    picked = defaultdict(list)
    need = {i: per_class for i in range(len(classes))}
    z = ZOOM_PX // 2

    for name in names:
        if all(v <= 0 for v in need.values()):
            break
        lab_path = LABELS / f"{name}.png"
        if not (TILES / f"{name}.png").exists() or not lab_path.exists():
            continue
        lab = np.asarray(Image.open(lab_path))
        if lab.ndim == 3:
            lab = lab[..., 0]
        lab = np.where(lab == 7, 0, lab)          # grass ถูกยุบเข้า background
        img = np.asarray(Image.open(TILES / f"{name}.png").convert("RGB"))
        nodata = img.sum(-1) < 12
        inner = lab[MARGIN:-MARGIN, MARGIN:-MARGIN]
        for ci in range(len(classes)):
            if need[ci] <= 0:
                continue
            ys, xs = np.nonzero(inner == ci)
            if len(ys) == 0:
                continue
            order = list(range(len(ys)))
            rng.shuffle(order)
            for j in order[:40]:              # ลองไม่เกิน 40 จุดต่อ tile ต่อคลาส
                y, x = int(ys[j]) + MARGIN, int(xs[j]) + MARGIN
                if nodata[y - z:y + z, x - z:x + z].mean() > MAX_NODATA:
                    continue
                picked[ci].append({"tile": name, "x": x, "y": y, "label": ci})
                need[ci] -= 1
                break
    return picked, need


def render_item(item):
    tile = np.asarray(Image.open(TILES / f"{item['tile']}.png").convert("RGB"))
    x, y = item["x"], item["y"]
    h = CONTEXT_PX // 2
    ctx = tile[y - h:y + h, x - h:x + h]
    ctx_img = Image.fromarray(ctx)
    d = ImageDraw.Draw(ctx_img)
    z = ZOOM_PX // 2
    d.rectangle([h - z, h - z, h + z, h + z], outline=(255, 255, 0), width=3)

    zh = ZOOM_PX // 2
    zoom = tile[y - zh:y + zh, x - zh:x + zh]
    zoom_img = Image.fromarray(zoom)
    return to_b64(ctx_img, 320), to_b64(zoom_img, 320, quality=88)


HTML = """<!doctype html><html lang="th"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ตรวจคุณภาพ label — THEOS-2 ลำพูน</title>
<style>
 :root{--bg:#12151a;--card:#1c2029;--ink:#e8eaed;--dim:#9aa0a6;--line:#2f3540;--ok:#3fb950}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--ink);
      font-family:-apple-system,"Segoe UI","Sarabun","TH Sarabun New",sans-serif}
 header{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;
        gap:16px;align-items:center;flex-wrap:wrap}
 h1{font-size:17px;margin:0;font-weight:600}
 .bar{flex:1;min-width:180px;height:8px;background:var(--line);border-radius:4px;overflow:hidden}
 .bar > i{display:block;height:100%;background:var(--ok);width:0}
 input{background:var(--card);border:1px solid var(--line);color:var(--ink);
       padding:6px 10px;border-radius:6px;font-size:14px}
 main{max-width:920px;margin:0 auto;padding:20px}
 .imgs{display:flex;gap:16px;justify-content:center;flex-wrap:wrap}
 figure{margin:0;text-align:center}
 figure img{width:320px;height:320px;border-radius:8px;border:1px solid var(--line);
            image-rendering:pixelated;display:block}
 figcaption{font-size:12px;color:var(--dim);margin-top:6px}
 .q{text-align:center;margin:18px 0 10px;font-size:17px}
 .opts{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
 button{background:var(--card);border:1px solid var(--line);color:var(--ink);
        padding:12px;border-radius:8px;font-size:15px;cursor:pointer;text-align:left}
 button:hover{border-color:var(--ok)}
 button b{display:inline-block;width:22px;color:var(--dim)}
 .row{display:flex;gap:10px;margin-top:16px;flex-wrap:wrap;align-items:center}
 .ghost{background:none;border:1px solid var(--line);color:var(--dim);padding:9px 14px;
        font-size:14px;width:auto}
 .note{color:var(--dim);font-size:13px;line-height:1.6}
 .done{text-align:center;padding:40px 0}
 code{background:var(--card);padding:2px 6px;border-radius:4px;font-size:13px}
</style></head><body>
<header>
  <h1>ตรวจคุณภาพ label</h1>
  <div class="bar"><i id="pg"></i></div>
  <span id="cnt" class="note"></span>
  <label class="note">ผู้ตรวจ <input id="rev" size="10" placeholder="ชื่อ/รหัส"></label>
</header>
<main>
 <div id="app"></div>
 <div class="row">
   <button class="ghost" id="back">← ย้อนกลับ</button>
   <button class="ghost" id="skip">ข้าม (ไม่แน่ใจ) — กด 0</button>
   <button class="ghost" id="save">บันทึกผลลงไฟล์</button>
   <span class="note" id="msg"></span>
 </div>
 <p class="note" style="margin-top:22px">
   ตอบว่า <b>ในกรอบเหลือง</b> คือคลาสอะไร โดยดูจากภาพอย่างเดียว —
   ระบบตั้งใจไม่แสดง label ปัจจุบันและไม่แสดงคำทำนายของโมเดล เพื่อไม่ให้คำตอบถูกชักนำ
   ถ้าไม่แน่ใจให้กด 0 ดีกว่าเดา<br>
   คำตอบถูกเก็บในเบราว์เซอร์อัตโนมัติ ปิดแล้วเปิดใหม่ทำต่อได้ เสร็จแล้วกด “บันทึกผลลงไฟล์”
   แล้วส่งไฟล์ JSON กลับมา
 </p>
</main>
<script>
const ITEMS = __ITEMS__, CLASSES = __CLASSES__, SETID = "__SETID__";
const KEY = "label_review_" + SETID;
let ans = JSON.parse(localStorage.getItem(KEY) || "{}");
let i = ITEMS.findIndex((_,k)=>!(k in ans)); if (i < 0) i = ITEMS.length;
const $ = s => document.querySelector(s);
$("#rev").value = localStorage.getItem(KEY+"_rev") || "";
$("#rev").oninput = e => localStorage.setItem(KEY+"_rev", e.target.value);

function draw(){
  const n = Object.keys(ans).length;
  $("#pg").style.width = (100*n/ITEMS.length) + "%";
  $("#cnt").textContent = n + " / " + ITEMS.length;
  if (i >= ITEMS.length){
    $("#app").innerHTML = '<div class="done"><h2>ตรวจครบแล้ว</h2>'
      + '<p class="note">กด “บันทึกผลลงไฟล์” แล้วส่งไฟล์กลับมา</p></div>';
    return;
  }
  const it = ITEMS[i];
  $("#app").innerHTML =
    '<div class="imgs">'
    + '<figure><img src="data:image/jpeg;base64,'+it.ctx+'"><figcaption>บริบท 128 ม. — กรอบเหลืองคือจุดที่ถาม</figcaption></figure>'
    + '<figure><img src="data:image/jpeg;base64,'+it.zoom+'"><figcaption>ซูมเฉพาะในกรอบ 32 ม.</figcaption></figure>'
    + '</div><div class="q">ในกรอบเหลืองคือคลาสอะไร</div><div class="opts">'
    + CLASSES.map((c,k)=>'<button data-k="'+k+'"><b>'+(k+1)+'</b> '+c.th+'</button>').join("")
    + '</div>';
  document.querySelectorAll(".opts button").forEach(b =>
    b.onclick = () => pick(parseInt(b.dataset.k)));
}
function pick(k){
  ans[i] = {answer:k, at:new Date().toISOString()};
  localStorage.setItem(KEY, JSON.stringify(ans)); i++; draw();
}
document.onkeydown = e => {
  if (document.activeElement === $("#rev")) return;
  if (e.key === "0"){ pick(-1); }
  const k = parseInt(e.key);
  if (k >= 1 && k <= CLASSES.length) pick(k-1);
  if (e.key === "Backspace" && i > 0){ i--; draw(); }
};
$("#back").onclick = ()=>{ if(i>0){ i--; draw(); } };
$("#skip").onclick = ()=> pick(-1);
$("#save").onclick = ()=>{
  const out = {set_id:SETID, reviewer:$("#rev").value.trim(),
               saved_at:new Date().toISOString(),
               answers:Object.fromEntries(Object.entries(ans).map(
                 ([k,v])=>[ITEMS[k].id, v]))};
  const blob = new Blob([JSON.stringify(out,null,1)], {type:"application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "label_review_" + (out.reviewer||"anon") + ".json";
  a.click();
  $("#msg").textContent = "บันทึกแล้ว " + Object.keys(ans).length + " ข้อ";
};
draw();
</script></body></html>"""

TH = {"background": "พื้นหลัง / อื่น ๆ", "vegetation": "ไม้ยืนต้น / พุ่มไม้",
      "water": "แหล่งน้ำ", "urban": "อาคาร สิ่งปลูกสร้าง",
      "agriculture": "พื้นที่เกษตร แปลงปลูก", "bare_soil": "ดินเปล่า",
      "road": "ถนน พื้นผิวจราจร"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=25)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=str(OUT_HTML))
    args = ap.parse_args()

    splits = json.loads(SPLITS.read_text())
    classes = splits["classes"]
    picked, need = sample_points(classes, args.per_class, args.seed, splits)

    items, key = [], []
    rng = random.Random(args.seed + 99)
    flat = [p for ci in sorted(picked) for p in picked[ci]]
    rng.shuffle(flat)                     # สลับลำดับ ผู้ตรวจจะเดาคลาสจากลำดับไม่ได้
    for n, p in enumerate(flat):
        ctx, zoom = render_item(p)
        items.append({"id": n, "ctx": ctx, "zoom": zoom})
        key.append({"id": n, **p})

    set_id = f"s{args.seed}p{args.per_class}"
    # เฉลยผูกกับไฟล์ HTML ที่สร้าง ไม่ใช่ path ตายตัว มิฉะนั้นการสร้างชุดทดสอบ
    # จะทับเฉลยของชุดที่ตรวจไปแล้ว และสร้างกลับให้ตรงเดิมไม่ได้ถ้าตัวสุ่มถูกแก้
    out_path = Path(args.out)
    key_path = (OUT_KEY if out_path == OUT_HTML
                else out_path.with_name(out_path.stem + "_key.json"))
    html = (HTML.replace("__ITEMS__", json.dumps(items))
                .replace("__CLASSES__", json.dumps(
                    [{"en": c, "th": TH.get(c, c)} for c in classes],
                    ensure_ascii=False))
                .replace("__SETID__", set_id))
    out = out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")

    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(json.dumps(
        {"set_id": set_id, "seed": args.seed, "per_class": args.per_class,
         "classes": classes, "context_px": CONTEXT_PX, "zoom_px": ZOOM_PX,
         "items": key}, indent=1))

    print(f"จุดที่สุ่มได้ {len(flat)} จุด จาก {len(classes)} คลาส")
    for ci, c in enumerate(classes):
        print(f"  {c:12s} {len(picked[ci]):3d}" +
              ("  (ไม่ครบตามที่ขอ)" if need[ci] > 0 else ""))
    print(f"\nwrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"wrote {key_path}")
    print("\nเปิดไฟล์ HTML ในเบราว์เซอร์เพื่อตรวจ — เฉลยอยู่คนละไฟล์ ผู้ตรวจไม่เห็น")


if __name__ == "__main__":
    main()
