#!/usr/bin/env python3
"""วิเคราะห์ผลตรวจ label แบบสุ่มปิดตา จากไฟล์ที่ผู้ตรวจส่งกลับมา

ให้ตัวเลข 3 ชุดที่ reviewer ของบทความขอไว้:

  1. อัตราความผิดพลาดของ label แยกรายคลาส พร้อมช่วงความเชื่อมั่น (Wilson)
     เนื่องจากสุ่มแบบแบ่งชั้น อัตรารวมทั้งภาพต้องถ่วงกลับตามสัดส่วนพิกเซลจริง
     ไม่ใช่เฉลี่ยตรง ๆ ของทุกคลาส
  2. เมทริกซ์ว่า label ปัจจุบันเป็นคลาสอะไร แล้วคนตอบว่าเป็นคลาสอะไร
     ซึ่งบอกทิศทางของความผิดพลาด ไม่ใช่แค่ปริมาณ
  3. ความสอดคล้องระหว่างผู้ตรวจ (Cohen's kappa) ถ้ามีไฟล์จากมากกว่าหนึ่งคน
     พร้อมรายการข้อที่ผู้ตรวจไม่ตรงกัน เพื่อเอาไปตัดสินร่วมกัน

ข้อที่ผู้ตรวจกด "ไม่แน่ใจ" ถูกนับแยก ไม่เอาไปรวมเป็นถูกหรือผิด และรายงานสัดส่วนไว้
เพราะสัดส่วนที่สูงเองก็เป็นผลลัพธ์ — แปลว่าเกณฑ์ของคลาสนั้นตัดสินยากจากภาพอย่างเดียว

ใช้งาน
------
  python paper1/scripts/analyze_label_review.py reviews/*.json
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
KEY = ROOT / "analysis" / "results" / "label_review_key.json"
SPLITS = ROOT / "analysis" / "results" / "spatial_splits.json"
SUMMARY = ROOT / "analysis" / "results" / "spatial_splits_summary.csv"
TILES = ROOT.parent / "paper2" / "data" / "tiles_falsecolor"
OUT = ROOT / "analysis" / "results" / "label_review_result.json"
UNSURE = -1
MAX_NODATA = 0.10     # ต้องตรงกับ build_label_review.py


def nodata_fraction(item, zoom_px):
    """สัดส่วนพิกเซลดำสนิทในกรอบที่ถาม

    ชุดแรก (seed 1) สร้างก่อนที่ตัวสุ่มจะกรอง no-data จึงมีจุดที่กรอบตกบนพื้นที่
    ไม่มีข้อมูล คลาส water โดนหนักสุดเพราะน้ำเข้มกับ no-data แยกด้วยตาไม่ออกในภาพ
    ที่ยืด contrast แล้ว จุดพวกนี้วัดคุณภาพ label ไม่ได้ ต้องตัดออกก่อนคิดอัตรา
    """
    from PIL import Image
    z = zoom_px // 2
    a = np.asarray(Image.open(TILES / f"{item['tile']}.png").convert("RGB"))
    crop = a[item["y"] - z:item["y"] + z, item["x"] - z:item["x"] + z]
    return float((crop.sum(-1) < 12).mean())


def wilson(k, n, z=1.96):
    """ช่วงความเชื่อมั่นของสัดส่วน แบบ Wilson — ใช้ได้กับ n เล็กและ p ใกล้ 0/1"""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def class_pixel_share(classes):
    """สัดส่วนพิกเซลจริงของแต่ละคลาส ใช้ถ่วงกลับจากการสุ่มแบบแบ่งชั้น"""
    if not SUMMARY.exists():
        return None
    import csv
    rows = list(csv.DictReader(SUMMARY.open()))
    train = next((r for r in rows if r["split"] == "train"), None)
    if not train:
        return None
    # สรุปมีเฉพาะคลาสที่ไม่ใช่ background และหารด้วยผลรวมของกลุ่มนั้น
    shares = {c: float(train[c]) for c in classes if c in train}
    total_nonbg = sum(shares.values())
    if total_nonbg <= 0:
        return None
    return shares, total_nonbg


def cohens_kappa(a, b, k):
    n = len(a)
    if n == 0:
        return float("nan")
    obs = sum(1 for x, y in zip(a, b) if x == y) / n
    ca = np.bincount(a, minlength=k) / n
    cb = np.bincount(b, minlength=k) / n
    exp = float((ca * cb).sum())
    return (obs - exp) / (1 - exp) if exp < 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reviews", nargs="+", help="ไฟล์ JSON ที่ผู้ตรวจส่งกลับ")
    # แต่ละรอบการตรวจมี key ของตัวเอง การตรึงไว้ที่ไฟล์เดียวทำให้รอบสองถูกให้คะแนน
    # ด้วยเฉลยของรอบแรก — ด่าน set_id ด้านล่างจะจับได้ แต่จับด้วยการข้ามทุกไฟล์
    ap.add_argument("--key", default=str(KEY), help="ไฟล์เฉลยของรอบที่ตรวจ")
    ap.add_argument("--out", default=str(OUT), help="ที่เก็บผลวิเคราะห์")
    args = ap.parse_args()

    key = json.loads(Path(args.key).read_text())
    classes = key["classes"]
    truth = {it["id"]: it for it in key["items"]}

    excluded = set()
    if TILES.exists():
        for it in key["items"]:
            if nodata_fraction(it, key["zoom_px"]) > MAX_NODATA:
                excluded.add(it["id"])
        if excluded:
            print(f"ตัดออก {len(excluded)} จุดที่กรอบมี no-data เกิน "
                  f"{100 * MAX_NODATA:.0f}% (ประเมินคุณภาพ label ไม่ได้)")
    else:
        print(f"เตือน: ไม่พบ {TILES} — ข้ามการกรอง no-data")

    reviews, roles = {}, {}
    for p in args.reviews:
        d = json.loads(Path(p).read_text())
        if d.get("set_id") != key["set_id"]:
            print(f"ข้าม {p}: set_id ไม่ตรง ({d.get('set_id')} != {key['set_id']})")
            continue
        name = d.get("reviewer") or Path(p).stem
        roles[name] = d.get("reviewer_role")
        reviews[name] = {int(k): v["answer"] for k, v in d["answers"].items()
                         if int(k) not in excluded}
        print(f"อ่าน {name}: {len(reviews[name])} ข้อ")
    if not reviews:
        raise SystemExit("ไม่มีไฟล์ผลตรวจที่ใช้ได้")

    # ---------- 1. อัตราความผิดพลาดรายคลาส ----------
    # ผู้ตรวจแต่ละคนได้ตารางของตัวเอง การหยิบไฟล์แรกมาเป็น "ชุดหลัก" ทำให้ตัวเลข
    # ที่รายงานขึ้นกับลำดับ argv ซึ่งไม่ใช่คุณสมบัติของข้อมูล
    primary = list(reviews)[0]
    if len(reviews) > 1:
        print("\nอัตราความผิดพลาดรายคลาส แยกตามผู้ตรวจแต่ละคน "
              "(ไม่รวมเป็นชุดเดียว เพราะการไม่ตรงกันเองก็เป็นผล):")
        for nm in reviews:
            sub = defaultdict(lambda: {"n": 0, "agree": 0})
            for i, a in reviews[nm].items():
                if a == UNSURE:
                    continue
                st = sub[truth[i]["label"]]
                st["n"] += 1
                st["agree"] += int(a == truth[i]["label"])
            txt = "  ".join(
                f"{classes[ci][:4]} {100 * (1 - sub[ci]['agree'] / sub[ci]['n']):.0f}%"
                for ci in range(len(classes)) if sub[ci]["n"])
            print(f"  {nm:<10} {txt}")
        print(f"\n(ตารางเต็มด้านล่างใช้ '{primary}' "
              f"ส่วนความสอดคล้องคำนวณจากทุกคู่)")
    ans = reviews[primary]

    per_class = defaultdict(lambda: {"n": 0, "agree": 0, "unsure": 0})
    conf = np.zeros((len(classes), len(classes)), dtype=int)
    for i, a in ans.items():
        lab = truth[i]["label"]
        st = per_class[lab]
        if a == UNSURE:
            st["unsure"] += 1
            continue
        st["n"] += 1
        st["agree"] += int(a == lab)
        conf[lab, a] += 1

    print(f"\n{'คลาสตาม label':<14} {'ตรวจ':>5} {'ตรงกัน':>7} {'ผิด %':>7} "
          f"{'95% CI':>18} {'ไม่แน่ใจ':>9}")
    rows = []
    for ci, c in enumerate(classes):
        st = per_class[ci]
        n, ag = st["n"], st["agree"]
        err = 1 - ag / n if n else float("nan")
        lo, hi = wilson(n - ag, n)
        rows.append({"class": c, "n_judged": n, "n_agree": ag,
                     "error_rate": err, "ci95": [lo, hi], "n_unsure": st["unsure"]})
        print(f"{c:<14} {n:>5} {ag:>7} {100 * err:>6.1f}% "
              f"[{100 * lo:>5.1f},{100 * hi:>5.1f}] {st['unsure']:>9}")

    share = class_pixel_share(classes)
    weighted = None
    if share:
        shares, tot = share
        num = den = 0.0
        for r in rows:
            w = shares.get(r["class"])
            if w is None or not r["n_judged"]:
                continue
            num += w * r["error_rate"]
            den += w
        weighted = num / den if den else float("nan")
        print(f"\nอัตราความผิดพลาดรวม ถ่วงตามสัดส่วนพิกเซลจริง (ไม่รวม background): "
              f"{100 * weighted:.1f}%")
        print("  ค่านี้ต่างจากค่าเฉลี่ยตรง ๆ ของตาราง เพราะสุ่มแบบแบ่งชั้น "
              "คลาสที่มีพิกเซลน้อยจึงถูกสุ่มมาเกินสัดส่วนจริงโดยตั้งใจ")

    # ---------- 2. เมทริกซ์ทิศทางความผิดพลาด ----------
    print(f"\nlabel ปัจจุบัน (แถว) เทียบกับคำตอบของผู้ตรวจ (คอลัมน์)")
    print(" " * 14 + "".join(f"{c[:6]:>8}" for c in classes))
    for ci, c in enumerate(classes):
        print(f"{c:<14}" + "".join(f"{conf[ci, cj]:>8}" for cj in range(len(classes))))

    # ---------- 3. ความสอดคล้องระหว่างผู้ตรวจ ----------
    agreement, disputes = {}, []
    names = list(reviews)
    for x in range(len(names)):
        for y in range(x + 1, len(names)):
            a_name, b_name = names[x], names[y]
            common = sorted(set(reviews[a_name]) & set(reviews[b_name]))
            common = [i for i in common
                      if reviews[a_name][i] != UNSURE and reviews[b_name][i] != UNSURE]
            if not common:
                continue
            aa = [reviews[a_name][i] for i in common]
            bb = [reviews[b_name][i] for i in common]
            same = sum(1 for p, q in zip(aa, bb) if p == q)
            k = cohens_kappa(aa, bb, len(classes))
            agreement[f"{a_name} vs {b_name}"] = {
                "n": len(common), "percent_agreement": same / len(common),
                "cohens_kappa": k}
            print(f"\nความสอดคล้อง {a_name} vs {b_name}: "
                  f"ตรงกัน {same}/{len(common)} ({100 * same / len(common):.1f}%) "
                  f"kappa {k:.3f}")
            for i in common:
                if reviews[a_name][i] != reviews[b_name][i]:
                    disputes.append({
                        "id": i, "tile": truth[i]["tile"],
                        "current_label": classes[truth[i]["label"]],
                        a_name: classes[reviews[a_name][i]],
                        b_name: classes[reviews[b_name][i]]})
    if len(reviews) == 1:
        print("\n(มีผู้ตรวจคนเดียว จึงยังรายงานความสอดคล้องระหว่างผู้ตรวจไม่ได้ — "
              "ให้คนที่สองตรวจชุดเดียวกันแล้วรันซ้ำ)")

    Path(args.out).write_text(json.dumps({
        "set_id": key["set_id"], "classes": classes,
        "excluded_nodata_items": sorted(excluded),
        "max_nodata_fraction": MAX_NODATA,
        "reviewers": list(reviews), "primary_reviewer": primary,
        "reviewer_roles": roles,
        "per_class": rows,
        "per_class_by_reviewer": {
            nm: [{"class": classes[ci],
                  "n_judged": sum(1 for i, a in reviews[nm].items()
                                  if a != UNSURE and truth[i]["label"] == ci),
                  "n_agree": sum(1 for i, a in reviews[nm].items()
                                 if a == ci and truth[i]["label"] == ci)}
                 for ci in range(len(classes))] for nm in reviews},
        "weighted_error_rate_excl_background": weighted,
        "confusion_label_vs_reviewer": conf.tolist(),
        "inter_reviewer": agreement,
        "disputes": disputes,
    }, indent=1, ensure_ascii=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
