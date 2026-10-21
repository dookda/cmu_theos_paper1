#!/usr/bin/env python3
"""ความเห็นพ้องระหว่างแผนที่ของทุกโมเดลในตาราง fig_scene_grid

ตัวเลขบอกว่าโมเดลไหนดีกว่า แต่ไม่บอกว่าแผนที่ที่ได้ต่างกันแค่ไหน สคริปต์นี้วัด
ข้อหลัง: โมเดลที่ mIoU ห่างกันหลายจุด ให้คลาสเดียวกันกี่เปอร์เซ็นต์ของฉาก

วัดบน thumbnail ที่ scene_thumbnail.py ย่อไว้ ไม่ใช่ที่ 0.5 ม. เพราะคำถามคือ
"แผนที่ที่ผู้อ่านเห็นในรูปต่างกันแค่ไหน" ความเห็นพ้องที่ความละเอียดเต็มจะต่ำกว่านี้
thumbnail ย่อด้วย Resampling.mode แล้ว map ผ่าน LUT จึงมีเฉพาะสีคลาสตรง ๆ
ไม่มีสีที่เกิดจากการ interpolate — แปลงกลับเป็นดัชนีคลาสได้ตรงตัว

เดิมไฟล์ผลนี้ถูกสร้างแบบ ad-hoc ไม่มีสคริปต์ ตัวเลขในเปเปอร์จึง reproduce ไม่ได้
สคริปต์นี้แทนที่ขั้นตอนนั้น และคำนวณ **ทุกคู่** แทนการสุ่ม 200 คู่ ผลจึงไม่ขึ้นกับ seed

ใช้งาน
------
  python paper1/scripts/scene_grid_agreement.py
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palette import CLASS                                     # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
THUMBS = ROOT / "analysis" / "scene_grid"
RES = ROOT / "analysis" / "results"
OUT = RES / "scene_grid_agreement.json"
CLASSES = ["background", "vegetation", "water", "urban", "agriculture",
           "bare_soil", "road"]
NODATA = 255            # LUT เติมขาวไว้สำหรับค่าที่ไม่ใช่คลาส


def rgb_to_index(img):
    """แปลงภาพสีคลาสกลับเป็นดัชนี พิกเซลที่ไม่ตรงสีใดเลยถือเป็น nodata"""
    a = np.asarray(img.convert("RGB"))
    idx = np.full(a.shape[:2], NODATA, dtype=np.uint8)
    for i, c in enumerate(CLASSES):
        h = CLASS[c]
        rgb = np.array([int(h[k:k + 2], 16) for k in (1, 3, 5)], dtype=np.uint8)
        idx[(a == rgb).all(axis=2)] = i
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thumbs", default=str(THUMBS))
    ap.add_argument("--seed", type=int, default=0,
                    help="seed ของ run ที่นำมาเทียบ (ตารางใช้ s0)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    pngs = {p.stem: p for p in sorted(Path(a.thumbs).glob(f"*__s{a.seed}.png"))}
    if not pngs:
        sys.exit(f"ไม่พบ thumbnail ที่ seed {a.seed} ใน {a.thumbs}")

    miou = {}
    ev = RES / "rerun_eval_test.json"
    if ev.exists():
        miou = {r["run"]: r["mean_iou"] for r in json.loads(ev.read_text())["results"]}
    scored = {k: miou[k] for k in pngs if k in miou}
    if not scored:
        sys.exit("ไม่มี run ใดใน rerun_eval_test.json ตรงกับ thumbnail ที่มี")

    maps, shape = {}, None
    for name, p in pngs.items():
        m = rgb_to_index(Image.open(p))
        if shape is None:
            shape = m.shape
        elif m.shape != shape:
            sys.exit(f"ขนาด thumbnail ไม่ตรงกัน: {name} {m.shape} เทียบ {shape}")
        maps[name] = m

    # เทียบเฉพาะพิกเซลที่ทุกแผนที่มีข้อมูล เพื่อไม่ให้ขอบ nodata ดันความเห็นพ้องขึ้น
    valid = np.ones(shape, dtype=bool)
    for m in maps.values():
        valid &= m != NODATA

    def agree(x, y):
        return float((maps[x][valid] == maps[y][valid]).mean())

    best = max(scored, key=scored.get)
    worst = min(scored, key=scored.get)
    names = sorted(maps)
    pairs = [agree(x, y) for x, y in itertools.combinations(names, 2)]
    vals = sorted(100 * v for v in scored.values())

    out = {
        "note": "วัดบนพิกเซลที่ทุกแผนที่มีข้อมูลร่วมกัน บน thumbnail ที่ย่อจาก 0.5 ม. "
                "มิใช่ที่ความละเอียดเต็ม ความเห็นพ้องที่ 0.5 ม. จะต่ำกว่านี้ "
                "ตัวเลขนี้ตอบคำถามว่า แผนที่ที่ผู้อ่านเห็นในรูปต่างกันแค่ไหน "
                "คำนวณครบทุกคู่ ไม่ได้สุ่ม จึงไม่ขึ้นกับ seed ของการสุ่ม",
        "n_models": len(maps),
        "n_models_with_miou": len(scored),
        "seed": a.seed,
        "thumbnail_px": [int(shape[1]), int(shape[0])],
        "valid_pixel_fraction": float(valid.mean()),
        "miou_test_pct": {
            "min": vals[0], "max": vals[-1], "range": vals[-1] - vals[0],
            "median": float(np.median(vals)),
        },
        "best_run": best,
        "worst_run": worst,
        "best_vs_worst_pixel_agreement": agree(best, worst),
        "pairwise_agreement": {
            "n_pairs": len(pairs), "sampling": "all pairs",
            "median": float(np.median(pairs)),
            "min": float(min(pairs)), "max": float(max(pairs)),
        },
    }
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"wrote {a.out}")
    print(f"  {out['n_models']} แผนที่, {out['pairwise_agreement']['n_pairs']} คู่, "
          f"median agreement {100 * out['pairwise_agreement']['median']:.1f}%")
    print(f"  best {best} ({100 * scored[best]:.2f}%) vs worst {worst} "
          f"({100 * scored[worst]:.2f}%) เห็นพ้อง "
          f"{100 * out['best_vs_worst_pixel_agreement']:.1f}%")


if __name__ == "__main__":
    main()
