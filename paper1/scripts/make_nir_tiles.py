"""
===========================================================
สคริปต์ตัด NIR band จาก scene TIF (4-band) ให้ตรง grid ของ
tiles ที่ render ไว้แล้ว (อ้างอิง bounds จาก metadata.json)
===========================================================
ผลลัพธ์: tile_XXXXX_nir.png (uint8, stretch แบบ global percentile)
ใน --output-dir แยกจาก tiles เดิม ไม่เขียนทับอะไรทั้งสิ้น

การใช้งาน (บน Erawan):
    python scripts/make_nir_tiles.py \
        --tiles-dir /project/gisnorth/cmu_theos_ai_data/tiles \
        --metadata /project/gisnorth/cmu_theos_ai_data/tiles_metadata.json \
        --scene /project/gisnorth/cmu_theos_ai_data/th2_20260108_034820_psp_gcs_ort_47n_001.tif \
        --scene /project/gisnorth/cmu_theos_ai_data/th2_20260108_034822_psp_gcs_ort_47n_002.tif \
        --output-dir /project/gisnorth/cmu_theos_ai_data/nir_tiles
"""

import argparse
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.windows import from_bounds

NIR_BAND = 4  # THEOS band order: R,G,B,NIR
TILE_PX = 512


def read_nir_window(src, bounds):
    """อ่าน NIR band ตาม bounds (boundless, นอกภาพ = 0)"""
    win = from_bounds(bounds["left"], bounds["bottom"], bounds["right"], bounds["top"],
                      transform=src.transform)
    return src.read(NIR_BAND, window=win, boundless=True, fill_value=0,
                    out_shape=(TILE_PX, TILE_PX))


def compute_stretch(scenes, meta_list, sample_n, seed=42):
    """หา percentile 2/98 ของ NIR จาก tiles ตัวอย่างทั่วทั้งสอง scene"""
    random.seed(seed)
    sample = random.sample(meta_list, min(sample_n, len(meta_list)))
    vals = []
    for m in sample:
        for src in scenes:
            arr = read_nir_window(src, m["bounds"])
            valid = arr[arr > 0]
            if valid.size > 1000:
                vals.append(valid)
                break
    allv = np.concatenate(vals)
    lo, hi = np.percentile(allv, [2, 98])
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser(description="ตัด NIR tiles ตาม grid เดิม")
    ap.add_argument("--tiles-dir", type=Path, required=True,
                    help="โฟลเดอร์ RGB tiles (ใช้กำหนดว่า tile ไหนต้องสร้าง NIR)")
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--scene", action="append", required=True,
                    help="scene TIF 4-band (ระบุซ้ำได้ เรียงตาม priority)")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--stretch-sample", type=int, default=300)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    with open(args.metadata, "r", encoding="utf-8") as f:
        meta = {m["filename"]: m for m in json.load(f)}

    rgb_files = sorted(f for f in os.listdir(args.tiles_dir)
                       if f.endswith(".png") and "_nir" not in f)
    if args.limit:
        rgb_files = rgb_files[: args.limit]

    missing_meta = [f for f in rgb_files if f not in meta]
    if missing_meta:
        raise SystemExit(f"ไม่พบ metadata ของ {len(missing_meta)} tiles เช่น {missing_meta[:5]}")

    scenes = [rasterio.open(p) for p in args.scene]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    lo, hi = compute_stretch(scenes, [meta[f] for f in rgb_files], args.stretch_sample)
    print(f"Global NIR stretch (p2/p98): {lo:.0f} - {hi:.0f}")

    n_done, per_scene = 0, [0] * len(scenes)
    for i, fname in enumerate(rgb_files):
        bounds = meta[fname]["bounds"]
        # เลือก scene ที่ครอบ tile ได้ดีสุด (ลำดับ priority) แล้วเติมช่องว่างจาก scene ถัดไป
        best = None
        for si, src in enumerate(scenes):
            arr = read_nir_window(src, bounds)
            cov = (arr > 0).mean()
            if best is None or cov > best[1]:
                best = (arr, cov, si)
            if cov > 0.999:
                break
        arr, cov, si = best
        if cov < 0.999:
            for sj, src in enumerate(scenes):
                if sj == si:
                    continue
                fill = read_nir_window(src, bounds)
                arr = np.where(arr > 0, arr, fill)
        per_scene[si] += 1

        out = np.clip((arr.astype(np.float32) - lo) / (hi - lo + 1e-8), 0, 1)
        out = (out * 255).astype(np.uint8)
        out_name = fname.replace(".png", "_nir.png")
        cv2.imwrite(str(args.output_dir / out_name), out)
        n_done += 1
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(rgb_files)}...", flush=True)

    for src in scenes:
        src.close()

    print(f"\nเสร็จ: {n_done} NIR tiles -> {args.output_dir}")
    print(f"แหล่งข้อมูลหลักต่อ scene: {per_scene}")


if __name__ == "__main__":
    main()
