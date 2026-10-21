#!/usr/bin/env python3
"""ทำแผนที่เต็มฉากด้วย ensemble ที่ val เลือกไว้ แล้วเขียนเป็น GeoTIFF

ฉากลำพูนมี 1.31 Gpx ถ้าเก็บ probability ของทั้งภาพไว้ในหน่วยความจำจะกิน 36 GB
(7 คลาส x float32) จึงประมวลผลเป็นแถบแนวนอน เฉลี่ย softmax ของสมาชิกภายในแถบ
แล้วเขียน argmax ออกทันที แถบซ้อนกันเท่าความกว้าง overlap เพื่อไม่ให้มีรอยต่อ

สมาชิกอ่านจาก ensemble_selection.json — สคริปต์นี้ไม่เลือกอะไรเอง

ใช้งาน
------
  python paper1/scripts/predict_ensemble_scene.py --out map_ensemble.tif
  python paper1/scripts/predict_ensemble_scene.py --bounds 494000 2049000 500000 2055000
"""
import argparse, json, os, sys
from collections import Counter
from pathlib import Path

os.environ.pop("PROJ_DATA", None); os.environ.pop("PROJ_LIB", None)
import numpy as np
import rasterio
from rasterio.windows import Window
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval_rerun import build_model                       # noqa: E402

TILE, STRIDE = 512, 448
MEAN = np.array([0.485, 0.456, 0.406, 0.5], np.float32)
STD = np.array([0.229, 0.224, 0.225, 0.25], np.float32)


def load_members(results, selection, device, runs=None):
    """โหลดสมาชิก — จาก ensemble_selection.json หรือจากรายชื่อ run ที่ระบุมา

    รับ runs เพื่อให้ทำแผนที่ของโมเดลเดี่ยวได้ด้วยโค้ดเส้นทางเดียวกับ ensemble
    จะได้ไม่มีความต่างของกระบวนการมาปนกับความต่างของโมเดล
    """
    if runs:
        sel = {"selected": list(runs), "selection_split": "n/a (single run)"}
    else:
        sel = json.loads(Path(selection).read_text())
    out = []
    for name, times in Counter(sel["selected"]).items():
        rd = Path(results) / name
        meta = json.loads(next(rd.glob("logs/*_best_metrics.json")).read_text())["run"]
        m = build_model(meta["model"], meta["encoder"],
                        4 if meta.get("use_nir") else 3,
                        meta.get("num_classes", 7)).to(device).eval()
        st = torch.load(rd / "models" / f"{meta['model']}_best.pth",
                        map_location=device)["model_state_dict"]
        if next(iter(st)).startswith("module."):
            st = {k[7:]: v for k, v in st.items()}
        m.load_state_dict(st)
        out.append((m, bool(meta.get("use_nir")), times))
        print(f"  โหลด {name} (x{times}, {'4' if meta.get('use_nir') else '3'} ช่อง)")
    return out, sel


def stretch(a):
    """ยืด contrast แบบเดียวกับที่ใช้ตัด tile ตอนเทรน (percentile 2-98 ต่อแบนด์)"""
    out = np.zeros(a.shape, np.float32)
    for i in range(a.shape[0]):
        b = a[i].astype(np.float32)
        lo, hi = np.percentile(b[b > 0], (2, 98)) if (b > 0).any() else (0, 1)
        out[i] = np.clip((b - lo) / max(hi - lo, 1e-6), 0, 1)
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="theos/lamphoon/lamphoon_merged.tif")
    ap.add_argument("--results", default="/project/gisnorth/paper1/rerun_output")
    ap.add_argument("--selection",
                    default="/project/gisnorth/paper1/analysis/results/ensemble_selection.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", nargs="+", default=None,
                    help="ชื่อ run ที่จะใช้ แทนการอ่านจาก ensemble_selection.json")
    ap.add_argument("--strip-rows", type=int, default=4480, help="ความสูงของแถบ (พิกเซล)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--bounds", nargs=4, type=float, default=None,
                    help="จำกัดพื้นที่: minx miny maxx maxy (CRS ของฉาก)")
    ap.add_argument("--suppress-background", action="store_true",
                    help="argmax เฉพาะหกคลาสจริง (ตัดความน่าจะเป็นของ background ออก) "
                         "แบบเดียวกับโหมด sup ของ addendum D/J; ไม่ต้องฝึกใหม่")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    members, sel = load_members(args.results, args.selection, device, args.runs)
    print(f"ensemble {len(sel['selected'])} สมาชิก | device {device}")

    with rasterio.open(args.scene) as src:
        if args.bounds:
            win = rasterio.windows.from_bounds(*args.bounds, transform=src.transform)
            col0, row0 = int(win.col_off), int(win.row_off)
            W, H = int(win.width), int(win.height)
        else:
            col0 = row0 = 0; W, H = src.width, src.height
        prof = src.profile.copy()
        prof.update(count=1, dtype="uint8", compress="lzw", nodata=255,
                    transform=rasterio.windows.transform(
                        Window(col0, row0, W, H), src.transform),
                    width=W, height=H)
        print(f"พื้นที่ {W} x {H} px ({W*H/1e6:.0f} Mpx)")

        with rasterio.open(args.out, "w", **prof) as dst:
            ov = TILE - STRIDE
            y = 0
            while y < H:
                h = min(args.strip_rows, H - y)
                pad = ov if y + h < H else 0
                block = src.read(window=Window(col0, row0 + y, W, min(h + pad, H - y)))
                bh = block.shape[1]
                if block[:3].max() == 0:
                    dst.write(np.full((bh - pad, W), 255, np.uint8), 1,
                              window=Window(0, y, W, bh - pad)); y += h; continue
                img = stretch(block)
                ncls = 7
                acc = np.zeros((ncls, bh, W), np.float32)
                cnt = np.zeros((bh, W), np.float32)
                for m, use_nir, times in members:
                    ch = 4 if use_nir else 3
                    mean, std = MEAN[:ch], STD[:ch]
                    batch, pos = [], []
                    for ty in range(0, max(bh - TILE, 0) + 1, STRIDE):
                        for tx in range(0, max(W - TILE, 0) + 1, STRIDE):
                            t = img[:ch, ty:ty+TILE, tx:tx+TILE]
                            if t.shape[1] != TILE or t.shape[2] != TILE:
                                continue
                            batch.append((t - mean[:, None, None]) / std[:, None, None])
                            pos.append((ty, tx))
                            if len(batch) == args.batch:
                                pr = torch.softmax(m(torch.from_numpy(
                                    np.stack(batch)).to(device)), 1).cpu().numpy()
                                for k, (py, px) in enumerate(pos):
                                    acc[:, py:py+TILE, px:px+TILE] += pr[k] * times
                                    cnt[py:py+TILE, px:px+TILE] += times
                                batch, pos = [], []
                    if batch:
                        pr = torch.softmax(m(torch.from_numpy(
                            np.stack(batch)).to(device)), 1).cpu().numpy()
                        for k, (py, px) in enumerate(pos):
                            acc[:, py:py+TILE, px:px+TILE] += pr[k] * times
                            cnt[py:py+TILE, px:px+TILE] += times
                if args.suppress_background:
                    acc[0] = -1.0                              # ความน่าจะเป็นจริงไม่ติดลบ
                lab = np.where(cnt > 0, acc.argmax(0), 255).astype(np.uint8)
                lab[block[:3].max(0) == 0] = 255              # no-data
                dst.write(lab[:bh - pad], 1, window=Window(0, y, W, bh - pad))
                print(f"  แถบ {y}-{y+h-pad} เขียนแล้ว", flush=True)
                y += h
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
