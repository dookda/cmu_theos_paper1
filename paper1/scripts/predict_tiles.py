#!/usr/bin/env python3
"""ผลทำนายของหลายวิธีบนภาพย่อยชุดเล็ก สำหรับรูปเปรียบเทียบเชิงภาพ (ไม่ใช่การประเมิน)

ภาพย่อยที่ใช้อยู่ในชุดที่กันไว้ซึ่งอ่านคะแนนไปแล้ว สคริปต์นี้ไม่คิดคะแนนใด ๆ
บันทึกเฉพาะ argmax ต่อพิกเซล ทั้งแบบเจ็ดคลาส (std) และไม่ให้ทำนายพื้นหลัง (sup)

เตรียมข้อมูลด้วย SegmentationDataset + get_val_transform ชุดเดียวกับการประเมิน
ensemble เฉลี่ย softmax ของสมาชิกตาม ensemble_selection.json (นับซ้ำตามที่เลือก)

ใช้งาน (บน Erawan)
------
  python3 scripts/predict_tiles.py --tiles tile_01826 tile_01925 \
      --runs rerun_output/unet_effb3__b1_rgb_aug__s0 ... --ensemble \
      --out analysis/results/method_tiles.npz
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_rerun import build_model                                  # noqa: E402
from utils.dataset import SegmentationDataset, get_val_transform    # noqa: E402

RGB_ROOT = "/project/gisnorth/cmu_theos_ai_data_fixed"
NIR_ROOT = "/project/gisnorth/cmu_theos_ai_data_nir_only"


def load(run_dir, device):
    meta = json.loads(next(run_dir.glob("logs/*_best_metrics.json")).read_text())["run"]
    nir = bool(meta.get("use_nir"))
    m = build_model(meta["model"], meta["encoder"], 4 if nir else 3,
                    meta.get("num_classes", 7)).to(device).eval()
    st = torch.load(run_dir / "models" / f"{meta['model']}_best.pth",
                    map_location=device)["model_state_dict"]
    if next(iter(st)).startswith("module."):
        st = {k[7:]: v for k, v in st.items()}
    m.load_state_dict(st)
    return m, nir


@torch.no_grad()
def probs(model, nir, tiles, device):
    root = NIR_ROOT if nir else RGB_ROOT
    ds = SegmentationDataset(f"{root}/tiles", f"{root}/semantic_labels",
                             file_list=[t + ".png" for t in tiles], num_classes=7,
                             transform=get_val_transform(512, use_nir=nir), use_nir=nir,
                             class_remap={7: 0})
    out, labels = [], []
    for i in range(len(ds)):
        x, y = ds[i]
        out.append(torch.softmax(model(x[None].to(device)), 1)[0].cpu().numpy())
        labels.append(y.numpy().astype(np.uint8))
    return np.stack(out), np.stack(labels)


def argmaxes(p):
    sup = p.copy(); sup[:, 0] = -1.0
    return p.argmax(1).astype(np.uint8), sup.argmax(1).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", nargs="+", required=True)
    ap.add_argument("--runs", nargs="+", default=[])
    ap.add_argument("--ensemble", action="store_true")
    ap.add_argument("--selection", default="analysis/results/ensemble_selection.json")
    ap.add_argument("--ensemble-root", default="rerun_output")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res, labels = {}, None
    for r in a.runs:
        m, nir = load(Path(r), device)
        p, labels = probs(m, nir, a.tiles, device)
        res[f"{Path(r).name}__std"], res[f"{Path(r).name}__sup"] = argmaxes(p)
        print(f"  {Path(r).name}")
        del m; torch.cuda.empty_cache()
    if a.ensemble:
        sel = json.loads(Path(a.selection).read_text())["selected"]
        tot = None
        for name, times in Counter(sel).items():
            m, nir = load(Path(a.ensemble_root) / name, device)
            p, labels = probs(m, nir, a.tiles, device)
            tot = p * times if tot is None else tot + p * times
            del m; torch.cuda.empty_cache()
        res["ensemble__std"], res["ensemble__sup"] = argmaxes(tot / len(sel))
        print(f"  ensemble ({len(sel)} members)")
    np.savez_compressed(a.out, tiles=np.array(a.tiles), labels=labels, **res)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
