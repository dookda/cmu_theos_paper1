#!/usr/bin/env python3
"""ประเมิน ensemble ที่เลือกจาก val แล้ว บน split ที่กันไว้

`eval_rerun.py` ประเมินโมเดลทีละตัว จึงไม่เคยให้คะแนนของ ensemble เลย สคริปต์นี้
อ่านรายชื่อสมาชิกจาก `ensemble_selection.json` (ซึ่งเลือกด้วย val เท่านั้น) แล้ว
เฉลี่ย softmax ของสมาชิกก่อน argmax — เป็นการอ่านค่าเดียว ไม่มีการเลือกอะไรอีก

สมาชิกที่ถูกเลือกซ้ำจะถูกนับซ้ำตามนั้น (greedy อนุญาตให้เลือกซ้ำเพื่อถ่วงน้ำหนัก)

ใช้งาน
------
  python paper1/scripts/eval_ensemble.py --results rerun_output --split test
  python paper1/scripts/eval_ensemble.py --results rerun_output --split external \
      --unlock "<เหตุผล>"
"""
import argparse, json, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.dataset import create_spatial_dataloaders          # noqa: E402
from utils.metrics import SegmentationMetrics                 # noqa: E402
from eval_rerun import build_model                            # noqa: E402

RGB_ROOT = "/project/gisnorth/cmu_theos_ai_data_fixed"
NIR_ROOT = "/project/gisnorth/cmu_theos_ai_data_nir_only"


@torch.no_grad()
def member_probs(run_dir, split, splits_file, device, batch_size):
    """softmax ของสมาชิกหนึ่งตัวบน split พร้อมชื่อ tile และ label"""
    meta = json.loads(next(run_dir.glob("logs/*_best_metrics.json")).read_text())["run"]
    ckpt = run_dir / "models" / f"{meta['model']}_best.pth"
    use_nir = bool(meta.get("use_nir"))
    root = NIR_ROOT if use_nir else RGB_ROOT
    config = {"data": {"tiles_dir": f"{root}/tiles", "labels_dir": f"{root}/semantic_labels",
                       "tile_size": 512, "use_nir": use_nir},
              "segmentation": {"batch_size": batch_size},
              "num_classes": meta.get("num_classes", 7)}
    loaders, ncls = create_spatial_dataloaders(config, splits_file, batch_size=batch_size,
                                               augment=False, return_name=True)
    model = build_model(meta["model"], meta["encoder"],
                        4 if use_nir else 3, ncls).to(device).eval()
    state = torch.load(ckpt, map_location=device)["model_state_dict"]
    if next(iter(state)).startswith("module."):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state)

    probs, names, labels = [], [], []
    for images, masks, nm in loaders[split]:
        p = torch.softmax(model(images.to(device)), 1).cpu().numpy().astype(np.float16)
        probs.append(p); names.extend(nm); labels.append(masks.numpy().astype(np.uint8))
    del model; torch.cuda.empty_cache()
    return np.concatenate(probs), names, np.concatenate(labels), ncls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test", "external"])
    ap.add_argument("--splits-file",
                    default="/project/gisnorth/paper1/analysis/results/spatial_splits.json")
    ap.add_argument("--selection",
                    default="/project/gisnorth/paper1/analysis/results/ensemble_selection.json")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--unlock", default=None)
    ap.add_argument("--data-root", default=None,
                    help="one tree with RGB tiles, _nir tiles and labels (Roi Et probe)")
    ap.add_argument("--out", default=None,
                    help="output path; supplementary evaluations must set it")
    ap.add_argument("--background-modes", action="store_true",
                    help="addendum J: write std and sup (background excluded from the argmax) "
                         "confusion matrices, pooled and per block; requires --out")
    args = ap.parse_args()
    if args.data_root:
        global RGB_ROOT, NIR_ROOT
        RGB_ROOT = NIR_ROOT = args.data_root

    if args.split == "external" and not args.unlock:
        raise SystemExit("external ล็อกอยู่ — ต้องระบุ --unlock \"<เหตุผล>\"")

    sel = json.loads(Path(args.selection).read_text())
    members = sel["selected"]
    print(f"ensemble {len(members)} สมาชิก (เลือกจาก {sel['selection_split']}) "
          f"| split={args.split}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = json.loads(Path(args.splits_file).read_text())
    blk_src = "blocks_external" if args.split == "external" else "blocks"
    tile_block = {t: b for b, i in payload[blk_src].items() for t in i["tiles"]}
    classes = payload["classes"]

    total, names, labels, ncls = None, None, None, None
    for name, times in Counter(members).items():
        p, nm, lb, ncls = member_probs(Path(args.results) / name, args.split,
                                       args.splits_file, device, args.batch_size)
        total = p.astype(np.float32) * times if total is None else total + p.astype(np.float32) * times
        names, labels = nm, lb
        print(f"  + {name} (x{times})")

    if args.background_modes:
        if not args.out:
            raise SystemExit("--background-modes writes a supplementary file: pass --out")
        n = total.shape[1]
        sup = total.copy(); sup[:, 0] = -np.inf
        preds = {"std": total.argmax(1), "sup": sup.argmax(1)}
        lab = labels.reshape(labels.shape[0], -1)
        cm = {m: np.zeros((n, n), np.int64) for m in preds}
        blocks = {}
        for i, fn in enumerate(names):
            t = lab[i].astype(np.int64)
            ok = t < n
            b = tile_block.get(Path(fn).stem)
            for m, pr in preds.items():
                c = np.bincount(n * t[ok] + pr[i].reshape(-1)[ok], minlength=n * n).reshape(n, n)
                cm[m] += c
                if b is not None:
                    blocks.setdefault(b, {k: np.zeros((n, n), np.int64) for k in preds})[m] += c
        Path(args.out).write_text(json.dumps({
            "split": args.split, "prereg": "paper1/configs/addendum_prereg_2026-10.md#J",
            "n_members": len(members), "members": members,
            "evaluated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "unlock_reason": args.unlock, "class_names": list(classes),
            "confusion_matrix": {m: v.tolist() for m, v in cm.items()},
            "per_block": {b: {m: v.tolist() for m, v in d.items()}
                          for b, d in sorted(blocks.items())}}, indent=1))
        print(f"wrote {args.out} (std and sup confusion matrices, {len(blocks)} blocks)")
        return

    pred = torch.from_numpy(total.argmax(1))
    gt = torch.from_numpy(labels)
    pooled = SegmentationMetrics(ncls, classes); pooled.update(pred, gt)
    per_block, counts = {}, {}
    for i, fn in enumerate(names):
        b = tile_block.get(Path(fn).stem)
        if b is None:
            continue
        per_block.setdefault(b, SegmentationMetrics(ncls, classes)).update(
            pred[i:i+1], gt[i:i+1])
        counts[b] = counts.get(b, 0) + 1

    s = pooled.get_summary()
    print(f"\nensemble บน {args.split}: mIoU {s['mean_iou']:.4f}  OA {s['overall_accuracy']:.4f}"
          f"  kappa {s['kappa']:.4f}  ({len(per_block)} บล็อก)")
    for c, v in s["per_class_iou"].items():
        print(f"    {c:<12} IoU {v:.4f}")

    # เมทริกซ์ความสับสนมีอยู่ใน SegmentationMetrics อยู่แล้ว แต่เดิมไม่ได้บันทึก
    # บทความจึงรายงานได้แค่ IoU รายคลาส ไม่เห็นว่าความผิดพลาดวิ่งไปทางไหน
    out = {"split": args.split, "n_members": len(members), "members": members,
           "selection_split": sel["selection_split"],
           "evaluated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "unlock_reason": args.unlock,
           "class_names": list(classes),
           "confusion_matrix": pooled.confusion_matrix.tolist(), **s,
           "per_block": {b: {**m.get_summary(), "n_tiles": counts[b]}
                         for b, m in sorted(per_block.items())}}
    dest = (Path(args.out) if args.out else
            Path(args.results).parent / "analysis" / "results" / f"ensemble_eval_{args.split}.json")
    dest.write_text(json.dumps(out, indent=1, default=float))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
