#!/usr/bin/env python3
"""Evaluate trained paper1 re-run checkpoints on a held-out split.

Kept separate from training on purpose. In the first round the test set was
consulted while comparing architectures, choosing among five ensembles and
checking label refinement, so it was no longer a holdout. Here training only ever
sees train and val; this script is the one place a held-out split is read, and
`external` (Scene 2) stays locked until every selection has been made on val.

The lock is enforced, not just documented: evaluating `--split external` requires
--unlock with the reason recorded into the output, so an accidental peek leaves a
trace.

Usage
-----
  python paper1/scripts/eval_rerun.py --results rerun_output --split test
  python paper1/scripts/eval_rerun.py --results rerun_output --split external \
      --unlock "final numbers, selection frozen 2026-09-01"
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.dataset import create_spatial_dataloaders  # noqa: E402
from utils.metrics import SegmentationMetrics          # noqa: E402


def build_model(model_name, encoder, in_channels, num_classes):
    if model_name == "unet":
        from models.unet import create_unet
        return create_unet(encoder, None, in_channels, num_classes)
    if model_name == "unetpp":
        from models.unet import create_unetpp
        return create_unetpp(encoder, None, in_channels, num_classes)
    if model_name == "deeplabv3":
        from models.deeplabv3 import create_deeplabv3
        return create_deeplabv3(encoder, None, in_channels, num_classes)
    if model_name == "hrnet":
        from models.hrnet import create_hrnet
        return create_hrnet(encoder, None, in_channels, num_classes)
    if model_name == "segformer":
        from models.transformer_seg import create_segformer
        return create_segformer(encoder, None, in_channels, num_classes)
    if model_name == "upernet":
        from models.transformer_seg import create_upernet
        return create_upernet(encoder, None, in_channels, num_classes)
    if model_name == "satlas":
        # the checkpoint holds every weight, so the pretrained file is not needed here
        from models.satlas_seg import create_satlas
        return create_satlas(num_classes, in_channels, load_weights=False, encoder=encoder)
    if model_name == "dinov3":
        from models.dinov3_seg import create_dinov3
        return create_dinov3(num_classes, in_channels, load_weights=False)
    raise ValueError(f"unknown model {model_name}")


@torch.no_grad()
def evaluate(model, loader, device, num_classes, class_names, tile_block=None):
    """คืน (pooled metrics, {block_id: metrics}, {block_id: จำนวน tile})

    per-block เป็นหน่วยสถิติของงานนี้: tile ที่อยู่บล็อกเดียวกันสัมพันธ์กันเชิงพื้นที่
    การ bootstrap ที่ระดับ tile จะให้ CI แคบเกินจริง
    """
    pooled = SegmentationMetrics(num_classes, class_names)
    per_block, counts = {}, {}
    model.eval()
    for batch in loader:
        if len(batch) == 3:
            images, masks, names = batch
        else:
            images, masks = batch
            names = None
        images = images.to(device, non_blocking=True)
        # SegmentationMetrics.update รับ torch tensor (เรียก .cpu()/.dim() ข้างใน)
        pred = model(images).argmax(1).cpu()
        pooled.update(pred, masks)
        if names is not None and tile_block is not None:
            for i, fn in enumerate(names):
                blk = tile_block.get(Path(fn).stem)
                if blk is None:
                    continue
                m = per_block.setdefault(
                    blk, SegmentationMetrics(num_classes, class_names))
                m.update(pred[i:i + 1], masks[i:i + 1])
                counts[blk] = counts.get(blk, 0) + 1
    return pooled, per_block, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test", "external"])
    ap.add_argument("--splits-file",
                    default="/project/gisnorth/paper1/analysis/results/spatial_splits.json")
    ap.add_argument("--rgb-root", default="/project/gisnorth/cmu_theos_ai_data_fixed")
    ap.add_argument("--nir-root", default="/project/gisnorth/cmu_theos_ai_data_nir_only")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--labels-root", default=None,
                    help="read semantic_labels from this root for every run (default: the run's own "
                         "data root). Used 2026-10-21 to re-score the NIR arm against the RGB reference, "
                         "after the NIR data root was found to hold the pre-fix masks")
    ap.add_argument("--only", default=None,
                    help="evaluate only runs whose name contains this substring, e.g. _rgbnir_")
    ap.add_argument("--nir-only", action="store_true",
                    help="evaluate only runs that read the NIR data root (use_nir or nir_mode)")
    ap.add_argument("--unlock", default=None,
                    help="เหตุผลที่เปิด external — บังคับใส่เมื่อ --split external")
    ap.add_argument("--out", default=None,
                    help="ไฟล์ผลลัพธ์ (ค่าเริ่มต้น analysis/results/rerun_eval_<split>.json) "
                         "การทดลองเสริมต้องระบุ เพื่อไม่ทับผลหลัก")
    args = ap.parse_args()

    if args.split == "external" and not args.unlock:
        raise SystemExit(
            "external ถูกล็อกไว้: ต้องเลือกสถาปัตยกรรมและ ensemble จาก val ให้เสร็จก่อน\n"
            "ถ้าตัดสินใจครบแล้วจริง ให้ระบุ --unlock \"<เหตุผล>\"")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path(args.results)

    # tile -> spatial block, หน่วยสถิติสำหรับ block bootstrap
    # external (Scene 2) อยู่นอก lattice ของ pool A จึงมีบล็อกชุดของตัวเอง
    splits_payload = json.loads(Path(args.splits_file).read_text())
    block_src = ("blocks_external" if args.split == "external" else "blocks")
    # random_splits.json ไม่มี "blocks" ของ pool A เพราะไม่ได้แบ่งเป็นบล็อก
    tile_block = {t: blk for blk, info in splits_payload.get(block_src, {}).items()
                  for t in info["tiles"]}

    runs = sorted(p.parent.parent for p in results_dir.glob("*/logs/*_best_metrics.json"))
    if args.only:
        runs = [r for r in runs if args.only in r.name]
    if not runs:
        raise SystemExit(f"ไม่พบ checkpoint ใน {results_dir}")
    print(f"พบ {len(runs)} runs | split={args.split} | device={device}")

    out = []
    for run_dir in runs:
        meta_path = next(run_dir.glob("logs/*_best_metrics.json"))
        meta = json.loads(meta_path.read_text()).get("run", {})
        model_name = meta.get("model")
        ckpt = run_dir / "models" / f"{model_name}_best.pth"
        if not model_name or not ckpt.exists():
            print(f"  ข้าม {run_dir.name}: ไม่มี checkpoint หรือ provenance")
            continue

        use_nir = bool(meta.get("use_nir"))
        nir_mode = meta.get("nir_mode")
        if args.nir_only and not (use_nir or nir_mode):
            continue
        data_root = args.nir_root if (use_nir or nir_mode) else args.rgb_root
        config = {
            "data": {"tiles_dir": f"{data_root}/tiles",
                     "labels_dir": f"{args.labels_root or data_root}/semantic_labels",
                     "tile_size": 512, "use_nir": use_nir, "nir_mode": nir_mode},
            "segmentation": {"batch_size": args.batch_size},
            "num_classes": meta.get("num_classes", 7),
        }
        loaders, num_classes = create_spatial_dataloaders(
            config, args.splits_file, batch_size=args.batch_size, augment=False,
            return_name=True)
        class_names = json.loads(Path(args.splits_file).read_text())["classes"]

        model = build_model(model_name, meta.get("encoder"),
                            4 if (use_nir or nir_mode == "lowpass") else 3, num_classes).to(device)
        state = torch.load(ckpt, map_location=device)["model_state_dict"]
        if next(iter(state)).startswith("module."):
            state = {k[7:]: v for k, v in state.items()}
        model.load_state_dict(state)

        pooled, per_block, counts = evaluate(model, loaders[args.split], device,
                                            num_classes, class_names, tile_block)
        summary = pooled.get_summary()
        row = {"run": run_dir.name, "split": args.split, **meta, **summary,
               "per_block": {b: {**m.get_summary(), "n_tiles": counts.get(b, 0)}
                             for b, m in sorted(per_block.items())}}
        out.append(row)
        print(f"  {run_dir.name:<45} mIoU={summary['mean_iou']:.4f} "
              f"({len(per_block)} blocks)")

    payload = {
        "split": args.split,
        "evaluated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "unlock_reason": args.unlock,
        "labels_root": args.labels_root,
        "splits_file": args.splits_file,
        "results": out,
    }
    dest = (Path(args.out) if args.out else
            results_dir.parent / "analysis" / "results" / f"rerun_eval_{args.split}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=1, default=float))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
