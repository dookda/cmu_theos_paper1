"""
Derive YOLO detect/segment .txt labels or COCO-format JSON annotations from
the semantic masks in data/labels/semantic.

Only the semantic masks are stored on disk natively -- YOLO and COCO formats
are otherwise only ever produced on the fly by the labeler app's zip-export
endpoint (backend/app.py). This script reuses the same conversion functions
(backend/label_io.py) to write the derived labels directly to local disk,
for use by scripts/train_yolo.py and scripts/train_maskrcnn.py on HPC where
the repo + data already live on the same filesystem (no zip step needed).

Usage:
  python scripts/export_labels.py --format yolo_detect  --output-dir data/yolo_labels_detect
  python scripts/export_labels.py --format yolo_segment --output-dir data/yolo_labels_segment
  python scripts/export_labels.py --format coco          --output-dir data/coco_annotations
"""

import os
import sys
import json
import argparse
import cv2
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.label_io import mask_to_yolo_detect, mask_to_yolo_segment, mask_to_coco_annotations


def load_classes(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return [{"index": i, "name": c["name"]} for i, c in enumerate(config.get("classes", []))]


def load_splits(tiles_dir):
    splits_path = os.path.join(tiles_dir, "splits.json")
    if os.path.exists(splits_path):
        with open(splits_path, "r", encoding="utf-8") as f:
            return json.load(f)
    all_files = sorted(f for f in os.listdir(tiles_dir) if f.endswith(".png") and "nir" not in f)
    return {"train": all_files, "val": [], "test": []}


def export_yolo(args, all_files, convert_fn):
    os.makedirs(args.output_dir, exist_ok=True)
    written = 0
    for filename in tqdm(all_files, desc=args.format):
        mask_path = os.path.join(args.labels_dir, filename)
        if not os.path.exists(mask_path):
            continue
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"  skipping unreadable mask: {mask_path}", file=sys.stderr)
            continue
        lines = convert_fn(mask)
        stem = os.path.splitext(filename)[0]
        with open(os.path.join(args.output_dir, f"{stem}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        written += 1
    print(f"Wrote {written} label files to {args.output_dir}")


def export_coco(args, splits, label_classes):
    os.makedirs(args.output_dir, exist_ok=True)
    categories = [{"id": c["index"], "name": c["name"]} for c in label_classes]
    for split_name, files in splits.items():
        coco_images, coco_anns, ann_id = [], [], 1
        for img_id, filename in enumerate(tqdm(files, desc=f"coco/{split_name}"), start=1):
            coco_images.append({
                "id": img_id, "file_name": filename,
                "width": args.tile_size, "height": args.tile_size,
            })
            mask_path = os.path.join(args.labels_dir, filename)
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    anns = mask_to_coco_annotations(mask, img_id, ann_id)
                    coco_anns.extend(anns)
                    ann_id += len(anns)
        out_path = os.path.join(args.output_dir, f"{split_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "info": {"description": "Derived from data/labels/semantic", "version": "1.0"},
                "categories": categories, "images": coco_images, "annotations": coco_anns,
            }, f, indent=2)
        print(f"Wrote {out_path}  ({len(coco_images)} images, {len(coco_anns)} annotations)")


def main(args):
    splits = load_splits(args.tiles_dir)
    all_files = [f for files in splits.values() for f in files]
    classes = load_classes(args.config)
    label_classes = [c for c in classes if c["index"] != 0]

    if args.format == "yolo_detect":
        export_yolo(args, all_files, mask_to_yolo_detect)
        print(f"Classes ({len(label_classes)}): {[c['name'] for c in label_classes]}")
    elif args.format == "yolo_segment":
        export_yolo(args, all_files, mask_to_yolo_segment)
        print(f"Classes ({len(label_classes)}): {[c['name'] for c in label_classes]}")
    elif args.format == "coco":
        export_coco(args, splits, label_classes)
    else:
        raise ValueError(f"Unknown format: {args.format}")

    # Sentinel marking a fully-completed export -- callers (e.g. the hpc/
    # train/*.slurm scripts) check for this instead of just "dir is non-empty",
    # so a job killed mid-export doesn't look like a finished one on rerun.
    with open(os.path.join(args.output_dir, ".done"), "w", encoding="utf-8") as f:
        f.write("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Derive YOLO/COCO labels from semantic masks")
    parser.add_argument("--format", required=True, choices=["yolo_detect", "yolo_segment", "coco"])
    parser.add_argument("--tiles-dir", default="data/tiles")
    parser.add_argument("--labels-dir", default="data/labels/semantic")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--tile-size", type=int, default=512)
    args = parser.parse_args()
    main(args)
