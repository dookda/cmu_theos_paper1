"""
===========================================================
สคริปต์สร้าง Offline Augmented Dataset
สำหรับเทรนทุกโมเดล (segmentation / YOLO / Mask R-CNN)
===========================================================
อ่าน tiles + semantic masks + splits.json จาก data root เดิม
แล้วสร้าง data root ใหม่ที่มี augmented variants เฉพาะ train split
(val/test คงเดิมทุกไฟล์ เพื่อเปรียบเทียบกับ baseline ได้ตรง ๆ)

YOLO/COCO labels ไม่ต้องสร้างที่นี่ -- hpc/run_all_models.py จะเรียก
scripts/export_labels.py สร้างจาก semantic masks ของ root ใหม่เอง

การใช้งาน:
    python scripts/augment_dataset.py \
        --data-root /project/gisnorth/cmu_theos_ai_data \
        --output-root /project/gisnorth/cmu_theos_ai_data_aug \
        --variants 3

    # ทดสอบกับ 20 tiles แรก
    python scripts/augment_dataset.py --limit 20 ...
"""

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import albumentations as A


def build_geo_transform():
    """Geometric augmentation -- ใช้ ReplayCompose เพื่อ replay ชุด transform
    เดียวกันกับ NIR band ได้ (mask albumentations จัดการ nearest ให้เอง)"""
    return A.ReplayCompose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.75),
        A.ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.1, rotate_limit=15,
            border_mode=cv2.BORDER_REFLECT_101, p=0.7,
        ),
    ])


def build_color_transform():
    """Radiometric augmentation -- ใช้กับ RGB เท่านั้น (NIR คงค่าเดิม)"""
    return A.Compose([
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1),
            A.RandomGamma(gamma_limit=(80, 120), p=1),
        ], p=0.8),
        A.OneOf([
            A.GaussNoise(p=1),
            A.GaussianBlur(blur_limit=(3, 5), p=1),
            A.CLAHE(clip_limit=2, p=1),
        ], p=0.3),
    ])


def link_or_copy(src: Path, dst: Path):
    """hardlink ถ้าอยู่ filesystem เดียวกัน (เร็ว/ไม่เปลืองที่) ไม่งั้น copy"""
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def resolve_labels_dir(data_root: Path) -> Path:
    for candidate in (data_root / "labels" / "semantic",
                      data_root / "semantic_labels",
                      data_root / "labels"):
        if candidate.is_dir():
            return candidate
    raise SystemExit(f"ไม่พบ semantic labels ใน {data_root}")


def main():
    parser = argparse.ArgumentParser(description="สร้าง offline augmented dataset")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variants", type=int, default=3,
                        help="จำนวน augmented variants ต่อ 1 train tile")
    parser.add_argument("--nir-dir", type=Path, default=None,
                        help="โฟลเดอร์ tile_XXXXX_nir.png (จาก make_nir_tiles.py) -- "
                             "ถ้าระบุ จะ copy/augment NIR คู่กับ RGB ทุก tile")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None,
                        help="จำกัดจำนวน train tiles (สำหรับทดสอบ)")
    args = parser.parse_args()

    src_tiles = args.data_root / "tiles"
    src_labels = resolve_labels_dir(args.data_root)
    splits_path = src_tiles / "splits.json"
    if not splits_path.exists():
        raise SystemExit(f"ไม่พบ {splits_path}")

    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    # tiles ที่มี label แล้วแต่ยังไม่อยู่ใน split ไหนเลย -> เพิ่มเข้า train
    in_splits = {f for files in splits.values() for f in files}
    unsplit = sorted(
        f for f in os.listdir(src_labels)
        if f.endswith(".png") and f not in in_splits and (src_tiles / f).exists()
    )
    if unsplit:
        print(f"พบ labeled tiles นอก splits {len(unsplit)} ไฟล์ -> เพิ่มเข้า train: {unsplit}")
        splits["train"] = splits["train"] + unsplit

    train_files = splits["train"][: args.limit] if args.limit else splits["train"]

    out_tiles = args.output_root / "tiles"
    out_labels = args.output_root / "semantic_labels"
    out_tiles.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    def nir_src_path(fname):
        return args.nir_dir / fname.replace(".png", "_nir.png") if args.nir_dir else None

    # 1) copy/link ไฟล์ต้นฉบับทุก split (รวม NIR ถ้าเปิดใช้)
    n_orig = 0
    for split_name, files in splits.items():
        use_files = train_files if split_name == "train" else files
        for fname in use_files:
            tile_src = src_tiles / fname
            label_src = src_labels / fname
            if not tile_src.exists() or not label_src.exists():
                print(f"  ข้าม {fname} (ไม่พบ tile หรือ label)")
                continue
            if args.nir_dir:
                nir_src = nir_src_path(fname)
                if not nir_src.exists():
                    raise SystemExit(f"ไม่พบ NIR ของ {fname}: {nir_src}")
                link_or_copy(nir_src, out_tiles / nir_src.name)
            link_or_copy(tile_src, out_tiles / fname)
            link_or_copy(label_src, out_labels / fname)
            n_orig += 1

    # 2) สร้าง augmented variants เฉพาะ train
    geo = build_geo_transform()
    color = build_color_transform()
    aug_names = []
    for i, fname in enumerate(train_files):
        tile_src = src_tiles / fname
        label_src = src_labels / fname
        if not tile_src.exists() or not label_src.exists():
            continue
        image = cv2.imread(str(tile_src), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(label_src), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            print(f"  ข้าม {fname} (อ่านไฟล์ไม่ได้)")
            continue
        nir = None
        if args.nir_dir:
            nir = cv2.imread(str(nir_src_path(fname)), cv2.IMREAD_GRAYSCALE)
            if nir is None:
                raise SystemExit(f"อ่าน NIR ของ {fname} ไม่ได้")

        stem = Path(fname).stem
        for v in range(1, args.variants + 1):
            # seed ต่อ variant -> ผลซ้ำได้ (reproducible)
            random.seed(args.seed + i * 100 + v)
            np.random.seed(args.seed + i * 100 + v)
            g = geo(image=image, mask=mask)
            out_img = color(image=g["image"])["image"]
            aug_name = f"{stem}_aug{v}.png"
            cv2.imwrite(str(out_tiles / aug_name), out_img)
            cv2.imwrite(str(out_labels / aug_name), g["mask"])
            if nir is not None:
                # replay geometric transforms เดิมเป๊ะ ๆ กับ NIR (ไม่ทำ color)
                g_nir = A.ReplayCompose.replay(g["replay"], image=nir)
                cv2.imwrite(str(out_tiles / f"{stem}_aug{v}_nir.png"), g_nir["image"])
            aug_names.append(aug_name)

        if (i + 1) % 200 == 0:
            print(f"  augmented {i + 1}/{len(train_files)} tiles...", flush=True)

    # 3) splits.json ใหม่ (train = ต้นฉบับ + augmented, val/test เดิม)
    new_splits = {
        "train": list(train_files) + aug_names,
        "val": splits["val"],
        "test": splits["test"],
    }
    with open(out_tiles / "splits.json", "w", encoding="utf-8") as f:
        json.dump(new_splits, f, indent=2)

    # 4) symlink ภาพ scene .tif สำหรับขั้น predict
    for tif in args.data_root.glob("*.tif"):
        dst = args.output_root / tif.name
        if not dst.exists():
            dst.symlink_to(tif)

    print("\n===== สรุป =====")
    print(f"ต้นฉบับ (ทุก split): {n_orig} tiles")
    print(f"Augmented (train):   {len(aug_names)} tiles ({args.variants} variants/tile)")
    print(f"Train รวม:           {len(new_splits['train'])}")
    print(f"Val/Test:            {len(new_splits['val'])}/{len(new_splits['test'])} (คงเดิม)")
    print(f"Output root:         {args.output_root}")

    # ตรวจความครบถ้วน: ทุกไฟล์ใน splits ต้องมีทั้ง tile และ label (+NIR ถ้าเปิดใช้)
    missing = [f for files in new_splits.values() for f in files
               if not (out_tiles / f).exists() or not (out_labels / f).exists()
               or (args.nir_dir and not (out_tiles / f.replace(".png", "_nir.png")).exists())]
    if missing:
        print(f"⚠️  ไฟล์หาย {len(missing)} รายการ เช่น {missing[:5]}")
        sys.exit(1)
    suffix = " (+NIR ครบทุกไฟล์)" if args.nir_dir else ""
    print(f"✅ ตรวจสอบครบถ้วน: ทุก tile มี label คู่กัน{suffix}")


if __name__ == "__main__":
    main()
