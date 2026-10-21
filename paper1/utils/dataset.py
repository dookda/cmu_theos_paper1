"""
===========================================================
Dataset & DataLoader สำหรับ Semantic Segmentation
รองรับ DeepLabV3, UNet, HRNet
===========================================================
"""

import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path


class SegmentationDataset(Dataset):
    """
    Dataset สำหรับ Semantic Segmentation

    โครงสร้างโฟลเดอร์:
    tiles_dir/
        tile_00000.png
        tile_00001.png
        ...
    labels_dir/
        tile_00000.png   (mask: แต่ละพิกเซลเป็นค่า class index 0-N)
        tile_00001.png
        ...
    """

    def __init__(
        self,
        tiles_dir: str,
        labels_dir: str,
        file_list: list = None,
        num_classes: int = 7,
        transform=None,
        use_nir: bool = False,
        class_remap: dict = None,
        return_name: bool = False,
        nir_mode: str = None,
    ):
        """
        Parameters:
        -----------
        tiles_dir : str - โฟลเดอร์เก็บ tiles (ภาพ)
        labels_dir : str - โฟลเดอร์เก็บ label masks
        file_list : list - ลิสต์ชื่อไฟล์ (จาก splits.json)
        num_classes : int - จำนวน classes
        transform : albumentations - data augmentation
        use_nir : bool - ใช้ NIR band ด้วยหรือไม่
        class_remap : dict - {src_index: dst_index} ใช้ยุบคลาสก่อนเทรน
                             (paper1 re-run: {7: 0} ยุบ grass เข้า background)
        """
        self.tiles_dir = tiles_dir
        self.labels_dir = labels_dir
        self.num_classes = num_classes
        self.transform = transform
        self.use_nir = use_nir
        # addendum G: "nrg" = NIR,R,G as 3 channels; "lowpass" = RGB + NIR at 2 m (4x4 mean)
        self.nir_mode = nir_mode
        self.class_remap = {int(k): int(v) for k, v in (class_remap or {}).items()}
        # per-block evaluation ต้องรู้ว่าแต่ละ sample มาจาก tile ไหน การอนุมานจาก
        # ลำดับ batch ใช้ได้เฉพาะตอน shuffle=False จึงคืนชื่อมาตรง ๆ แทน
        self.return_name = return_name

        if file_list is not None:
            self.file_list = file_list
        else:
            self.file_list = sorted([
                f for f in os.listdir(tiles_dir)
                if f.endswith('.png') and 'nir' not in f
            ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]

        # โหลดภาพ
        img_path = os.path.join(self.tiles_dir, filename)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.nir_mode in ("nrg", "lowpass"):
            nir = cv2.imread(os.path.join(self.tiles_dir, filename.replace('.png', '_nir.png')),
                             cv2.IMREAD_GRAYSCALE)
            if nir is None:
                raise FileNotFoundError(f"{filename}: NIR tile missing for nir_mode={self.nir_mode}")
            if self.nir_mode == "nrg":
                image = np.dstack([nir, image[..., 0], image[..., 1]])
            else:
                h, w = nir.shape
                nir = cv2.resize(cv2.resize(nir, (w // 4, h // 4), interpolation=cv2.INTER_AREA),
                                 (w, h), interpolation=cv2.INTER_LINEAR)
                image = np.dstack([image, nir])
        # โหลด NIR (ถ้าใช้)
        elif self.use_nir:
            nir_filename = filename.replace('.png', '_nir.png')
            nir_path = os.path.join(self.tiles_dir, nir_filename)
            if os.path.exists(nir_path):
                nir = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)
                image = np.dstack([image, nir])

        # โหลด label mask
        label_path = os.path.join(self.labels_dir, filename)
        if os.path.exists(label_path):
            mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        else:
            # ถ้าไม่มี label ให้ใช้ dummy mask (สำหรับทดสอบ)
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # ยุบคลาสตาม remap ก่อนทุกอย่าง แล้วยืนยันว่าไม่มี index หลุดช่วง —
        # 280 runs พึ่งจุดนี้จุดเดียว ถ้าพลาดจะเงียบและพังทั้งชุด
        for src, dst in self.class_remap.items():
            mask[mask == src] = dst
        # 255 = unlabelled (Roi Et probe, addendum C); SegmentationMetrics skips it
        if mask[mask != 255].max(initial=0) >= self.num_classes:
            raise ValueError(
                f"{filename}: class index {mask.max()} >= num_classes "
                f"{self.num_classes} (remap={self.class_remap})")

        # Apply augmentation
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).long()

        if self.return_name:
            return image, mask, filename
        return image, mask


def get_train_transform(img_size=512, use_nir=False):
    """Augmentation สำหรับ training"""
    # HueSaturationValue รองรับแค่ 1/3 channels -- ห้ามใช้กับ 4-channel (RGB+NIR)
    color_ops = [
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
        A.RandomGamma(gamma_limit=(80, 120), p=1) if use_nir else
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1),
    ]
    return A.Compose([
        A.RandomCrop(width=img_size, height=img_size, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf(color_ops, p=0.5),
        A.OneOf([
            A.GaussNoise(p=1),
            A.GaussianBlur(blur_limit=(3, 5), p=1),
        ], p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406] if not use_nir else [0.485, 0.456, 0.406, 0.5],
            std=[0.229, 0.224, 0.225] if not use_nir else [0.229, 0.224, 0.225, 0.25],
        ),
        ToTensorV2(),
    ])


def get_val_transform(img_size=512, use_nir=False):
    """Augmentation สำหรับ validation (ไม่มี random transform)"""
    return A.Compose([
        A.CenterCrop(width=img_size, height=img_size, p=1.0),
        A.Normalize(
            mean=[0.485, 0.456, 0.406] if not use_nir else [0.485, 0.456, 0.406, 0.5],
            std=[0.229, 0.224, 0.225] if not use_nir else [0.229, 0.224, 0.225, 0.25],
        ),
        ToTensorV2(),
    ])


def load_spatial_splits(path: str):
    """อ่าน spatial_splits.json ที่สร้างโดย build_paper1_spatial_splits.py

    คืน (splits, remap, classes) โดย splits เป็น {name: [ไฟล์ .png]}
    ต่างจาก splits.json เดิมตรงที่มี key `external` (Scene 2) และเก็บชื่อ tile
    แบบไม่มีนามสกุล
    """
    with open(path) as f:
        payload = json.load(f)
    splits = {k: [f"{n}.png" for n in v] for k, v in payload["splits"].items()}
    return splits, payload.get("remap", {}), payload.get("classes", [])


def create_spatial_dataloaders(config: dict, splits_path: str, batch_size: int = None,
                               augment: bool = True, return_name: bool = False):
    """DataLoaders สำหรับ paper1 re-run — ใช้ spatial-block splits + external test

    คืน dict {"train","val","test","external"} ; `external` คือ Scene 2 ที่ไม่แชร์
    พิกเซลกับ train เลย ห้ามแตะจนกว่าจะตัดสินใจทุกอย่างบน val เสร็จ

    `augment=False` ปิด online augmentation จริง ๆ (ใช้ transform เดียวกับ val)
    ซึ่งรอบแรกของ paper1 ทำไม่ได้: create_dataloaders ใส่ flip/rotate/brightness/noise
    ให้ทุก run รวมถึงแขนที่รายงานว่า "no augmentation"
    """
    tiles_dir = config["data"]["tiles_dir"]
    labels_dir = config["data"]["labels_dir"]
    tile_size = config["data"]["tile_size"]
    batch_size = batch_size or config["segmentation"]["batch_size"]
    use_nir = config["data"].get("use_nir", False)
    nir_mode = config["data"].get("nir_mode")
    four = use_nir or nir_mode == "lowpass"

    splits, remap, classes = load_spatial_splits(splits_path)
    num_classes = len(classes) or config["num_classes"]

    loaders = {}
    for name, files in splits.items():
        train_side = name == "train"
        ds = SegmentationDataset(
            tiles_dir=tiles_dir,
            labels_dir=labels_dir,
            file_list=files,
            num_classes=num_classes,
            transform=(get_train_transform(tile_size, use_nir=four)
                       if train_side and augment
                       else get_val_transform(tile_size, use_nir=four)),
            use_nir=use_nir,
            class_remap=remap,
            return_name=return_name and not train_side,
            nir_mode=nir_mode,
        )
        # UPerNet มีสาขา pyramid pooling ที่ยุบเหลือ 1x1 ทำให้ BatchNorm พังถ้า batch
        # สุดท้ายเหลือใบเดียวตอน train จึงตัดทิ้งเฉพาะกรณีนั้น ไม่ตัดทั่วไปเพราะจะเสีย
        # ข้อมูลสอนโดยไม่จำเป็น
        drop_last = train_side and len(ds) % batch_size == 1
        loaders[name] = DataLoader(
            ds, batch_size=batch_size, shuffle=train_side,
            num_workers=0, pin_memory=False, drop_last=drop_last,
        )
    return loaders, num_classes


def create_dataloaders(config: dict):
    """
    สร้าง DataLoaders สำหรับ train/val/test
    """
    tiles_dir = config["data"]["tiles_dir"]
    labels_dir = config["data"]["labels_dir"]
    tile_size = config["data"]["tile_size"]
    batch_size = config["segmentation"]["batch_size"]
    num_classes = config["num_classes"]
    use_nir = config["data"].get("use_nir", False)
    # num_workers > 0 deadlocks when DataLoader is created inside a daemon thread
    num_workers = 0

    # โหลด splits
    splits_path = os.path.join(tiles_dir, "splits.json")
    if os.path.exists(splits_path):
        with open(splits_path, 'r') as f:
            splits = json.load(f)
    else:
        # ถ้าไม่มี splits ให้ใช้ไฟล์ทั้งหมดเป็น train
        all_files = sorted([
            f for f in os.listdir(tiles_dir)
            if f.endswith('.png') and 'nir' not in f
        ])
        splits = {"train": all_files,
                  "val": all_files[:10], "test": all_files[:10]}

    if use_nir:
        print("   NIR band enabled (4-channel input: RGBNIR)")

    # สร้าง datasets
    train_dataset = SegmentationDataset(
        tiles_dir=tiles_dir,
        labels_dir=labels_dir,
        file_list=splits["train"],
        num_classes=num_classes,
        transform=get_train_transform(tile_size, use_nir=use_nir),
        use_nir=use_nir,
    )

    val_dataset = SegmentationDataset(
        tiles_dir=tiles_dir,
        labels_dir=labels_dir,
        file_list=splits["val"],
        num_classes=num_classes,
        transform=get_val_transform(tile_size, use_nir=use_nir),
        use_nir=use_nir,
    )

    test_dataset = SegmentationDataset(
        tiles_dir=tiles_dir,
        labels_dir=labels_dir,
        file_list=splits["test"],
        num_classes=num_classes,
        transform=get_val_transform(tile_size, use_nir=use_nir),
        use_nir=use_nir,
    )

    # สร้าง DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader, test_loader
