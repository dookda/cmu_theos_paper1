"""
===========================================================
สคริปต์ตัดภาพดาวเทียมเป็น Tiles
สำหรับเตรียมข้อมูล Training
===========================================================
ขั้นตอน:
1. อ่านภาพ GeoTIFF ขนาดใหญ่
2. ตัดเป็น tiles ขนาด 512x512 พิกเซล
3. บันทึกเป็นไฟล์ PNG/TIF แยกแต่ละ tile
4. สร้าง metadata (พิกัด, bounding box) สำหรับแต่ละ tile
"""

import os
import sys
import argparse
import numpy as np
import rasterio
from rasterio.windows import Window
from pathlib import Path
import json
from tqdm import tqdm
import yaml
import cv2


def load_config(config_path="configs/config.yaml"):
    """โหลดไฟล์ config"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def normalize_band(band, min_val=None, max_val=None):
    """ปรับค่าพิกเซลให้อยู่ในช่วง 0-255 (8-bit)"""
    if min_val is None:
        min_val = np.nanmin(band)
    if max_val is None:
        max_val = np.nanmax(band)

    band = band.astype(np.float32)
    band = (band - min_val) / (max_val - min_val + 1e-8)
    band = np.clip(band * 255, 0, 255).astype(np.uint8)
    return band


def create_tiles(
    tif_path: str,
    output_dir: str,
    tile_size: int = 512,
    overlap: int = 64,
    bands_to_use: list = None,
    normalize: bool = True,
    min_valid_ratio: float = 0.5
):
    """
    ตัดภาพ GeoTIFF เป็น tiles

    Parameters:
    -----------
    tif_path : str - พาธไปยังไฟล์ GeoTIFF
    output_dir : str - โฟลเดอร์สำหรับเก็บ tiles
    tile_size : int - ขนาดของแต่ละ tile (พิกเซล)
    overlap : int - จำนวนพิกเซลที่ซ้อนทับ
    bands_to_use : list - bands ที่ต้องการใช้ (เริ่มจาก 1)
    normalize : bool - ทำ normalization หรือไม่
    min_valid_ratio : float - สัดส่วนขั้นต่ำของพิกเซลที่มีค่า (ไม่ใช่ nodata)
    """
    os.makedirs(output_dir, exist_ok=True)
    metadata_list = []

    with rasterio.open(tif_path) as src:
        print(f"📡 เปิดไฟล์: {tif_path}")
        print(f"   ขนาด: {src.width} x {src.height} พิกเซล")
        print(f"   จำนวน bands: {src.count}")
        print(f"   ระบบพิกัด: {src.crs}")
        print(f"   ชนิดข้อมูล: {src.dtypes}")

        if bands_to_use is None:
            # ใช้ไม่เกิน 4 bands
            bands_to_use = list(range(1, min(src.count + 1, 5)))

        step = tile_size - overlap
        n_cols = (src.width - overlap) // step
        n_rows = (src.height - overlap) // step
        total_tiles = n_cols * n_rows

        print(f"\n🔲 จะตัดเป็น tiles ขนาด {tile_size}x{tile_size}")
        print(f"   overlap: {overlap} พิกเซล")
        print(f"   จำนวน tiles ทั้งหมด (ก่อนกรอง): {total_tiles}")

        tile_idx = 0
        valid_count = 0

        for row in tqdm(range(n_rows), desc="แถว"):
            for col in range(n_cols):
                # คำนวณตำแหน่งของ window
                x_off = col * step
                y_off = row * step

                # ตรวจสอบขอบเขต
                if x_off + tile_size > src.width or y_off + tile_size > src.height:
                    continue

                window = Window(x_off, y_off, tile_size, tile_size)

                # อ่านข้อมูล
                tile_data = src.read(bands_to_use, window=window)

                # ตรวจสอบว่ามีข้อมูลเพียงพอ (ไม่ใช่ nodata ทั้งหมด)
                valid_pixels = np.sum(tile_data[0] > 0)
                total_pixels = tile_size * tile_size
                valid_ratio = valid_pixels / total_pixels

                if valid_ratio < min_valid_ratio:
                    continue

                # Normalize
                if normalize:
                    tile_normalized = np.zeros(
                        (len(bands_to_use), tile_size, tile_size), dtype=np.uint8
                    )
                    for i, band_data in enumerate(tile_data):
                        tile_normalized[i] = normalize_band(band_data)
                    tile_data = tile_normalized

                # บันทึก tile (CHW -> HWC สำหรับ OpenCV)
                tile_hwc = np.transpose(tile_data, (1, 2, 0))

                # ถ้ามี 4 bands ให้บันทึกเป็น 4 channels
                tile_filename = f"tile_{tile_idx:05d}.png"
                tile_path = os.path.join(output_dir, tile_filename)

                if tile_hwc.shape[2] == 4:
                    # THEOS band order: R,G,B,NIR → สลับเป็น BGR สำหรับ OpenCV
                    bgr = tile_hwc[:, :, [2, 1, 0]]
                    cv2.imwrite(tile_path, bgr)
                    # บันทึก NIR band แยก
                    nir_path = os.path.join(
                        output_dir, f"tile_{tile_idx:05d}_nir.png")
                    cv2.imwrite(nir_path, tile_hwc[:, :, 3])
                elif tile_hwc.shape[2] == 3:
                    # THEOS band order: R,G,B → สลับเป็น BGR สำหรับ OpenCV
                    bgr = tile_hwc[:, :, [2, 1, 0]]
                    cv2.imwrite(tile_path, bgr)
                else:
                    cv2.imwrite(tile_path, tile_hwc[:, :, 0])

                # เก็บ metadata
                transform = src.window_transform(window)
                bounds = rasterio.windows.bounds(window, src.transform)
                metadata_list.append({
                    "tile_id": tile_idx,
                    "filename": tile_filename,
                    "x_off": x_off,
                    "y_off": y_off,
                    "bounds": {
                        "left": bounds[0],
                        "bottom": bounds[1],
                        "right": bounds[2],
                        "top": bounds[3],
                    },
                    "valid_ratio": round(valid_ratio, 4),
                })

                tile_idx += 1
                valid_count += 1

        print(f"\n✅ สร้าง tiles สำเร็จ: {valid_count} tiles")
        print(f"   บันทึกที่: {output_dir}")

    # บันทึก metadata
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata_list, f, indent=2)
    print(f"   Metadata: {meta_path}")

    return metadata_list


def split_dataset(
    tiles_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
):
    """
    แบ่งข้อมูลเป็น train/val/test
    """
    from sklearn.model_selection import train_test_split

    # อ่าน metadata
    meta_path = os.path.join(tiles_dir, "metadata.json")
    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    indices = list(range(len(metadata)))

    # แบ่ง train / (val+test)
    train_idx, temp_idx = train_test_split(
        indices, test_size=(val_ratio + test_ratio), random_state=seed
    )

    # แบ่ง val / test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=(1 - val_size), random_state=seed
    )

    splits = {
        "train": [metadata[i]["filename"] for i in train_idx],
        "val": [metadata[i]["filename"] for i in val_idx],
        "test": [metadata[i]["filename"] for i in test_idx],
    }

    split_path = os.path.join(tiles_dir, "splits.json")
    with open(split_path, 'w') as f:
        json.dump(splits, f, indent=2)

    print(f"\n📊 แบ่งข้อมูล:")
    print(f"   Train: {len(train_idx)} tiles ({train_ratio*100:.0f}%)")
    print(f"   Val:   {len(val_idx)} tiles ({val_ratio*100:.0f}%)")
    print(f"   Test:  {len(test_idx)} tiles ({test_ratio*100:.0f}%)")
    print(f"   บันทึกที่: {split_path}")

    return splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ตัดภาพดาวเทียมเป็น tiles สำหรับ training")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="พาธไปยังไฟล์ config")
    parser.add_argument("--tif", type=str, default=None,
                        help="พาธไปยังไฟล์ GeoTIFF (override config)")
    parser.add_argument("--output", type=str, default=None,
                        help="โฟลเดอร์เก็บ tiles (override config)")
    parser.add_argument("--tile-size", type=int, default=None,
                        help="ขนาด tile (override config)")
    parser.add_argument("--no-split", action="store_true",
                        help="ไม่ต้องแบ่ง train/val/test")

    args = parser.parse_args()
    config = load_config(args.config)

    tif_path = args.tif or config["data"]["source_tif"]
    output_dir = args.output or config["data"]["tiles_dir"]
    tile_size = args.tile_size or config["data"]["tile_size"]
    overlap = config["data"]["overlap"]

    print("=" * 60)
    print("  🛰️  เตรียมข้อมูลภาพดาวเทียม THEOS")
    print("=" * 60)

    # ตัด tiles
    metadata = create_tiles(
        tif_path=tif_path,
        output_dir=output_dir,
        tile_size=tile_size,
        overlap=overlap,
    )

    # แบ่ง train/val/test
    if not args.no_split and len(metadata) > 0:
        split_dataset(
            tiles_dir=output_dir,
            train_ratio=config["data"]["train_ratio"],
            val_ratio=config["data"]["val_ratio"],
            test_ratio=config["data"]["test_ratio"],
        )

    print("\n✅ เตรียมข้อมูลเสร็จสมบูรณ์!")
