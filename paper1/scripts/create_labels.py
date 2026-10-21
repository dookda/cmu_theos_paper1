"""
===========================================================
สคริปต์สร้าง Label สำหรับภาพดาวเทียม
===========================================================
ใช้วิธี Unsupervised Classification (K-Means)
เพื่อสร้าง label เบื้องต้น ก่อนที่จะ refine ด้วยมือ
"""

import os
import sys
import argparse
import numpy as np
import cv2
from sklearn.cluster import KMeans, MiniBatchKMeans
from pathlib import Path
from tqdm import tqdm
import json
import yaml


def create_labels_kmeans(
    tiles_dir: str,
    labels_dir: str,
    n_clusters: int = 7,
    use_mini_batch: bool = True,
    sample_ratio: float = 0.1,
):
    """
    สร้าง label masks ด้วย K-Means clustering
    สำหรับเป็น label เบื้องต้น (pseudo-labels)

    ⚠️ หมายเหตุ: labels เหล่านี้เป็นเพียง approximation
    ควร refine ด้วยมือก่อนใช้เทรนโมเดลจริง

    Parameters:
    -----------
    tiles_dir : str - โฟลเดอร์ tiles
    labels_dir : str - โฟลเดอร์เก็บ labels
    n_clusters : int - จำนวน clusters (classes)
    use_mini_batch : bool - ใช้ MiniBatchKMeans (เร็วกว่า)
    sample_ratio : float - สัดส่วนพิกเซลสำหรับ fit KMeans
    """
    os.makedirs(labels_dir, exist_ok=True)

    # รวบรวม tiles
    tile_files = sorted([
        f for f in os.listdir(tiles_dir)
        if f.endswith('.png') and 'nir' not in f
    ])

    if not tile_files:
        print("❌ ไม่พบไฟล์ tiles!")
        return

    print(f"📊 สร้าง labels ด้วย K-Means (k={n_clusters})")
    print(f"   จำนวน tiles: {len(tile_files)}")

    # ขั้นตอนที่ 1: สุ่มตัวอย่างพิกเซลสำหรับ fit KMeans
    print("\n🔄 ขั้นตอนที่ 1: สุ่มตัวอย่างพิกเซล...")
    all_pixels = []

    for filename in tqdm(tile_files[:min(50, len(tile_files))], desc="  Sampling"):
        img = cv2.imread(os.path.join(tiles_dir, filename), cv2.IMREAD_COLOR)
        if img is None:
            continue

        pixels = img.reshape(-1, 3).astype(np.float32)
        n_sample = int(len(pixels) * sample_ratio)
        idx = np.random.choice(len(pixels), n_sample, replace=False)
        all_pixels.append(pixels[idx])

    all_pixels = np.vstack(all_pixels)
    print(f"   จำนวนพิกเซลตัวอย่าง: {len(all_pixels):,}")

    # ขั้นตอนที่ 2: fit KMeans
    print(f"\n🔄 ขั้นตอนที่ 2: Fit K-Means...")
    if use_mini_batch:
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, random_state=42, batch_size=5000)
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

    kmeans.fit(all_pixels)
    print(f"   ✅ Fit เสร็จแล้ว!")
    print(f"   Cluster centers:\n{kmeans.cluster_centers_}")

    # ขั้นตอนที่ 3: สร้าง label masks
    print(f"\n🔄 ขั้นตอนที่ 3: สร้าง label masks...")

    for filename in tqdm(tile_files, desc="  Creating labels"):
        img = cv2.imread(os.path.join(tiles_dir, filename), cv2.IMREAD_COLOR)
        if img is None:
            continue

        h, w, c = img.shape
        pixels = img.reshape(-1, 3).astype(np.float32)

        labels = kmeans.predict(pixels)
        label_mask = labels.reshape(h, w).astype(np.uint8)

        label_path = os.path.join(labels_dir, filename)
        cv2.imwrite(label_path, label_mask)

    print(f"\n✅ สร้าง labels เสร็จแล้ว!")
    print(f"   จำนวน labels: {len(tile_files)}")
    print(f"   บันทึกที่: {labels_dir}")
    print(f"\n⚠️  หมายเหตุ: labels นี้เป็น pseudo-labels จาก K-Means")
    print(f"   ควร refine ด้วยเครื่องมือ annotation เช่น:")
    print(f"   - QGIS (ฟรี)")
    print(f"   - Label Studio (ฟรี)")
    print(f"   - CVAT (ฟรี)")
    print(f"   - Roboflow (ฟรี tier)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="สร้าง labels ด้วย K-Means")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--clusters", type=int, default=None,
                        help="จำนวน clusters (override config)")

    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    n_clusters = args.clusters or config["num_classes"]

    create_labels_kmeans(
        tiles_dir=config["data"]["tiles_dir"],
        labels_dir=config["data"]["labels_dir"],
        n_clusters=n_clusters,
    )
