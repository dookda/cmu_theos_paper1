"""
Inference script for Semantic Segmentation on full satellite imagery.

Two modes:
  --mode tif   : Sliding-window on a GeoTIFF (requires rasterio)
  --mode tiles : Stitch predictions from pre-cut tiles using metadata.json
                 (most reliable — model was trained on these exact tiles)

Usage:
  # GeoTIFF sliding-window (original image)
  python scripts/predict.py --mode tif \
      --model unet --encoder mobilenet_v2 \
      --checkpoint train_output/best_unet_mobilenet_v2.pth \
      --input theos/theos_4326.tif \
      --output results/prediction.tif

  # Tile-stitch (using labelled tiles already on disk)
  python scripts/predict.py --mode tiles \
      --model unet --encoder mobilenet_v2 \
      --checkpoint train_output/best_unet_mobilenet_v2.pth \
      --tiles-dir data/tiles \
      --metadata data/tiles/metadata.json \
      --output results/prediction.tif
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import cv2
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Must match config.yaml classes order
CLASS_NAMES = ["background", "vegetation", "water", "urban",
               "agriculture", "bare_soil", "road", "grass"]
CLASS_COLORS = np.array([
    [0,   0,   0],
    [0,   255, 0],
    [0,   0,   255],
    [255, 0,   0],
    [255, 255, 0],
    [139, 69,  19],
    [128, 128, 128],
    [34,  197, 94],
], dtype=np.uint8)

# Must match albumentations A.Normalize used during training
# (4th value = NIR stats from utils/dataset.py when --use-nir)
def get_mean_std(channels: int):
    mean = np.array([0.485, 0.456, 0.406, 0.5][:channels], dtype=np.float32).reshape(channels, 1, 1)
    std  = np.array([0.229, 0.224, 0.225, 0.25][:channels], dtype=np.float32).reshape(channels, 1, 1)
    return mean, std


MEAN, STD = get_mean_std(3)


def load_model(model_name: str, encoder: str, checkpoint_path: str,
               num_classes: int, in_channels: int, device: torch.device):
    from models.unet import create_unet, create_unetpp
    from models.deeplabv3 import create_deeplabv3
    from models.hrnet import create_hrnet

    if model_name == "unet":
        model = create_unet(encoder, None, in_channels, num_classes)
    elif model_name == "unetpp":
        model = create_unetpp(encoder, None, in_channels, num_classes)
    elif model_name == "deeplabv3":
        model = create_deeplabv3(encoder, None, in_channels, num_classes)
    elif model_name == "hrnet":
        model = create_hrnet(encoder, None, in_channels, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model = model.to(device).eval()
    print(f"Loaded {model_name}+{encoder} from {checkpoint_path}")
    return model


def normalize_uint8(tile_bgr: np.ndarray, nir: np.ndarray = None) -> np.ndarray:
    """uint8 BGR (H,W,3) [+NIR (H,W)] → float32 (C,H,W) with ImageNet stats."""
    tile = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tile = tile.transpose(2, 0, 1)
    if nir is not None:
        tile = np.concatenate([tile, nir[np.newaxis].astype(np.float32) / 255.0])
    mean, std = get_mean_std(tile.shape[0])
    return (tile - mean) / std


def normalize_geotiff_band(band: np.ndarray, lo_pct=2, hi_pct=98) -> np.ndarray:
    """Stretch a single band to [0,1] with percentile clipping."""
    lo, hi = np.percentile(band[band > 0], [lo_pct, hi_pct]) if np.any(band > 0) else (0, 1)
    if hi == lo:
        return np.zeros_like(band, dtype=np.float32)
    return np.clip((band.astype(np.float32) - lo) / (hi - lo), 0, 1)


def normalize_geotiff_tile(tile_data: np.ndarray) -> np.ndarray:
    """(C, H, W) uint16/float → (C, H, W) float32 with ImageNet stats."""
    mean, std = get_mean_std(tile_data.shape[0])
    return (np.stack([normalize_geotiff_band(band) for band in tile_data]) - mean) / std


@torch.no_grad()
def _run_batch(model, tiles_batch, device):
    """Run model on a list of (C,H,W) arrays; return (N,C,H,W) soft probabilities."""
    batch = torch.from_numpy(np.stack(tiles_batch)).float().to(device)
    # softmax required for correct probability averaging in overlapping tiles
    return torch.softmax(model(batch), dim=1).cpu().numpy()


def _flush(tiles_batch, positions_batch, model, device, prob_map, count_map):
    """Blend a batch of tiles into prob_map/count_map, then clear the lists."""
    if not tiles_batch:
        return
    probs = _run_batch(model, tiles_batch, device)   # (N, C, H, W)
    for prob, (px, py, tw, th) in zip(probs, positions_batch):
        prob_map[:, py:py+th, px:px+tw] += prob[:, :th, :tw]
        count_map[py:py+th, px:px+tw]   += 1.0
    tiles_batch.clear()
    positions_batch.clear()


@torch.no_grad()
def predict_from_tiles(model, tiles_dir, metadata_path, output_path,
                       tile_size=512, num_classes=8, batch_size=4, device=None,
                       in_channels=3):
    """
    Run inference on every tile in metadata.json and stitch back into a
    full-resolution probability map → argmax.
    Most reliable mode because the model was trained on these exact tiles.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(metadata_path) as f:
        meta = json.load(f)

    max_x = max(m["x_off"] for m in meta) + tile_size
    max_y = max(m["y_off"] for m in meta) + tile_size
    print(f"Canvas: {max_x} x {max_y} pixels, {len(meta)} tiles")

    prob_map  = np.zeros((num_classes, max_y, max_x), dtype=np.float32)
    count_map = np.zeros((max_y, max_x), dtype=np.float32)
    tiles_batch, positions_batch = [], []

    for entry in tqdm(meta, desc="Tiles"):
        img = cv2.imread(os.path.join(tiles_dir, entry["filename"]))
        if img is None:
            continue

        if img.shape[0] != tile_size or img.shape[1] != tile_size:
            img = cv2.resize(img, (tile_size, tile_size))

        nir = None
        if in_channels == 4:
            nir_path = os.path.join(
                tiles_dir, entry["filename"].replace(".png", "_nir.png"))
            nir = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)
            if nir is None:
                raise FileNotFoundError(f"in_channels=4 แต่ไม่พบ NIR: {nir_path}")
            if nir.shape[0] != tile_size or nir.shape[1] != tile_size:
                nir = cv2.resize(nir, (tile_size, tile_size))

        tiles_batch.append(normalize_uint8(img, nir))
        positions_batch.append((entry["x_off"], entry["y_off"], tile_size, tile_size))

        if len(tiles_batch) >= batch_size:
            _flush(tiles_batch, positions_batch, model, device, prob_map, count_map)

    _flush(tiles_batch, positions_batch, model, device, prob_map, count_map)

    prediction = (prob_map / np.maximum(count_map, 1e-6)[np.newaxis]).argmax(axis=0).astype(np.uint8)
    _save_results(prediction, output_path, num_classes)
    return prediction


@torch.no_grad()
def predict_full_tif(model, tif_path, output_path, tile_size=512, overlap=64,
                     num_classes=8, batch_size=4, device=None, in_channels=3):
    """Sliding-window inference on a GeoTIFF. Requires: pip install rasterio"""
    try:
        import rasterio
        from rasterio.windows import Window
    except ImportError:
        raise ImportError("rasterio is required for --mode tif. "
                          "Install with: pip install rasterio")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    step = tile_size - overlap

    with rasterio.open(tif_path) as src:
        W, H = src.width, src.height
        print(f"Image: {W} x {H}, {src.count} bands, CRS={src.crs}")

        n_cols = max(1, (W - overlap + step - 1) // step)
        n_rows = max(1, (H - overlap + step - 1) // step)
        print(f"Sliding window: {n_rows} rows x {n_cols} cols (tile={tile_size}, step={step})")

        prob_map  = np.zeros((num_classes, H, W), dtype=np.float32)
        count_map = np.zeros((H, W), dtype=np.float32)
        tiles_batch, positions_batch = [], []

        pbar = tqdm(total=n_rows * n_cols, desc="Sliding window")
        for row in range(n_rows):
            for col in range(n_cols):
                x_off = min(col * step, max(0, W - tile_size))
                y_off = min(row * step, max(0, H - tile_size))
                tw = min(tile_size, W - x_off)
                th = min(tile_size, H - y_off)

                bands = [1, 2, 3, 4][:in_channels]
                if src.count < in_channels:
                    raise ValueError(f"in_channels={in_channels} แต่ TIF มี {src.count} bands")
                tile_data = src.read(bands, window=Window(x_off, y_off, tw, th))

                # Skip no-data tiles
                if np.mean(tile_data[0] > 0) < 0.10:
                    pbar.update(1)
                    continue

                if tw < tile_size or th < tile_size:
                    padded = np.zeros((in_channels, tile_size, tile_size), dtype=tile_data.dtype)
                    padded[:, :th, :tw] = tile_data
                    tile_data = padded

                tiles_batch.append(normalize_geotiff_tile(tile_data))
                positions_batch.append((x_off, y_off, tw, th))

                if len(tiles_batch) >= batch_size:
                    _flush(tiles_batch, positions_batch, model, device, prob_map, count_map)
                pbar.update(1)

        _flush(tiles_batch, positions_batch, model, device, prob_map, count_map)
        pbar.close()

        prediction = (prob_map / np.maximum(count_map, 1e-6)[np.newaxis]).argmax(axis=0).astype(np.uint8)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        profile = src.profile.copy()
        profile.update(count=1, dtype="uint8", compress="lzw")
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(prediction, 1)
        print(f"Saved GeoTIFF: {output_path}")

    _save_results(prediction, output_path, num_classes, save_label_map=False)
    return prediction


def _save_results(prediction: np.ndarray, output_path: str,
                  num_classes: int, save_label_map=True):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if save_label_map:
        label_path = output_path if output_path.endswith(".png") else os.path.splitext(output_path)[0] + "_labels.png"
        cv2.imwrite(label_path, prediction)
        print(f"Saved label map: {label_path}")

    # Vectorized colormap: single pass over the prediction array
    color_map = CLASS_COLORS[:num_classes][prediction]

    h, w = prediction.shape
    scale = min(4096 / h, 4096 / w, 1.0)
    if scale < 1.0:
        color_map = cv2.resize(color_map, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_NEAREST)

    color_path = os.path.splitext(output_path)[0] + "_color.png"
    cv2.imwrite(color_path, cv2.cvtColor(color_map, cv2.COLOR_RGB2BGR))
    print(f"Saved colour map: {color_path}")

    print("\nClass distribution:")
    for cls_id in range(num_classes):
        pct = 100.0 * np.sum(prediction == cls_id) / prediction.size
        print(f"  {cls_id} {CLASS_NAMES[cls_id]:<12}: {pct:5.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict land cover from satellite imagery")
    parser.add_argument("--mode",        default="tiles", choices=["tif", "tiles"])
    parser.add_argument("--model",       default="unet", choices=["unet", "unetpp", "deeplabv3", "hrnet"])
    parser.add_argument("--encoder",     default="mobilenet_v2")
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--input",       default="theos/theos_4326.tif",
                        help="[--mode tif] Input GeoTIFF path")
    parser.add_argument("--tiles-dir",   default="data/tiles",
                        help="[--mode tiles] Directory containing tile PNGs")
    parser.add_argument("--metadata",    default="data/tiles/metadata.json",
                        help="[--mode tiles] metadata.json with x_off/y_off")
    parser.add_argument("--output",      default=None)
    parser.add_argument("--tile-size",   type=int, default=512)
    parser.add_argument("--overlap",     type=int, default=64, help="[--mode tif only]")
    parser.add_argument("--batch-size",  type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--in-channels", type=int, default=3)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}")

    output_path = args.output or f"results/{args.model}_{args.encoder}_prediction.tif"
    model = load_model(args.model, args.encoder, args.checkpoint,
                       args.num_classes, args.in_channels, device)

    if args.mode == "tiles":
        predict_from_tiles(model, args.tiles_dir, args.metadata, output_path,
                           tile_size=args.tile_size, num_classes=args.num_classes,
                           batch_size=args.batch_size, device=device,
                           in_channels=args.in_channels)
    else:
        predict_full_tif(model, args.input, output_path,
                         tile_size=args.tile_size, overlap=args.overlap,
                         num_classes=args.num_classes, batch_size=args.batch_size,
                         device=device, in_channels=args.in_channels)
