"""
===========================================================
GPU Training Script — THEOS Satellite Image Segmentation
Supports: Multi-GPU (DataParallel), Mixed Precision (AMP),
          Google Colab, CMU HPC (SLURM)
===========================================================
Usage:
  # Single GPU
  python scripts/train_gpu.py --model unet --encoder resnet50 --epochs 100

  # With custom data paths (Colab / HPC)
  python scripts/train_gpu.py \
      --model unet \
      --encoder resnet50 \
      --tiles-dir /content/drive/MyDrive/theos/tiles \
      --labels-dir /content/drive/MyDrive/theos/labels/semantic \
      --output-dir /content/drive/MyDrive/theos/outputs \
      --epochs 100 --batch-size 16 --lr 0.001 --pretrained

  # Resume from checkpoint
  python scripts/train_gpu.py --model unet --resume outputs/unet_last.pth
"""

import os
import sys
import argparse
import time
import json
import random
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset import create_dataloaders, create_spatial_dataloaders, load_spatial_splits
from utils.losses import get_loss_function, BackgroundMaskedComboLoss
from utils.metrics import SegmentationMetrics


# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
def set_seed(seed: int):
    """ตรึง RNG ทุกตัวที่มีผลต่อผลลัพธ์

    รอบแรกของ paper1 ไม่ได้ตั้ง seed เลยและรัน config ละครั้งเดียว จึงแยกไม่ออกว่า
    ส่วนต่างระดับ +0.1 ถึง +1.6 จุดมาจากปัจจัยที่ทดลองหรือจาก training randomness
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN autotuning เลือก kernel ต่างกันในแต่ละรัน ทำให้ผลไม่ซ้ำแม้ seed เท่ากัน
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0)
        print(f"GPU: {name}  ({n} device(s))")
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Apple Silicon MPS")
        return torch.device("mps")
    else:
        print("CPU only")
        return torch.device("cpu")


# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────
def create_model(args, config):
    num_classes = config["num_classes"]
    in_channels = 4 if (config["data"].get("use_nir", False)
                        or config["data"].get("nir_mode") == "lowpass") else 3
    weights = "imagenet" if args.pretrained else None

    if args.model == "unet":
        from models.unet import create_unet
        model = create_unet(args.encoder, weights, in_channels, num_classes)
    elif args.model == "unetpp":
        from models.unet import create_unetpp
        model = create_unetpp(args.encoder, weights, in_channels, num_classes)
    elif args.model == "deeplabv3":
        from models.deeplabv3 import create_deeplabv3
        model = create_deeplabv3(args.encoder, weights, in_channels, num_classes)
    elif args.model == "hrnet":
        from models.hrnet import create_hrnet
        model = create_hrnet(args.encoder, weights, in_channels, num_classes)
    elif args.model == "segformer":
        from models.transformer_seg import create_segformer
        model = create_segformer(args.encoder, weights, in_channels, num_classes)
    elif args.model == "upernet":
        from models.transformer_seg import create_upernet
        model = create_upernet(args.encoder, weights, in_channels, num_classes)
    elif args.model == "satlas":
        # addendum E: remote-sensing pretraining (weights always loaded from local file)
        from models.satlas_seg import create_satlas
        model = create_satlas(num_classes, in_channels, encoder=args.encoder)
    elif args.model == "dinov3":
        from models.dinov3_seg import create_dinov3
        model = create_dinov3(num_classes, in_channels)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    total = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model.upper()} + {args.encoder}  ({total/1e6:.1f}M params)")
    return model


# ──────────────────────────────────────────────
# Config builder (works without labeler)
# ──────────────────────────────────────────────
def build_config(args):
    # Load base config if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    # Classes
    config.setdefault("classes", [{"name": "background", "color": [0, 0, 0]}])
    config["num_classes"] = len(config["classes"])

    # spatial_splits.json เป็นเจ้าของ class scheme (re-run ใช้ 7 คลาส ไม่ใช่ 8 ของ
    # config.yaml) ต้อง resolve ที่นี่ เพราะ create_model อ่าน num_classes ก่อนที่
    # dataloader จะถูกสร้าง — ถ้าปล่อยไว้ head จะมี 8 ช่องแต่ label มี 7 คลาส
    if getattr(args, "splits", None):
        with open(args.splits, "r", encoding="utf-8") as f:
            names = json.load(f).get("classes") or []
        if names:
            config["classes"] = [{"name": n} for n in names]
            config["num_classes"] = len(names)

    # Data
    config.setdefault("data", {})
    config["data"]["tiles_dir"]  = args.tiles_dir
    config["data"]["labels_dir"] = args.labels_dir
    config["data"]["tile_size"]  = args.crop_size
    config["data"]["use_nir"]    = args.use_nir
    if args.nir_mode and args.use_nir:
        raise SystemExit("--nir-mode replaces --use-nir; pass one of them")
    config["data"]["nir_mode"]   = args.nir_mode

    # Segmentation
    config.setdefault("segmentation", {})
    seg = config["segmentation"]
    seg["epochs"]        = args.epochs
    seg["batch_size"]    = args.batch_size
    seg["learning_rate"] = args.lr
    seg["weight_decay"]  = args.weight_decay
    seg["optimizer"]     = args.optimizer
    seg["loss"]          = args.loss
    seg["scheduler"]     = args.scheduler
    seg["patience"]      = args.patience

    # Output
    config.setdefault("output", {})
    config["output"]["models_dir"] = os.path.join(args.output_dir, "models")
    config["output"]["log_dir"]    = os.path.join(args.output_dir, "logs")

    return config


# ──────────────────────────────────────────────
# Class weights (inverse-sqrt frequency from train masks)
# ──────────────────────────────────────────────
def compute_class_weights(tiles_dir, labels_dir, num_classes,
                          file_list=None, class_remap=None):
    """นับ pixel ต่อ class จาก mask ของ train split แล้วให้ weight = 1/sqrt(freq)
    (normalize ให้ mean = 1, clip ที่ 10 กัน class ที่แทบไม่มี pixel ครอบงำ loss)

    `file_list` ใช้ส่งรายชื่อ tile ของ split train มาโดยตรง (spatial splits)
    `class_remap` ต้องตรงกับที่ dataset ใช้ ไม่งั้นพิกเซลของคลาสที่ถูกยุบจะหายไป
    จากการนับแทนที่จะไปสมทบคลาสปลายทาง
    """
    import cv2
    import json as _json

    if file_list is not None:
        files = file_list
    else:
        splits_path = os.path.join(tiles_dir, "splits.json")
        if os.path.exists(splits_path):
            with open(splits_path, "r", encoding="utf-8") as f:
                files = _json.load(f)["train"]
        else:
            files = sorted(f for f in os.listdir(labels_dir) if f.endswith(".png"))

    remap = {int(k): int(v) for k, v in (class_remap or {}).items()}
    n_src = max([num_classes] + [k + 1 for k in remap])
    counts = np.zeros(num_classes, dtype=np.int64)
    for fname in files:
        mask = cv2.imread(os.path.join(labels_dir, fname), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        c = np.bincount(mask.flatten(), minlength=n_src)[:n_src]
        for src, dst in remap.items():
            c[dst] += c[src]
            c[src] = 0
        counts += c[:num_classes]

    freq = counts / max(counts.sum(), 1)
    weights = 1.0 / np.sqrt(np.maximum(freq, 1e-6))
    # normalize ด้วย median (ทนต่อ class ที่หายไป/น้อยมากซึ่งทำให้ weight พุ่ง)
    weights = weights / np.median(weights)
    weights = np.clip(weights, 0.2, 5.0)
    return weights.astype(np.float32), freq


# ──────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device, scaler, epoch, epochs):
    model.train()
    total_loss = 0
    use_amp = scaler is not None and device.type == "cuda"

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            outputs = model(images)
            loss    = criterion(outputs, masks)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate_epoch(model, loader, criterion, device, metrics, epoch, epochs,
                   ignore_background=False):
    model.eval()
    total_loss = 0
    metrics.reset()

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs} [val]  ", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)
        outputs = model(images)
        loss    = criterion(outputs, masks)
        total_loss += loss.item()
        if ignore_background:
            # addendum D: background is never predicted and its reference pixels are
            # not scored (SegmentationMetrics drops targets < 0)
            outputs = outputs.clone()
            outputs[:, 0] = float("-inf")
            masks = torch.where(masks > 0, masks, torch.full_like(masks, -1))
        metrics.update(outputs, masks)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(len(loader), 1), metrics.get_summary()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main(args):
    config = build_config(args)
    os.makedirs(config["output"]["models_dir"], exist_ok=True)
    os.makedirs(config["output"]["log_dir"],    exist_ok=True)

    # ตั้ง seed ก่อนสร้างโมเดล เพราะน้ำหนักของ decoder head ถูกสุ่มตอน create_model
    if args.seed is not None:
        set_seed(args.seed)
        print(f"seed: {args.seed} (cudnn deterministic)")
    elif args.splits:
        raise SystemExit("--splits ต้องมาคู่กับ --seed: การเปรียบเทียบของ re-run "
                         "ตัดสินด้วย mean +/- SD ข้าม seed")

    device = get_device()
    use_amp = device.type == "cuda"
    scaler  = GradScaler() if use_amp else None

    # Model
    model = create_model(args, config)

    # Multi-GPU (DataParallel)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)

    # Resume
    start_epoch = 0
    best_miou   = 0.0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        state = ckpt["model_state_dict"]
        # strip DataParallel prefix if needed
        if list(state.keys())[0].startswith("module.") and not isinstance(model, nn.DataParallel):
            state = {k[7:]: v for k, v in state.items()}
        model.load_state_dict(state)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_miou   = ckpt.get("best_miou", 0.0)
        print(f"Resumed from {args.resume}  (epoch {start_epoch}, best mIoU {best_miou:.4f})")

    # Data
    print("Loading dataset...")
    train_files, train_remap = None, None
    if args.splits:
        loaders, num_classes = create_spatial_dataloaders(
            config, args.splits, batch_size=args.batch_size, augment=args.augment)
        train_loader, val_loader = loaders["train"], loaders["val"]
        config["num_classes"] = num_classes
        splits_map, train_remap, class_names_split = load_spatial_splits(args.splits)
        train_files = splits_map["train"]
        print(f"  spatial splits: " +
              "  ".join(f"{k}={len(v)}" for k, v in splits_map.items()))
        print(f"  augmentation: {'on' if args.augment else 'off'}  |  "
              f"classes: {num_classes}")
    else:
        train_loader, val_loader, _ = create_dataloaders(config)
    print(f"  train batches: {len(train_loader)}  |  val batches: {len(val_loader)}")

    # Optimizer / scheduler / loss
    seg = config["segmentation"]
    if args.splits:
        class_names = class_names_split
    else:
        class_names = [c["name"] for c in config["classes"]]

    class_weights = None
    if args.class_weights == "auto":
        print("Computing class weights from train masks...")
        class_weights, freq = compute_class_weights(
            args.tiles_dir, args.labels_dir, config["num_classes"],
            file_list=train_files, class_remap=train_remap)
        for n, f_, w in zip(class_names, freq, class_weights):
            print(f"  {n:<14} freq={f_*100:6.2f}%  weight={w:.3f}")
    elif args.class_weights not in (None, "", "none"):
        class_weights = np.array([float(x) for x in args.class_weights.split(",")], dtype=np.float32)
        assert len(class_weights) == config["num_classes"], \
            f"ต้องมี weight {config['num_classes']} ค่า ได้ {len(class_weights)}"

    if args.ignore_background:
        if class_weights is not None or seg["loss"] != "combo":
            raise SystemExit("--ignore-background is defined for combo loss without class weights")
        criterion = BackgroundMaskedComboLoss(config["num_classes"])
    else:
        criterion = get_loss_function(seg["loss"], config["num_classes"], class_weights)
    if isinstance(criterion, nn.Module):
        criterion = criterion.to(device)

    params = model.parameters()
    if args.freeze_backbone:
        bb = model.module.backbone if isinstance(model, nn.DataParallel) else model.backbone
        for q in bb.parameters():
            q.requires_grad_(False)
        params = [q for q in model.parameters() if q.requires_grad]
    elif args.backbone_lr_mult != 1.0:
        # addendum E: pretrained foundation backbones get a lower rate than the new head
        bb = model.module.backbone if isinstance(model, nn.DataParallel) else model.backbone
        bb_ids = {id(q) for q in bb.parameters()}
        params = [
            {"params": [q for q in model.parameters() if id(q) in bb_ids],
             "lr": seg["learning_rate"] * args.backbone_lr_mult},
            {"params": [q for q in model.parameters() if id(q) not in bb_ids]},
        ]
    if seg["optimizer"] == "adamw":
        optimizer = AdamW(params, lr=seg["learning_rate"], weight_decay=seg["weight_decay"])
    elif seg["optimizer"] == "adam":
        optimizer = Adam(params, lr=seg["learning_rate"], weight_decay=seg["weight_decay"])
    else:
        optimizer = SGD(params, lr=seg["learning_rate"], momentum=0.9, weight_decay=seg["weight_decay"])

    epochs = seg["epochs"]
    if seg["scheduler"] == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    elif seg["scheduler"] == "step":
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    elif seg["scheduler"] == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    else:
        scheduler = None

    metrics     = SegmentationMetrics(config["num_classes"], class_names)
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "miou": [], "oa": [], "lr": [],
               "per_class_iou": []}
    best_summary = None

    print(f"\n{'='*60}")
    print(f"  {args.model.upper()} + {args.encoder}")
    print(f"  Epochs={epochs}  Batch={seg['batch_size']}  LR={seg['learning_rate']}")
    print(f"  Optimizer={seg['optimizer']}  Loss={seg['loss']}  AMP={use_amp}")
    print(f"  Crop={args.crop_size}x{args.crop_size}  Classes={config['num_classes']}")
    print(f"{'='*60}\n")

    t0 = time.time()
    for epoch in range(start_epoch + 1, epochs + 1):
        train_loss             = train_epoch(model, train_loader, optimizer, criterion, device, scaler, epoch, epochs)
        val_loss, val_summary  = validate_epoch(model, val_loader, criterion, device, metrics, epoch, epochs,
                                                ignore_background=args.ignore_background)

        miou = val_summary["mean_iou"]
        if args.ignore_background:
            # select on the six real classes; background IoU is 0 by construction
            miou = float(np.mean([v for k, v in val_summary["per_class_iou"].items()
                                  if k != "background"]))
            val_summary["mean_iou_excl_background"] = miou
        oa   = val_summary["overall_accuracy"]
        lr   = optimizer.param_groups[0]["lr"]

        if scheduler:
            scheduler.step(val_loss) if isinstance(scheduler, ReduceLROnPlateau) else scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["miou"].append(miou)
        history["oa"].append(oa)
        history["lr"].append(lr)
        history["per_class_iou"].append(
            {k: round(float(v), 4) for k, v in val_summary["per_class_iou"].items()})

        elapsed = (time.time() - t0) / 60
        eta     = elapsed / epoch * (epochs - epoch)
        print(f"Ep {epoch:3d}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}  "
              f"mIoU={miou:.4f}  OA={oa:.4f}  lr={lr:.2e}  "
              f"elapsed={elapsed:.0f}m  ETA={eta:.0f}m")

        # Save last checkpoint
        last_path = os.path.join(config["output"]["models_dir"], f"{args.model}_last.pth")
        state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save({"epoch": epoch, "model_state_dict": state_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_miou": best_miou, "config": config}, last_path)

        if miou > best_miou:
            best_miou = miou
            best_summary = val_summary
            patience_counter = 0
            best_path = os.path.join(config["output"]["models_dir"], f"{args.model}_best.pth")
            torch.save({"epoch": epoch, "model_state_dict": state_dict,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_miou": best_miou, "config": config}, best_path)
            per_cls = "  ".join(f"{k}={v:.3f}" for k, v in val_summary["per_class_iou"].items())
            print(f"  --> New best mIoU: {best_miou:.4f}  (saved)")
            print(f"      per-class IoU: {per_cls}")
        else:
            patience_counter += 1
            if patience_counter >= seg["patience"]:
                print(f"Early stopping after {seg['patience']} epochs without improvement.")
                break

    # Save history
    hist_path = os.path.join(config["output"]["log_dir"], f"{args.model}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    # Save metrics of the best epoch (incl. per-class IoU/F1, kappa)
    if best_summary is not None:
        best_metrics_path = os.path.join(config["output"]["log_dir"], f"{args.model}_best_metrics.json")
        if class_weights is not None:
            best_summary["class_weights"] = [round(float(w), 4) for w in class_weights]
        # ติดเงื่อนไขของ run ไว้กับผลเสมอ — 280 runs ต่างกันแค่ธงไม่กี่ตัว และไฟล์ผล
        # ที่ไม่บอกที่มาเคยทำให้หยิบเลขผิดชุดมาแล้ว
        best_summary["run"] = {
            "model": args.model, "encoder": args.encoder,
            "seed": args.seed, "augment": bool(args.augment),
            "class_weights_mode": args.class_weights,
            "use_nir": bool(args.use_nir), "splits": args.splits,
            "ignore_background": bool(args.ignore_background),
            "backbone_lr_mult": args.backbone_lr_mult,
            "nir_mode": args.nir_mode,
            "freeze_backbone": bool(args.freeze_backbone),
            "num_classes": config["num_classes"],
            "n_train_tiles": len(train_loader.dataset),
            "n_val_tiles": len(val_loader.dataset),
        }
        # ต้นทุนการคำนวณต่อ run — รายงานรวมเป็นชั่วโมง GPU ในบทความ และตอบคำถาม
        # เรื่องความคุ้มค่าของ ensemble ที่ต้อง forward หลายรอบต่อพิกเซล
        best_summary["compute"] = {
            "train_minutes": round((time.time() - t0) / 60, 2),
            "epochs_run": len(history.get("miou", [])),
            "params_millions": round(
                sum(p.numel() for p in model.parameters()) / 1e6, 2),
            "gpu": (torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() else str(device)),
            "peak_gpu_mem_gb": (round(torch.cuda.max_memory_allocated() / 1e9, 2)
                                if torch.cuda.is_available() else None),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
        with open(best_metrics_path, "w") as f:
            json.dump(best_summary, f, indent=2, default=float)

    total_min = (time.time() - t0) / 60
    print(f"\nDone in {total_min:.0f} min  |  Best mIoU: {best_miou:.4f}")
    print(f"Model saved to: {config['output']['models_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="THEOS Segmentation — GPU Training")

    # Model
    parser.add_argument("--model",    default="unet",
                        choices=["unet", "unetpp", "deeplabv3", "hrnet",
                                 "segformer", "upernet", "satlas", "dinov3"])
    parser.add_argument("--encoder",  default="resnet50",
                        help="Encoder backbone (resnet34/50/101, efficientnet-b3, mobilenet_v2, ...)")
    parser.add_argument("--pretrained", action="store_true",
                        help="Load ImageNet pretrained weights")

    # Data
    parser.add_argument("--tiles-dir",  default="data/tiles")
    parser.add_argument("--labels-dir", default="data/labels/semantic")
    parser.add_argument("--output-dir", default="train_output")
    parser.add_argument("--config",     default="configs/config.yaml",
                        help="Path to config.yaml (for class definitions)")
    parser.add_argument("--use-nir",    action="store_true")
    parser.add_argument("--splits",     default=None,
                        help="พาธ spatial_splits.json (paper1 re-run); "
                             "ถ้าไม่ระบุจะใช้ splits.json เดิมใน --tiles-dir")
    parser.add_argument("--seed",       type=int, default=None,
                        help="seed สำหรับ RNG ทุกตัว — ต้องระบุในทุก run ของ re-run")
    aug = parser.add_mutually_exclusive_group()
    aug.add_argument("--augment",    dest="augment", action="store_true",
                     help="เปิด online augmentation (flip/rotate/brightness/noise)")
    aug.add_argument("--no-augment", dest="augment", action="store_false",
                     help="ปิด augmentation จริง ๆ — แขนควบคุมของ factorial")
    parser.set_defaults(augment=True)
    parser.add_argument("--crop-size",  type=int, default=512,
                        help="Training crop size (use 256 to save GPU memory)")

    # Training
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch-size",   type=int,   default=16)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer",    default="adamw",
                        choices=["adamw", "adam", "sgd"])
    parser.add_argument("--loss",         default="combo",
                        choices=["combo", "ce", "dice", "focal"])
    parser.add_argument("--class-weights", default="none",
                        help="'auto' = คำนวณจาก train masks (inverse-sqrt freq), "
                             "'none', หรือค่าคั่น comma เช่น '1.0,2.5,...'")
    parser.add_argument("--scheduler",    default="cosine",
                        choices=["cosine", "step", "plateau", "none"])
    parser.add_argument("--patience",     type=int, default=20)
    parser.add_argument("--nir-mode", default=None, choices=["nrg", "lowpass"],
                        help="addendum G: NIR,R,G in a 3-channel stem, or RGB + NIR low-passed to 2 m")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="addendum H1: train the head only")
    parser.add_argument("--backbone-lr-mult", type=float, default=1.0,
                        help="learning-rate multiplier for model.backbone (addendum E)")
    parser.add_argument("--ignore-background", action="store_true",
                        help="paper1 addendum D: exclude background from loss and selection")
    parser.add_argument("--resume",       default=None,
                        help="Path to checkpoint to resume from")

    args = parser.parse_args()
    main(args)
