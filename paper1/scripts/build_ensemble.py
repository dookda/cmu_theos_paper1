#!/usr/bin/env python3
"""Select a softmax-averaging ensemble for the paper1 re-run using val only.

The first round evaluated five hand-made ensembles on the test set and reported
the best of them (ens_mixed9, test mIoU 57.06%). That makes the reported number a
maximum over five test evaluations, not an estimate of held-out performance, and
it is the main reason the test set stopped being a holdout.

Here selection is greedy forward selection driven entirely by validation mIoU:
members are added one at a time, each round picking whichever candidate most
improves val, and stopping when no candidate improves it. The chosen ensemble is
then evaluated once on test (and later, once, on external) by eval_rerun.py. The
size of the ensemble is an outcome of the procedure, not a knob.

Selection with replacement is allowed: a member may be added twice, which is how
softmax averaging expresses an unequal weight without introducing a weight
parameter to tune.

Because averaging is over probabilities, argmax(sum) == argmax(mean), so the
running sum is kept unnormalised.

Stage 1 caches each candidate's val softmax to disk (float16, ~1.1 GB per member
for 304 tiles at 512x512x7); stage 2 is then pure arithmetic over those caches.

Usage
-----
  python paper1/scripts/build_ensemble.py --results rerun_output --cache ens_cache
  python paper1/scripts/build_ensemble.py --results rerun_output --cache ens_cache \
      --max-members 12 --skip-cache
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.dataset import create_spatial_dataloaders  # noqa: E402

RGB_ROOT = "/project/gisnorth/cmu_theos_ai_data_fixed"
NIR_ROOT = "/project/gisnorth/cmu_theos_ai_data_nir_only"


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
    raise ValueError(f"unknown model {model_name}")


def miou_from_confusion(cm):
    inter = np.diag(cm).astype(np.float64)
    union = cm.sum(0) + cm.sum(1) - np.diag(cm)
    valid = union > 0
    return float((inter[valid] / union[valid]).mean()) if valid.any() else 0.0


@torch.no_grad()
def cache_member(run_dir, splits_file, cache_dir, device, batch_size):
    """เขียน softmax ของ val ลง memmap float16 คืน (path, labels_path)"""
    meta = json.loads(next(run_dir.glob("logs/*_best_metrics.json")).read_text())["run"]
    ckpt = run_dir / "models" / f"{meta['model']}_best.pth"
    if not ckpt.exists():
        return None

    use_nir = bool(meta.get("use_nir"))
    data_root = NIR_ROOT if use_nir else RGB_ROOT
    config = {
        "data": {"tiles_dir": f"{data_root}/tiles",
                 "labels_dir": f"{data_root}/semantic_labels",
                 "tile_size": 512, "use_nir": use_nir},
        "segmentation": {"batch_size": batch_size},
        "num_classes": meta.get("num_classes", 7),
    }
    loaders, num_classes = create_spatial_dataloaders(
        config, splits_file, batch_size=batch_size, augment=False)
    loader = loaders["val"]
    n_tiles = len(loader.dataset)

    out_path = cache_dir / f"{run_dir.name}.npy"
    lbl_path = cache_dir / "_labels.npy"
    if out_path.exists():
        return out_path, lbl_path

    model = build_model(meta["model"], meta["encoder"],
                        4 if use_nir else 3, num_classes).to(device).eval()
    state = torch.load(ckpt, map_location=device)["model_state_dict"]
    if next(iter(state)).startswith("module."):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state)

    probs = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.float16,
        shape=(n_tiles, num_classes, 512, 512))
    need_labels = not lbl_path.exists()
    labels = (np.lib.format.open_memmap(
        lbl_path, mode="w+", dtype=np.uint8, shape=(n_tiles, 512, 512))
        if need_labels else None)

    i = 0
    for images, masks in loader:
        p = torch.softmax(model(images.to(device)), dim=1).cpu().numpy()
        probs[i:i + len(p)] = p.astype(np.float16)
        if labels is not None:
            labels[i:i + len(p)] = masks.numpy().astype(np.uint8)
        i += len(p)
    probs.flush()
    if labels is not None:
        labels.flush()
    del model
    torch.cuda.empty_cache()
    return out_path, lbl_path


def confusion_for(running, candidate, labels, num_classes, chunk=16):
    """confusion matrix ของ argmax(running + candidate) เทียบ labels"""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    n = labels.shape[0]
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        tot = running[s:e].astype(np.float32)
        if candidate is not None:
            tot = tot + candidate[s:e].astype(np.float32)
        pred = tot.argmax(1).ravel()
        gt = labels[s:e].ravel().astype(np.int64)
        m = gt < num_classes
        cm += np.bincount(gt[m] * num_classes + pred[m],
                          minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--cache", required=True, help="โฟลเดอร์เก็บ softmax ของ val")
    ap.add_argument("--splits-file",
                    default="/project/gisnorth/paper1/analysis/results/spatial_splits.json")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-members", type=int, default=12)
    ap.add_argument("--min-gain", type=float, default=1e-4,
                    help="กำไร val mIoU ขั้นต่ำที่ยังถือว่าคุ้มจะเพิ่มสมาชิก")
    ap.add_argument("--skip-cache", action="store_true",
                    help="ใช้ cache ที่มีอยู่ ไม่รัน inference ใหม่")
    ap.add_argument("--pool", default="*__s0",
                    help="glob จำกัดผู้สมัคร ค่าเริ่มต้น seed 0 หนึ่งตัวต่อเซลล์ของ "
                         "factorial — seed อื่นของ config เดียวกันแทบไม่เพิ่มความ "
                         "หลากหลายของ error แต่เพิ่ม cache 1.1 GB และรอบ greedy ต่อตัว")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = Path(args.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)
    import fnmatch
    runs = sorted(p.parent.parent
                  for p in Path(args.results).glob("*/logs/*_best_metrics.json")
                  if fnmatch.fnmatch(p.parent.parent.name, args.pool))
    if not runs:
        raise SystemExit(f"ไม่พบ run ใน {args.results}")

    print(f"ผู้สมัคร {len(runs)} runs | device {device}")
    members = {}
    for r in runs:
        if args.skip_cache:
            p = cache_dir / f"{r.name}.npy"
            if p.exists():
                members[r.name] = p
            continue
        res = cache_member(r, args.splits_file, cache_dir, device, args.batch_size)
        if res:
            members[r.name] = res[0]
            print(f"  cached {r.name}")
    if not members:
        raise SystemExit("ไม่มีสมาชิกที่ cache ได้")

    labels = np.load(cache_dir / "_labels.npy", mmap_mode="r")
    num_classes = len(json.loads(Path(args.splits_file).read_text())["classes"])

    # ค่า val ของสมาชิกเดี่ยว ใช้เป็นจุดอ้างอิงว่า ensemble ได้กำไรเท่าไร
    solo = {}
    for name, path in members.items():
        arr = np.load(path, mmap_mode="r")
        solo[name] = miou_from_confusion(confusion_for(arr, None, labels, num_classes))
    best_solo = max(solo, key=solo.get)
    print(f"\nสมาชิกเดี่ยวที่ดีที่สุดบน val: {best_solo}  mIoU {solo[best_solo]:.4f}")

    selected, trace = [], []
    running = np.zeros(labels.shape[:1] + (num_classes, 512, 512), dtype=np.float32)
    current = 0.0
    for step in range(args.max_members):
        best_name, best_score = None, current
        for name, path in members.items():
            arr = np.load(path, mmap_mode="r")
            score = miou_from_confusion(
                confusion_for(running, arr, labels, num_classes))
            if score > best_score + (args.min_gain if step else 0):
                best_name, best_score = name, score
        if best_name is None:
            print(f"หยุดที่ {len(selected)} สมาชิก — ไม่มีตัวไหนเพิ่ม val mIoU ได้อีก")
            break
        running += np.load(members[best_name], mmap_mode="r").astype(np.float32)
        selected.append(best_name)
        trace.append({"step": step + 1, "added": best_name,
                      "val_miou": best_score, "gain": best_score - current})
        print(f"  +{best_name:<45} val mIoU {best_score:.4f} "
              f"({best_score - current:+.4f})")
        current = best_score

    out = {
        "selected": selected, "n_members": len(selected),
        "val_miou": current,
        "best_single_val": {"run": best_solo, "val_miou": solo[best_solo]},
        "gain_over_best_single": current - solo[best_solo],
        "solo_val_miou": solo, "trace": trace,
        "selection_split": "val",
        "note": "test/external ยังไม่ถูกแตะในขั้นตอนนี้",
    }
    dest = Path(args.results).parent / "analysis" / "results" / "ensemble_selection.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print(f"\nเลือกได้ {len(selected)} สมาชิก | val mIoU {current:.4f} "
          f"(+{current - solo[best_solo]:.4f} จากตัวเดี่ยวที่ดีที่สุด)")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
