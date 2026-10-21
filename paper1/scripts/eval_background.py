#!/usr/bin/env python3
"""Addendum D: score checkpoints with background removed as a prediction target.

See `paper1/configs/addendum_prereg_2026-10.md` section D. For every run under the
given result roots, write the full 7x7 confusion matrix (rows = reference, columns =
prediction) for each requested mode:

  std  ordinary 7-class argmax
  sup  argmax over the six real classes (background logit suppressed)

Runs trained with --ignore-background are scored in `sup` only, since they never learnt
a background logit. The metrics of the pre-registration (mIoU6 on non-background
reference pixels, predicted-background share) are computed from these matrices by
`analyze_background.py`, so this script makes no metric choice of its own.

Usage
-----
  python scripts/eval_background.py --split val \
      --runs rerun_output/unetpp_effb3__b1_rgb_aug__s0 ... --results-root rerun_output_ignbg \
      --out analysis/results/background_eval_val.json
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_rerun import build_model                    # noqa: E402
from utils.dataset import create_spatial_dataloaders  # noqa: E402


@torch.no_grad()
def confusion(model, loader, device, n, modes):
    cm = {m: np.zeros((n, n), dtype=np.int64) for m in modes}
    model.eval()
    for batch in loader:
        images, masks = batch[0].to(device, non_blocking=True), batch[1]
        logits = model(images)
        t = masks.numpy().ravel()
        ok = (t >= 0) & (t < n)
        for m in modes:
            lg = logits.clone()
            if m == "sup":
                lg[:, 0] = float("-inf")
            p = lg.argmax(1).cpu().numpy().ravel()
            cm[m] += np.bincount(n * t[ok] + p[ok], minlength=n * n).reshape(n, n)
    return cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["val", "test", "external"])
    ap.add_argument("--runs", nargs="*", default=[], help="run directories (std + sup)")
    ap.add_argument("--results-root", nargs="*", default=[],
                    help="roots whose runs are all scored (ign runs -> sup only)")
    ap.add_argument("--splits-file",
                    default="/project/gisnorth/paper1/analysis/results/spatial_splits.json")
    ap.add_argument("--rgb-root", default="/project/gisnorth/cmu_theos_ai_data_fixed")
    ap.add_argument("--nir-root", default="/project/gisnorth/cmu_theos_ai_data_nir_only")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--unlock", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.split == "external" and not args.unlock:
        raise SystemExit("external is locked: pass --unlock \"<reason>\"")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dirs = [Path(r) for r in args.runs]
    for root in args.results_root:
        run_dirs += sorted(p.parent.parent for p in Path(root).glob("*/logs/*_best_metrics.json"))
    if not run_dirs:
        raise SystemExit("no runs given")
    class_names = json.loads(Path(args.splits_file).read_text())["classes"]

    out = []
    for run_dir in run_dirs:
        meta = json.loads(next(run_dir.glob("logs/*_best_metrics.json")).read_text())["run"]
        ckpt = run_dir / "models" / f"{meta['model']}_best.pth"
        use_nir = bool(meta.get("use_nir"))
        root = args.nir_root if use_nir else args.rgb_root
        config = {"data": {"tiles_dir": f"{root}/tiles", "labels_dir": f"{root}/semantic_labels",
                           "tile_size": 512, "use_nir": use_nir},
                  "segmentation": {"batch_size": args.batch_size},
                  "num_classes": meta.get("num_classes", 7)}
        loaders, n = create_spatial_dataloaders(config, args.splits_file,
                                                batch_size=args.batch_size, augment=False)
        model = build_model(meta["model"], meta["encoder"], 4 if use_nir else 3, n).to(device)
        state = torch.load(ckpt, map_location=device)["model_state_dict"]
        if next(iter(state)).startswith("module."):
            state = {k[7:]: v for k, v in state.items()}
        model.load_state_dict(state)

        ign = bool(meta.get("ignore_background"))
        modes = ["sup"] if ign else ["std", "sup"]
        cm = confusion(model, loaders[args.split], device, n, modes)
        for m in modes:
            arm = "ign" if ign else m
            out.append({"run": run_dir.name, "arm": arm, "split": args.split,
                        "model": meta["model"], "encoder": meta["encoder"],
                        "seed": meta["seed"], "class_names": class_names,
                        "confusion_matrix": cm[m].tolist()})
            print(f"  {run_dir.name:<45} {arm}")

    Path(args.out).write_text(json.dumps({
        "split": args.split,
        "evaluated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "unlock_reason": args.unlock, "splits_file": args.splits_file,
        "prereg": "paper1/configs/addendum_prereg_2026-10.md#D",
        "results": out}, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
