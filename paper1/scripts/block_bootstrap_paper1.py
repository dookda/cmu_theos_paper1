#!/usr/bin/env python3
"""Paired spatial-block bootstrap between two arms of the paper1 re-run.

Consumes the per-block metrics written by eval_rerun.py. The tile is not a valid
resampling unit here — tiles inside a block share landscape and were cut from the
same neighbourhood, so resampling tiles yields CIs that are far too narrow. The
1.5 km block is the unit, matching paper2's canonical analysis (10 test blocks).

For a pair (A, B):
  1. average each block's metric over the seeds of that arm
  2. keep blocks that pass inclusion in BOTH arms
  3. paired bootstrap over blocks (10,000 resamples) -> mean diff + 95% percentile CI
  4. Wilcoxon signed-rank on per-block differences
  5. Cohen's d_z = mean(diff) / sd(diff)
  6. count blocks where A beats / ties / loses to B

With 10 blocks the Wilcoxon floor is p = 0.002, so the CI stays the primary
decision rule and the block win count the primary description.

Usage
-----
  python paper1/scripts/block_bootstrap_paper1.py \
      --eval paper1/analysis/results/rerun_eval_test.json \
      --a "*__b4_rgbnir_plain__*" --b "*__b0_rgb_plain__*"
"""
import argparse
import fnmatch
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

N_BOOT = 10000
MIN_TILES = 5          # บล็อกเล็กกว่านี้ให้ค่า metric ที่ผันผวนเกินกว่าจะใช้เป็นหน่วย


def arm_blocks(rows, pattern, metric, min_tiles):
    """{block: ค่าเฉลี่ยข้าม seed} ของ run ที่ชื่อตรง pattern

    บล็อกที่เล็กกว่า min_tiles ถูกตัดทิ้ง — external มีบล็อกที่เหลือ 1-4 tile ซึ่ง
    ให้ค่า metric ที่ผันผวนจนกลายเป็นหน่วยที่ให้น้ำหนักเท่าบล็อก 49 tile ไม่ได้
    """
    picked = [r for r in rows if fnmatch.fnmatch(r["run"], pattern)]
    if not picked:
        raise SystemExit(f"ไม่มี run ที่ตรงกับ {pattern!r}")
    per_block, sizes = defaultdict(list), {}
    for r in picked:
        for blk, summary in r.get("per_block", {}).items():
            if metric not in summary:
                continue
            n = summary.get("n_tiles")
            if n is not None:
                sizes[blk] = n
                if n < min_tiles:
                    continue
            per_block[blk].append(summary[metric])
    excluded = sorted(b for b, n in sizes.items() if n < min_tiles)
    return {b: float(np.mean(v)) for b, v in per_block.items()}, picked, excluded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, help="rerun_eval_<split>.json")
    ap.add_argument("--a", required=True, help="glob ของ run ฝั่ง A")
    ap.add_argument("--b", required=True, help="glob ของ run ฝั่ง B")
    ap.add_argument("--metric", default="mean_iou")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-tiles", type=int, default=MIN_TILES,
                    help="ตัดบล็อกที่มี tile น้อยกว่านี้ออกจากหน่วย bootstrap")
    args = ap.parse_args()

    payload = json.loads(Path(args.eval).read_text())
    rows = payload["results"]
    a_blocks, a_runs, excl = arm_blocks(rows, args.a, args.metric, args.min_tiles)
    b_blocks, b_runs, _ = arm_blocks(rows, args.b, args.metric, args.min_tiles)

    blocks = sorted(set(a_blocks) & set(b_blocks))
    if not blocks:
        raise SystemExit("ไม่มีบล็อกที่ทั้งสองฝั่งมีร่วมกัน")

    a = np.array([a_blocks[b] for b in blocks])
    b = np.array([b_blocks[b] for b in blocks])
    diff = a - b

    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, len(blocks), size=(args.boot, len(blocks)))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    try:
        from scipy.stats import wilcoxon
        w_p = float(wilcoxon(diff).pvalue) if not np.allclose(diff, 0) else float("nan")
    except ImportError:
        w_p = float("nan")
    d_z = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else float("nan")
    wins = int((diff > 0).sum())
    ties = int((diff == 0).sum())
    losses = int((diff < 0).sum())

    print(f"split      : {payload['split']}   metric: {args.metric}")
    print(f"arm A      : {args.a}   ({len(a_runs)} runs)")
    print(f"arm B      : {args.b}   ({len(b_runs)} runs)")
    print(f"blocks     : {len(blocks)}"
          + (f"  (ตัดบล็อกเล็กกว่า {args.min_tiles} tile ออก {len(excl)}: "
             f"{', '.join(excl)})" if excl else ""))
    print(f"A mean     : {a.mean():.4f}")
    print(f"B mean     : {b.mean():.4f}")
    print(f"A - B      : {diff.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
          f"{'  *' if lo > 0 or hi < 0 else ''}")
    print(f"Wilcoxon p : {w_p:.4f}")
    print(f"Cohen d_z  : {d_z:+.3f}")
    print(f"blocks A>B / = / A<B : {wins} / {ties} / {losses}")
    print("\n*  = 95% CI ไม่คร่อมศูนย์")

    out = {
        "split": payload["split"], "metric": args.metric,
        "arm_a": args.a, "arm_b": args.b,
        "n_runs_a": len(a_runs), "n_runs_b": len(b_runs),
        "blocks": blocks, "n_blocks": len(blocks),
        "min_tiles": args.min_tiles, "excluded_small_blocks": excl,
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "mean_diff": float(diff.mean()), "ci95": [float(lo), float(hi)],
        "wilcoxon_p": w_p, "cohens_dz": d_z,
        "block_wins": wins, "block_ties": ties, "block_losses": losses,
        "per_block_diff": {b_: float(d) for b_, d in zip(blocks, diff)},
        "n_boot": args.boot,
    }
    dest = Path(args.eval).parent / (
        f"block_bootstrap_{payload['split']}_{args.metric}.json")
    prev = json.loads(dest.read_text()) if dest.exists() else []
    if not isinstance(prev, list):
        prev = [prev]
    prev = [p for p in prev if not (p.get("arm_a") == args.a and p.get("arm_b") == args.b)]
    prev.append(out)
    dest.write_text(json.dumps(prev, indent=1))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
