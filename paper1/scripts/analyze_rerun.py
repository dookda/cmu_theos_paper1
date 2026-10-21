#!/usr/bin/env python3
"""Aggregate the paper1 re-run across seeds and estimate the factorial effects.

The first round reported single runs with no seed control, so differences of
+0.11 to +1.61 points were indistinguishable from training noise. Here every cell
of the 2x2x2 grid has 5 seeds, and each effect is estimated once per seed, giving
a paired sample whose spread says whether the effect is real.

Factorial estimation
--------------------
For one architecture the eight conditions form a 2^3 design in
(bands, augment, weights). Within a seed, a main effect is the mean of the four
cells at the high level minus the mean of the four at the low level; a two-way
interaction uses the product of the two factors' +-1 codes. Because seeds are
matched across cells, this yields 5 independent estimates per effect, summarised
as mean +- SD with a 95% CI and a paired test against zero.

Reporting the effect this way — rather than comparing two single runs — is what
lets the paper state that NIR helps (or does not) with a quantified uncertainty.

Usage
-----
  python paper1/scripts/analyze_rerun.py --results rerun_output
  python paper1/scripts/analyze_rerun.py --results rerun_output --metric mean_iou
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "results"

# +-1 coding of the three factors for each condition name
FACTORS = ["bands", "augment", "weights"]
CODES = {
    "b0_rgb_plain":     (-1, -1, -1),
    "b1_rgb_aug":       (-1, +1, -1),
    "b2_rgb_wt":        (-1, -1, +1),
    "b3_rgb_aug_wt":    (-1, +1, +1),
    "b4_rgbnir_plain":  (+1, -1, -1),
    "b5_rgbnir_aug":    (+1, +1, -1),
    "b6_rgbnir_wt":     (+1, -1, +1),
    "b7_rgbnir_aug_wt": (+1, +1, +1),
}
RUN_RE = re.compile(r"^(?P<arch>.+?)__(?P<cond>.+?)__s(?P<seed>\d+)$")


def load(results_dir, metric):
    """{arch: {condition: {seed: value}}} จากผลที่รันเสร็จแล้ว"""
    data = defaultdict(lambda: defaultdict(dict))
    n_files = 0
    for path in sorted(Path(results_dir).glob("*/logs/*_best_metrics.json")):
        run = path.parent.parent.name
        m = RUN_RE.match(run)
        if not m:
            continue
        payload = json.loads(path.read_text())
        if metric not in payload:
            continue
        data[m["arch"]][m["cond"]][int(m["seed"])] = payload[metric]
        n_files += 1
    return data, n_files


def ci95(x):
    """95% CI ของค่าเฉลี่ยจาก t-distribution; คืน (lo, hi) และ NaN ถ้า n < 2"""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return float("nan"), float("nan")
    # scipy is required: a normal 1.96 in place of t(0.975, 4) = 2.776 made the intervals of the
    # first build ~30% too narrow (found 2026-10-21 when rerunning from the per-run logs)
    from scipy import stats
    t = stats.t.ppf(0.975, n - 1)
    half = t * x.std(ddof=1) / np.sqrt(n)
    return float(x.mean() - half), float(x.mean() + half)


def paired_p(x):
    """p-value ของ H0: mean = 0 — Wilcoxon และ t-test (ต้องมี scipy)

    n = 5 seeds ทำให้ Wilcoxon มี p ต่ำสุดที่ 0.0625 จึงรายงาน t-test คู่กัน
    และให้ CI เป็นตัวตัดสินหลักตามที่ paper2 ใช้
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 3 or np.allclose(x, 0):
        return float("nan"), float("nan")
    from scipy import stats
    w = stats.wilcoxon(x).pvalue
    t = stats.ttest_1samp(x, 0.0).pvalue
    return float(w), float(t)


def effects(cells):
    """cells = {condition: {seed: value}} -> {effect: [ต่อ seed]}"""
    seeds = sorted(set.intersection(*(set(v) for v in cells.values()))) if cells else []
    if len(cells) != len(CODES) or not seeds:
        return {}, seeds
    named = [
        ("bands (NIR)", lambda c: c[0]),
        ("augment", lambda c: c[1]),
        ("weights", lambda c: c[2]),
        ("bands x augment", lambda c: c[0] * c[1]),
        ("bands x weights", lambda c: c[0] * c[2]),
        ("augment x weights", lambda c: c[1] * c[2]),
    ]
    out = {}
    for label, sign in named:
        per_seed = []
        for s in seeds:
            hi = [cells[k][s] for k, c in CODES.items() if sign(c) > 0]
            lo = [cells[k][s] for k, c in CODES.items() if sign(c) < 0]
            per_seed.append(np.mean(hi) - np.mean(lo))
        out[label] = per_seed
    return out, seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="โฟลเดอร์ผล (rerun_output)")
    ap.add_argument("--metric", default="mean_iou",
                    choices=["mean_iou", "overall_accuracy", "kappa"])
    args = ap.parse_args()

    data, n_files = load(args.results, args.metric)
    if not n_files:
        raise SystemExit(f"ไม่พบผลใน {args.results}")
    print(f"อ่านผล {n_files} runs | metric = {args.metric}\n")

    rows = []
    print(f"{'architecture':<20} {'condition':<18} {'n':>2} {'mean':>8} {'SD':>7}")
    for arch in sorted(data):
        for cond in sorted(data[arch], key=lambda c: list(CODES).index(c)
                           if c in CODES else 99):
            vals = np.array(list(data[arch][cond].values()), dtype=float)
            lo, hi = ci95(vals)
            rows.append({"architecture": arch, "condition": cond, "n": len(vals),
                         "mean": vals.mean(), "sd": vals.std(ddof=1) if len(vals) > 1
                         else float("nan"), "ci_lo": lo, "ci_hi": hi})
            print(f"{arch:<20} {cond:<18} {len(vals):>2} {vals.mean():>8.4f} "
                  f"{rows[-1]['sd']:>7.4f}")

    print(f"\n{'architecture':<20} {'effect':<20} {'n':>2} {'mean':>8} {'SD':>7} "
          f"{'95% CI':>18} {'wilcox':>7} {'t':>7}")
    eff_rows = []
    for arch in sorted(data):
        eff, seeds = effects(data[arch])
        if not eff:
            print(f"{arch:<20} (ยังไม่ครบ 8 เงื่อนไข — ข้าม)")
            continue
        for label, per_seed in eff.items():
            v = np.array(per_seed)
            lo, hi = ci95(v)
            wp, tp = paired_p(v)
            eff_rows.append({"architecture": arch, "effect": label, "n": len(v),
                             "mean": v.mean(),
                             "sd": v.std(ddof=1) if len(v) > 1 else float("nan"),
                             "ci_lo": lo, "ci_hi": hi, "wilcoxon_p": wp, "ttest_p": tp})
            excl = "  *" if not (lo <= 0 <= hi) else ""
            print(f"{arch:<20} {label:<20} {len(v):>2} {v.mean():>+8.4f} "
                  f"{eff_rows[-1]['sd']:>7.4f} [{lo:>+7.4f},{hi:>+7.4f}] "
                  f"{wp:>7.4f} {tp:>7.4f}{excl}")
    print("\n*  = 95% CI ไม่คร่อมศูนย์")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"rerun_cells_{args.metric}.json").write_text(json.dumps(rows, indent=1))
    (OUT / f"rerun_effects_{args.metric}.json").write_text(json.dumps(eff_rows, indent=1))
    print(f"\nwrote {OUT / f'rerun_cells_{args.metric}.json'}")
    print(f"wrote {OUT / f'rerun_effects_{args.metric}.json'}")


if __name__ == "__main__":
    main()
