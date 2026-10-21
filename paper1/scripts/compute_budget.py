#!/usr/bin/env python3
"""Total the computational cost of the paper1 re-run for the manuscript.

Reviewers of the first version asked for the cost of the ensemble (nine forward
passes per pixel) to be quantified rather than asserted, and a study that trains
280 models should state what that took. Two sources are combined:

  sacct              wall-clock and GPU allocation per SLURM job — authoritative,
                     and covers every job including ones whose python process
                     predates the in-run instrumentation
  best_metrics.json  per-run epochs, parameters, peak GPU memory, in-process
                     minutes — richer, but only for runs trained after the
                     `compute` block was added

Where both exist they are reported side by side; sacct is the number quoted.

Usage
-----
  # on Erawan (has sacct)
  python paper1/scripts/compute_budget.py --results rerun_output --job-name p1-rerun
  # locally, from pulled results only
  python paper1/scripts/compute_budget.py --results rerun_output --no-sacct
"""
import argparse
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

ELAPSED_RE = re.compile(
    r"^(?:(?P<d>\d+)-)?(?:(?P<h>\d+):)?(?P<m>\d+):(?P<s>\d+(?:\.\d+)?)$")


def parse_elapsed(text):
    """'1-02:03:04' / '02:03:04' / '03:04' -> ชั่วโมง"""
    m = ELAPSED_RE.match(text.strip())
    if not m:
        return None
    d = int(m["d"] or 0)
    h = int(m["h"] or 0)
    return d * 24 + h + int(m["m"]) / 60 + float(m["s"]) / 3600


def from_sacct(job_name, since):
    """{job_id: hours} ของ job ที่จบสมบูรณ์"""
    cmd = ["sacct", "--name", job_name, "--starttime", since,
           "--format", "JobID,JobName,State,Elapsed,AllocTRES", "--parsable2", "--noheader"]
    try:
        raw = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"  (sacct ใช้ไม่ได้: {e.__class__.__name__}) — ข้ามไป")
        return {}
    jobs = {}
    for line in raw.strip().splitlines():
        parts = line.split("|")
        if len(parts) < 5:
            continue
        jid, name, state, elapsed, tres = parts[:5]
        if "." in jid:                 # ข้าม job step (batch/extern)
            continue
        hours = parse_elapsed(elapsed)
        if hours is None:
            continue
        n_gpu = 0
        mg = re.search(r"gres/gpu=(\d+)", tres)
        if mg:
            n_gpu = int(mg.group(1))
        jobs[jid] = {"state": state, "hours": hours, "gpus": max(n_gpu, 1)}
    return jobs


def from_results(results_dir):
    rows = []
    for path in sorted(Path(results_dir).glob("*/logs/*_best_metrics.json")):
        payload = json.loads(path.read_text())
        c = payload.get("compute")
        if not c:
            continue
        rows.append({"run": path.parent.parent.name, **c,
                     "use_nir": payload.get("run", {}).get("use_nir"),
                     "augment": payload.get("run", {}).get("augment")})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--job-name", default="p1-rerun")
    ap.add_argument("--since", default="2026-08-12")
    ap.add_argument("--no-sacct", action="store_true")
    args = ap.parse_args()

    print("== sacct ==")
    jobs = {} if args.no_sacct else from_sacct(args.job_name, args.since)
    done = {k: v for k, v in jobs.items() if v["state"].startswith("COMPLETED")}
    gpu_hours = sum(v["hours"] * v["gpus"] for v in done.values())
    all_hours = sum(v["hours"] * v["gpus"] for v in jobs.values())
    print(f"  jobs ทั้งหมด {len(jobs)} | สำเร็จ {len(done)}")
    print(f"  GPU-hours (เฉพาะที่สำเร็จ) {gpu_hours:.1f}")
    print(f"  GPU-hours (รวมที่ล้ม/ถูกยกเลิก) {all_hours:.1f}")

    print("\n== best_metrics.json ==")
    rows = from_results(args.results)
    print(f"  run ที่มีบล็อก compute: {len(rows)}")
    summary = {}
    if rows:
        mins = np.array([r["train_minutes"] for r in rows], dtype=float)
        eps = np.array([r.get("epochs_run") or 0 for r in rows], dtype=float)
        print(f"  นาที/run   mean {mins.mean():.1f}  median {np.median(mins):.1f}  "
              f"min {mins.min():.1f}  max {mins.max():.1f}")
        print(f"  epochs/run mean {eps.mean():.1f}  median {np.median(eps):.1f}")
        per_arm = defaultdict(list)
        for r in rows:
            per_arm[("NIR" if r.get("use_nir") else "RGB",
                     "aug" if r.get("augment") else "no-aug")].append(r["train_minutes"])
        by_arm = {}
        for k in sorted(per_arm):
            v = np.array(per_arm[k])
            print(f"    {k[0]:>3} / {k[1]:<6} n={len(v):<3} mean {v.mean():.1f} min")
            # เก็บลงไฟล์ด้วย ไม่ใช่แค่พิมพ์ — เนื้อความในเปเปอร์ต้องอ่านจากไฟล์ผลได้
            by_arm[f"{k[0]}_{k[1].replace('-', '')}"] = {
                "n": int(len(v)), "mean_minutes": float(v.mean()),
                "median_minutes": float(np.median(v))}
        # สรุปรายสถาปัตยกรรม สำหรับตารางต้นทุนในเปเปอร์ — ต้องอ่านจากไฟล์ ไม่ใช่พิมพ์เอง
        per_arch = defaultdict(list)
        for r in rows:
            per_arch[r["run"].split("__")[0]].append(r)
        by_arch = {}
        for arch in sorted(per_arch):
            g = per_arch[arch]
            m = np.array([x["train_minutes"] for x in g], dtype=float)
            e = np.array([x.get("epochs_run") or 0 for x in g], dtype=float)
            mem = [x["peak_gpu_mem_gb"] for x in g if x.get("peak_gpu_mem_gb")]
            pr = [x["params_millions"] for x in g if x.get("params_millions")]
            by_arch[arch] = {
                "n_runs": int(len(g)),
                "params_millions": float(max(pr)) if pr else None,
                "peak_gpu_mem_gb": float(max(mem)) if mem else None,
                "mean_minutes": float(m.mean()),
                "median_minutes": float(np.median(m)),
                "mean_epochs": float(e.mean()),
            }
        gpus = sorted({r.get("gpu") for r in rows if r.get("gpu")})
        mem = [r["peak_gpu_mem_gb"] for r in rows if r.get("peak_gpu_mem_gb")]
        params = sorted({(r["run"].split("__")[0], r.get("params_millions"))
                         for r in rows if r.get("params_millions")})
        print(f"  GPU: {', '.join(gpus)}")
        if mem:
            print(f"  peak GPU mem: {min(mem):.1f}-{max(mem):.1f} GB")
        print("  พารามิเตอร์ต่อสถาปัตยกรรม (ล้าน):")
        for arch, p in params:
            print(f"    {arch:<20} {p}")
        summary = {
            "n_runs_instrumented": len(rows),
            "train_minutes": {"mean": float(mins.mean()),
                              "median": float(np.median(mins)),
                              "min": float(mins.min()), "max": float(mins.max()),
                              "total_hours": float(mins.sum() / 60)},
            "epochs": {"mean": float(eps.mean()), "median": float(np.median(eps))},
            "gpus": gpus,
            "peak_gpu_mem_gb": {"min": float(min(mem)), "max": float(max(mem))} if mem else None,
            "params_millions": {a: p for a, p in params},
            "train_minutes_by_arm": by_arm,
            "by_architecture": by_arch,
        }

    out = {"sacct": {"n_jobs": len(jobs), "n_completed": len(done),
                     "gpu_hours_completed": gpu_hours,
                     "gpu_hours_including_failed": all_hours,
                     "job_name": args.job_name, "since": args.since},
           "runs": summary}
    dest = Path(args.results).parent / "analysis" / "results" / "compute_budget.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))

    quoted = gpu_hours or (summary.get("train_minutes", {}).get("total_hours", 0.0))
    print(f"\nประโยคสำหรับ manuscript:")
    print(f"  Training the {len(done) or len(rows)} models required approximately "
          f"{quoted:.0f} GPU-hours on NVIDIA A100-80GB accelerators.")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
