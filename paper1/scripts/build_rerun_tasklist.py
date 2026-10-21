#!/usr/bin/env python3
"""Expand experiments_rerun.yaml into a flat task list for the SLURM array.

One line per run, so a SLURM array task only has to index into the file. Emitting
the list up front (rather than computing it inside the job) keeps the grid
auditable: the same file records what was meant to run, and `--check` compares it
against what actually produced results.

Usage
-----
  python paper1/scripts/build_rerun_tasklist.py                 # phase 1 + 2
  python paper1/scripts/build_rerun_tasklist.py --phase 1
  python paper1/scripts/build_rerun_tasklist.py --check hpc_output/rerun
"""
import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
REG = ROOT / "configs" / "experiments_rerun.yaml"
OUT = ROOT / "analysis" / "results"


def build(reg, phase=None):
    cond = {c["name"]: c for c in reg["conditions"]}
    core = [a for a in reg["architectures"] if a["core"]]
    other = [a for a in reg["architectures"] if not a["core"]]
    baseline = reg["phases"][1]["other_architectures"]

    tasks = []

    def add(arch, c, seed, ph):
        tasks.append({
            "id": len(tasks),
            "phase": ph,
            "run": f"{arch['name']}__{c['name']}__s{seed}",
            "model": arch["model"],
            "encoder": arch["encoder"],
            "condition": c["name"],
            "use_nir": c["bands"] == "rgbnir",
            "augment": c["augment"],
            "class_weights": "auto" if c["weights"] else "none",
            "seed": seed,
        })

    for seed in reg["seeds"]:
        for arch in core:                      # phase 1: full factorial
            for c in reg["conditions"]:
                add(arch, c, seed, 1)
        for arch in other:                     # phase 1: baseline only
            for name in baseline:
                add(arch, cond[name], seed, 1)
        for arch in other:                     # phase 2: rest of the grid
            for c in reg["conditions"]:
                if c["name"] in baseline:
                    continue
                add(arch, c, seed, 2)

    if phase:
        tasks = [t for t in tasks if t["phase"] == phase]
        for i, t in enumerate(tasks):
            t["id"] = i
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, choices=[1, 2])
    ap.add_argument("--check", metavar="RESULTS_DIR",
                    help="รายงาน run ที่ยังไม่มีผลใน RESULTS_DIR")
    args = ap.parse_args()

    reg = yaml.safe_load(open(REG))
    tasks = build(reg, args.phase)

    n_arch = len(reg["architectures"])
    n_cond = len(reg["conditions"])
    n_seed = len(reg["seeds"])
    by_phase = {}
    for t in tasks:
        by_phase[t["phase"]] = by_phase.get(t["phase"], 0) + 1
    print(f"grid: {n_cond} conditions x {n_arch} architectures x {n_seed} seeds "
          f"= {n_cond * n_arch * n_seed} runs")
    for p in sorted(by_phase):
        print(f"  phase {p}: {by_phase[p]} runs")
    print(f"tasks emitted: {len(tasks)}  (array 0-{len(tasks) - 1})")

    if args.check:
        base = Path(args.check)
        missing = [t["run"] for t in tasks
                   if not list((base / t["run"] / "logs").glob("*_best_metrics.json"))]
        print(f"\nยังไม่มีผล: {len(missing)}/{len(tasks)}")
        for r in missing[:20]:
            print(f"  {r}")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    suffix = f"_phase{args.phase}" if args.phase else ""
    path = OUT / f"rerun_tasklist{suffix}.json"
    path.write_text(json.dumps({"splits": reg["splits"],
                                "training": reg["training"],
                                "tasks": tasks}, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
