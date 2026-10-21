#!/usr/bin/env python3
"""Characterise how far each held-out tile sits from the nearest training tile.

The external set is not one compact area. Scene 2 was labelled in patches, so its
287 tiles fall into a strip that abuts the training area and several groups far to
the south. Reporting them as a single "external" number would average a
near-transfer and a far-transfer regime together and hide the more interesting
result.

Writing the distances out separately keeps the live spatial_splits.json untouched
while training jobs are reading it, and lets the evaluation stratify held-out
performance by distance — which turns an awkward property of the reference data
into a distance-decay result the paper can actually use.

Groups are cut at natural gaps in the north-south ordering rather than at a round
number, so they follow the labelling patches instead of imposing a grid on them.

Usage
-----
  python paper1/scripts/external_distance.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
SPLITS = ROOT / "paper1" / "analysis" / "results" / "spatial_splits.json"
META = ROOT / "paper2" / "analysis" / "results" / "tiles_metadata.json"
OUT = ROOT / "paper1" / "analysis" / "results" / "external_distance.json"

GAP_M = 1000.0        # ช่องว่างที่ถือว่าเป็นคนละก้อน
FAR_M = 2000.0        # เกณฑ์แบ่ง near / far transfer


def centres():
    meta = json.loads(META.read_text())
    meta = meta["tiles"] if isinstance(meta, dict) and "tiles" in meta else meta
    return {Path(m["filename"]).stem:
            ((m["bounds"]["left"] + m["bounds"]["right"]) / 2,
             (m["bounds"]["bottom"] + m["bounds"]["top"]) / 2) for m in meta}


def main():
    sp = json.loads(SPLITS.read_text())
    C = centres()
    train = np.array([C[t] for t in sp["splits"]["train"] if t in C])

    report = {"far_threshold_m": FAR_M, "gap_m": GAP_M, "splits": {}}
    for split in ("val", "test", "external"):
        names = [t for t in sp["splits"][split] if t in C]
        P = np.array([C[t] for t in names])
        d = np.sqrt(((P[:, None, :] - train[None, :, :]) ** 2).sum(-1)).min(1)
        entry = {
            "n": len(names),
            "distance_m": {"min": float(d.min()), "median": float(np.median(d)),
                           "mean": float(d.mean()), "max": float(d.max())},
            "n_beyond_far_threshold": int((d > FAR_M).sum()),
            "per_tile": {n: round(float(v), 1) for n, v in zip(names, d)},
        }
        if split == "external":
            ys = P[:, 1]
            order = np.argsort(ys)
            ysort, dsort = ys[order], d[order]
            nsort = [names[i] for i in order]
            cuts = list(np.where(np.diff(ysort) > GAP_M)[0]) + [len(ysort) - 1]
            groups, start = [], 0
            for c in cuts:
                sl = slice(start, c + 1)
                groups.append({
                    "northing_min": float(ysort[sl].min()),
                    "northing_max": float(ysort[sl].max()),
                    "n": int(c + 1 - start),
                    "distance_median_m": float(np.median(dsort[sl])),
                    "regime": "far" if np.median(dsort[sl]) > FAR_M else "near",
                    "tiles": sorted(nsort[sl]),
                })
                start = c + 1
            entry["groups"] = groups
            near = [t for g in groups if g["regime"] == "near" for t in g["tiles"]]
            far = [t for g in groups if g["regime"] == "far" for t in g["tiles"]]
            entry["near_tiles"], entry["far_tiles"] = sorted(near), sorted(far)
        report["splits"][split] = entry

    OUT.write_text(json.dumps(report, indent=1))

    for split, e in report["splits"].items():
        dm = e["distance_m"]
        print(f"{split:9s} n={e['n']:3d}  distance to nearest train tile (m): "
              f"min {dm['min']:6.0f}  median {dm['median']:6.0f}  max {dm['max']:7.0f}")
    ext = report["splits"]["external"]
    print(f"\nexternal splits into {len(ext['groups'])} labelling patches:")
    for g in ext["groups"]:
        print(f"  N {g['northing_min']:,.0f}-{g['northing_max']:,.0f}  n={g['n']:3d}  "
              f"median {g['distance_median_m']:6,.0f} m  [{g['regime']}]")
    print(f"\nnear-transfer tiles: {len(ext['near_tiles'])}  "
          f"far-transfer tiles: {len(ext['far_tiles'])}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
