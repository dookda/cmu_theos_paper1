#!/usr/bin/env python3
"""Random tile-level split matched to the spatial split, for addendum A.

See `paper1/configs/addendum_prereg_2026-10.md`. The only thing that differs from
`spatial_splits.json` is how pool-A tiles are assigned: here at random, with the same
train/val/test sizes, so any change in accuracy or in the factorial conclusions comes
from the spatial arrangement and not from training-set size, labels or code.

External (Scene 2) is copied unchanged from `spatial_splits.json`. Pool-A tiles in pixel
contact with an external tile are excluded before drawing, so external stays at zero
contact for both designs and can serve as the common yardstick.

Usage
-----
  python paper1/scripts/build_random_split.py
"""
import json
import random
from pathlib import Path

from build_paper1_spatial_splits import audit, load_tiles, neighbours

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "analysis" / "results"
SEED = 401  # same seed as the spatial split


def main():
    spatial = json.loads((RES / "spatial_splits.json").read_text())
    sizes = {s: len(spatial["splits"][s]) for s in ("train", "val", "test")}
    external = list(spatial["splits"]["external"])

    tiles, _ = load_tiles()
    pool_a = sorted(n for n, t in tiles.items() if t["pool"] == "A")
    ext_set = set(external)
    index = {(t["col"], t["row"]): n for n, t in tiles.items()
             if n in ext_set or t["pool"] == "A"}
    touching_ext = {n for n in pool_a
                    if any(o in ext_set for o in neighbours(n, tiles, index))}
    eligible = [n for n in pool_a if n not in touching_ext]
    need = sum(sizes.values())
    if len(eligible) < need:
        raise SystemExit(f"only {len(eligible)} eligible tiles, need {need}")

    rng = random.Random(SEED)
    rng.shuffle(eligible)
    split_of, i = {}, 0
    for s in ("train", "val", "test"):
        for n in eligible[i:i + sizes[s]]:
            split_of[n] = s
        i += sizes[s]
    for n in external:
        split_of[n] = "external"

    leak = audit(split_of, tiles)
    if leak["external"]["touching_train"]:
        raise SystemExit(f"external touches train: {leak['external']}")

    payload = {
        "design": "random tile-level split, sizes matched to spatial_splits.json",
        "seed": SEED,
        "classes": spatial["classes"],
        "remap": spatial["remap"],
        "tile_px": spatial["tile_px"],
        "step_px": spatial["step_px"],
        "pool_a_excluded_touching_external": sorted(touching_ext),
        "unused_pool_a": sorted(eligible[i:]),
        "leakage_audit": leak,
        "splits": {s: sorted(n for n, v in split_of.items() if v == s)
                   for s in ("train", "val", "test", "external")},
        # same Scene 2 blocks as the spatial split, so external is scored per block alike
        "blocks_external": spatial["blocks_external"],
        "matched_to": "spatial_splits.json",
    }
    out = RES / "random_splits.json"
    out.write_text(json.dumps(payload, indent=1))
    print({s: len(v) for s, v in payload["splits"].items()})
    print("excluded (touch external):", len(touching_ext), "unused:", len(payload["unused_pool_a"]))
    print("leakage audit:", leak)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
