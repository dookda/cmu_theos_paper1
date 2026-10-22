#!/usr/bin/env python3
"""Seven-class and six-class scores side by side (for reporting both in the manuscript).

Three scorings of the same predictions, all from files already read (no new evaluation):

  P7       seven classes, background scored like any class (the headline metric)
  P6-eval  six real classes on reference pixels that are not background; a background
           prediction on such a pixel counts as an error (cm_std of the unified pass)
  P6-sup   as P6-eval, with the background logit suppressed (cm_sup of the unified pass)

Inputs:  analysis/results/unified/rerun_output/*.npz          (360 runs, test/external)
         analysis/results/ensemble_bg_eval_{test,external}.json (locked ensemble, std/sup)
         analysis/results/unified_results.json                   (F4 factor effects)
Outputs: analysis/results/six_class_summary.json
         analysis/figure/fig_models_7v6.png   (two panels: P7 and P6-sup)

Usage:  python paper1/scripts/six_class_summary.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_unified as A                     # noqa: E402
from analyze_addendum import miou6, kendall_tau  # noqa: E402

RES = Path(__file__).resolve().parent.parent / "analysis" / "results"
FIG = Path(__file__).resolve().parent.parent / "analysis" / "figure"
PROTO = {"P7": ("cm_std", A.miou7),
         "P6-eval": ("cm_std", lambda c: miou6(c)[0]),
         "P6-sup": ("cm_sup", lambda c: miou6(c)[0])}


def ensemble():
    out = {}
    for sp in ("test", "external"):
        cm = json.loads((RES / f"ensemble_bg_eval_{sp}.json").read_text())["confusion_matrix"]
        out[sp] = {"P7": A.miou7(np.array(cm["std"], float)),
                   "P6-eval": miou6(cm["std"])[0], "P6-sup": miou6(cm["sup"])[0],
                   "per_class_P6-sup": dict(zip(
                       ["vegetation", "water", "urban", "agriculture", "bare_soil", "road"],
                       miou6(cm["sup"])[1]))}
    return out


def main():
    runs = A.load_root("rerun_output")
    if len(runs) != 360:
        raise SystemExit(f"expected 360 runs in the unified pass, found {len(runs)}")
    out = {"source": "unified pass + ensemble_bg_eval (no new read)", "ensemble": ensemble(),
           "runs": {}, "arch_mean": {}, "kendall_tau_arch_vs_P7": {}}
    for sp in ("test", "external"):
        out["runs"][sp], out["arch_mean"][sp] = {}, {}
        for p, (k, fn) in PROTO.items():
            v = np.array([A.metric(d, sp, k, fn) for d in runs.values()])
            y, archs = A.grid(runs, sp, k, fn)
            am = y.mean(axis=(1, 2, 3, 4))
            out["runs"][sp][p] = {"mean": float(v.mean()), "min": float(v.min()),
                                  "max": float(v.max())}
            out["arch_mean"][sp][p] = dict(zip(archs, am.tolist()))
        a7 = out["arch_mean"][sp]["P7"]
        out["kendall_tau_arch_vs_P7"][sp] = {
            p: kendall_tau([a7[a] for a in a7], [out["arch_mean"][sp][p][a] for a in a7])
            for p in ("P6-eval", "P6-sup")}
    uni = json.loads((RES / "unified_results.json").read_text())["F4"]
    out["factor_effects_pp"] = {
        sp: {p: {f: {"mean": uni[sp][p][f]["mean"], "ci95": uni[sp][p][f]["ci95"]}
                 for f in ("bands", "augment", "weights")} for p in PROTO}
        for sp in ("test", "external")}
    (RES / "six_class_summary.json").write_text(json.dumps(out, indent=2))
    figure(runs, out["ensemble"])

    e = out["ensemble"]
    print("ensemble  test  P7 {P7:.2f}  P6-eval {P6-eval:.2f}  P6-sup {P6-sup:.2f}".format(**e["test"]))
    print("ensemble  ext   P7 {P7:.2f}  P6-eval {P6-eval:.2f}  P6-sup {P6-sup:.2f}".format(**e["external"]))
    for sp in ("test", "external"):
        for p in PROTO:
            r = out["runs"][sp][p]
            f = out["factor_effects_pp"][sp][p]
            print(f"{sp:8s} {p:8s} runs mean {r['mean']:.2f} [{r['min']:.1f}, {r['max']:.1f}]  "
                  + "  ".join(f"{k} {v['mean']:+.2f}" for k, v in f.items()))
    print("tau vs P7:", out["kendall_tau_arch_vs_P7"])
    print(f"wrote {RES / 'six_class_summary.json'} and {FIG / 'fig_models_7v6.png'}")


def figure(runs, ens):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from fig_results import NICE, _mk, _sr
    from palette import marker  # noqa: F401  (same palette module as fig_results)

    by = {p: defaultdict(list) for p in ("P7", "P6-sup")}
    for (a, c, s), d in runs.items():
        for p in by:
            k, fn = PROTO[p]
            by[p][a].append((A.metric(d, "test", k, fn), A.metric(d, "external", k, fn)))
    order = sorted(by["P7"], key=lambda a: -np.mean([t for t, _ in by["P7"][a]]))

    # drawn at print size (16 cm wide): every text element prints at 7-9 pt
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 4.0))
    titles = {"P7": "Seven classes, background scored",
              "P6-sup": "Six real classes, background not predicted"}
    for ax, p in zip(axes, ("P7", "P6-sup")):
        pts = [c for v in by[p].values() for pair in v for c in pair] + \
              [ens["test"][p], ens["external"][p]]
        lo, hi = min(pts) - 1.0, max(pts) + 1.4
        ax.plot([lo, hi], [lo, hi], ls=(0, (5, 4)), color="0.5", lw=0.8, zorder=0)
        for i, a in enumerate(order):
            v = np.array(by[p][a])
            ax.scatter(v[:, 0], v[:, 1], s=11, marker=_mk(i), facecolor=_sr(i),
                       edgecolor="black", lw=0.4, alpha=0.9,
                       label=f"{NICE.get(a, a)} (n={len(v)})", zorder=3)
        ax.scatter([ens["test"][p]], [ens["external"][p]], s=70, marker="*",
                   facecolor="black", edgecolor="black", zorder=4)
        ax.annotate(f"ensemble\n{ens['test'][p]:.2f} / {ens['external'][p]:.2f}",
                    xy=(ens["test"][p], ens["external"][p]), xytext=(-12, -3), va="center",
                    textcoords="offset points", fontsize=7.5, ha="right",
                    arrowprops=dict(arrowstyle="-", lw=0.6, color="black"))
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        ax.set_xlabel("mIoU, internal test split (%)", fontsize=8)
        ax.set_ylabel("mIoU, held-out second scene (%)", fontsize=8)
        ax.tick_params(labelsize=7.5, width=0.6, length=2.5)
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
        ax.grid(color="0.9", lw=0.5); ax.set_axisbelow(True)
        ax.set_title(titles[p], fontsize=8.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7.5, frameon=False, loc="lower center", ncol=3,
               handletextpad=0.3, columnspacing=1.0, labelspacing=0.3, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=1.5)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_models_7v6.png", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
