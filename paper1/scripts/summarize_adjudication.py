#!/usr/bin/env python3
"""Recompute the parcel-adjudication summary (Sec. 3.3, Fig. 7 right) from the raw reviewer sheets.

Inputs  analysis/results/from_paper3/g1_pilot_key.json   80 parcels: 60 candidates drawn from the
                                                         disagreement with the 2020 layer, 20 stable controls
        analysis/results/from_paper3/g1_pilot_r1.json, g1_pilot_r2.json   blinded answers:
                                                         q1 = current class, q2 = changed / unchanged
Output  printed; compared with the stored summary g1_adjudication_pilot.json, which was computed
        in a companion project.

Reproduced exactly: Cohen's kappa of both questions, the candidate counts (changed / unchanged /
no consensus), the confirmed-change rate and its Wilson interval, and the stable-control figures.
The four-way decomposition of the candidates (real change / reference error / 2020 layer or
definition / undecided) depends on a rule that the stored file does not record; the rule below is
the plain reading of the two questions and differs from the stored counts by two parcels.
"""
import collections
import json
from math import sqrt
from pathlib import Path

D = Path(__file__).resolve().parent.parent / "analysis" / "results" / "from_paper3"


def kappa(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    n = len(pairs)
    po = sum(x == y for x, y in pairs) / n
    ca, cb = collections.Counter(x for x, _ in pairs), collections.Counter(y for _, y in pairs)
    pe = sum(ca[c] * cb[c] for c in set(ca) | set(cb)) / n / n
    return (po - pe) / (1 - pe)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 4), round(c + h, 4)


def main():
    key = json.loads((D / "g1_pilot_key.json").read_text())
    ans = {r: json.loads((D / f"g1_pilot_{r}.json").read_text())["answers"] for r in ("r1", "r2")}
    items = key["items"]
    ids = [str(it["id"]) for it in items]
    q1 = [[ans[r][i].get("q1") for i in ids] for r in ("r1", "r2")]
    q2 = [[ans[r][i].get("q2") for i in ids] for r in ("r1", "r2")]

    def cons(q, j):
        return q[0][j] if q[0][j] is not None and q[0][j] == q[1][j] else None

    cand = [(j, it) for j, it in enumerate(items) if it["kind"] == "candidate"]
    stab = [(j, it) for j, it in enumerate(items) if it["kind"] == "stable"]
    c2 = collections.Counter(cons(q2, j) or "no_consensus" for j, _ in cand)
    decided = c2["changed"] + c2["unchanged"]
    s2 = collections.Counter(cons(q2, j) or "no_consensus" for j, _ in stab)
    s_dec = s2["changed"] + s2["unchanged"]
    s1 = collections.Counter(("agree" if cons(q1, j) == it["to_name"] else "disagree") if cons(q1, j)
                             else "undecided" for j, it in stab)
    decomp = collections.Counter()
    for j, it in cand:
        a1, a2 = cons(q1, j), cons(q2, j)
        if a1 is None or a2 is None:
            decomp["undecided"] += 1
        elif a1 != it["to_name"]:
            decomp["t1_label_error"] += 1
        else:
            decomp["real_change" if a2 == "changed" else "t0_or_definition"] += 1
    got = {
        "inter_reviewer_kappa": {"r1|r2:q1": round(kappa(*q1), 4), "r1|r2:q2": round(kappa(*q2), 4)},
        "candidates": {"n": len(cand), "n_decided": decided, "changed": c2["changed"],
                       "unchanged": c2["unchanged"], "no_consensus": c2["no_consensus"],
                       "confirmed_change_rate": round(c2["changed"] / decided, 4),
                       "ci95": list(wilson(c2["changed"], decided))},
        "stable_controls": {"n": len(stab), "n_decided": s_dec,
                            "false_alarm_rate": round(s2["changed"] / s_dec, 4),
                            "ci95": list(wilson(s2["changed"], s_dec)), "q1_vs_label": dict(s1)},
        "disagreement_decomposition": dict(decomp),
    }
    ref = json.loads((D / "g1_adjudication_pilot.json").read_text())
    for k, v in got.items():
        stored = ref[k]
        if isinstance(v, dict):
            stored = {kk: stored.get(kk) for kk in v}
        print(f"{k:28s} {'OK ' if stored == v else 'DIFF'} recomputed {v}")
        if stored != v:
            print(f"{'':28s}      stored     {stored}")


if __name__ == "__main__":
    main()
