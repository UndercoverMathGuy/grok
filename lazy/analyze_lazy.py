"""Analyzer for the linearized-twin cohort (prereg: lazy/README.md,
P-J1..P-J3). numpy-only; runs on the pod.

Usage: python3 lazy/analyze_lazy.py [--runs runs_lazy]
           [--out notes/lazy_summary.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dial"))
from analyze_dial import committee_from_coeffs, jaccard, mw_auc  # noqa: E402


def sustained(epochs, ok):
    hits = np.where(ok[:-1] & ok[1:])[0]
    return int(epochs[hits[0]]) if len(hits) else -1


def load(rd):
    z = np.load(rd / "spectra.npz")
    ep = np.asarray(z["epochs"])
    comm = committee_from_coeffs(z["coeffs"][-1])
    nf = z["energy"].shape[1]
    m = np.zeros(nf, bool)
    m[[k - 1 for k in comm]] = True
    E = np.asarray(z["energy"])
    auc = np.array([mw_auc(E[s], m) for s in range(len(ep))])
    return {"committee": comm, "epochs": ep,
            "t_gen": sustained(ep, np.asarray(z["test_acc"]) >= 0.99),
            "t_ident": sustained(ep, auc >= 0.9),
            "train_acc": float(z["train_acc"][-1]),
            "test_acc": float(z["test_acc"][-1]),
            "energy0": np.asarray(z["energy"][0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "runs_lazy"))
    ap.add_argument("--out", default=str(ROOT / "notes" / "lazy_summary.json"))
    args = ap.parse_args()
    R = Path(args.runs)

    real = {rd.name: load(rd) for rd in sorted((R / "real_wd1").glob("*"))
            if (rd / "spectra.npz").exists()}

    def pairs_for(arm):
        rows = []
        for rd in sorted((R / arm).glob("*")):
            if not (rd / "spectra.npz").exists() or rd.name not in real:
                continue
            tw, re = load(rd), real[rd.name]
            K = len(re["committee"])
            static = sorted((np.argsort(re["energy0"])[::-1][:K] + 1).tolist())
            rows.append({"base": rd.name,
                         "J": jaccard(tw["committee"], re["committee"]),
                         "J_static": jaccard(static, re["committee"]),
                         "twin_committee": tw["committee"],
                         "real_committee": re["committee"],
                         "twin_train_acc": tw["train_acc"],
                         "twin_test_acc": tw["test_acc"],
                         "twin_t_gen": tw["t_gen"],
                         "twin_t_ident": tw["t_ident"],
                         "real_t_gen": re["t_gen"],
                         "real_t_ident": re["t_ident"]})
        return rows

    tables = {}
    for arm in ("lin_wd1", "quad_wd1"):
        rows = tables[arm] = pairs_for(arm)
        if not rows:
            continue
        print(f"\n{arm}:")
        print(f"{'base':<12} {'J(tw,real)':>10} {'J_static':>8} {'tw_tr':>6} "
              f"{'tw_te':>6} {'tw_t_id':>8} {'real_t_gen':>10}")
        for r in rows:
            print(f"{r['base']:<12} {r['J']:>10.2f} {r['J_static']:>8.2f} "
                  f"{r['twin_train_acc']:>6.3f} {r['twin_test_acc']:>6.3f} "
                  f"{r['twin_t_ident']:>8} {r['real_t_gen']:>10}")

    V = {}
    rows = tables["lin_wd1"]
    beats = sum(r["J"] > r["J_static"] for r in rows)
    medJ = float(np.median([r["J"] for r in rows])) if rows else None
    medS = float(np.median([r["J_static"] for r in rows])) if rows else None
    V["P-J1"] = {"beats_static": beats, "n": len(rows), "median_J": medJ,
                 "median_J_static": medS,
                 "result": ("KILL -> order-2 branch"
                            if medJ is not None and medJ <= medS else
                            "PASS" if beats >= 6 and medJ >= 0.5
                            else "FAIL/partial")}
    gen = sum(r["twin_t_gen"] >= 0 for r in rows)
    fit = sum(r["twin_train_acc"] >= 0.9 for r in rows)
    V["P-J2"] = {"lin_generalized": gen, "lin_train_fit": fit,
                 "result": ("FALSIFIED (lazy model generalized)" if gen > 0
                            else "PASS" if fit >= 6
                            else "GRAY (tangent model did not fit train)")}
    early = sum(0 <= r["twin_t_ident"] <= 500 for r in rows)
    V["P-J3"] = {"lin_t_ident_le_500": early,
                 "result": "PASS" if early >= 6 else "FAIL/partial"}

    q = tables["quad_wd1"]
    if q:
        qbeats = sum(r["J"] > r["J_static"] for r in q)
        qmedJ = float(np.median([r["J"] for r in q]))
        qmedS = float(np.median([r["J_static"] for r in q]))
        V["P-Q1"] = {"beats_static": qbeats, "n": len(q), "median_J": qmedJ,
                     "median_J_static": qmedS, "lin_median_J": medJ,
                     "result": ("KILL -> order >= 3"
                                if qmedJ <= qmedS else
                                "PASS" if qbeats >= 6 and qmedJ >= 0.5
                                and (medJ is None or qmedJ > medJ)
                                else "FAIL/partial")}
        qgen = sum(r["twin_t_gen"] >= 0 for r in q)
        V["P-Q2"] = {"quad_generalized": qgen,
                     "max_test_acc": float(max(r["twin_test_acc"]
                                               for r in q)),
                     "result": ("BRANCH: order-2 generalizes — "
                                "'learning at order infinity' retracted"
                                if qgen > 0 else "PASS")}

    # exploratory: wd0 arms
    ex = {}
    for arm in ("lin_wd0", "quad_wd0"):
        w0 = [load(rd) | {"base": rd.name}
              for rd in sorted((R / arm).glob("*"))
              if (rd / "spectra.npz").exists()]
        ex[arm] = [{"base": r["base"],
                    "J": jaccard(r["committee"],
                                 real[r["base"]]["committee"]),
                    "train_acc": r["train_acc"], "test_acc": r["test_acc"],
                    "t_gen": r["t_gen"]}
                   for r in w0 if r["base"] in real]

    print()
    for k, v in V.items():
        print(f"{k}: {json.dumps(v, default=str)}")
    o = Path(args.out)
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(
        {"pairs": {arm: rr for arm, rr in tables.items()},
         "wd0_exploratory": ex, "verdicts": V}, indent=1, default=str))
    print(f"wrote {o}")


if __name__ == "__main__":
    main()
