"""Analyzer for the lesion cohort (prereg: lesion/README.md).

Usage: python3 lesion/analyze_lesion.py [--runs runs_lesion] [--out notes/lesion_summary.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dial"))
from analyze_dial import committee_from_coeffs, jaccard  # noqa: E402

ARMS = ["rand_we", "rand_rest"]
TS = [150, 500]


def comm(rd):
    z = np.load(rd / "spectra.npz")
    return committee_from_coeffs(z["coeffs"][-1]), float(z["test_acc"][-1])


def perm_p(obs_med, vals, null_pool, n=20000, seed=0):
    """One-sided: are vals drawn higher than the null pool?"""
    rng = np.random.default_rng(seed)
    combined = np.array(list(vals) + list(null_pool))
    k = len(vals)
    hits = sum(np.median(rng.permutation(combined)[:k]) >= obs_med
               for _ in range(n))
    return (hits + 1) / (n + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "runs_lesion"))
    ap.add_argument("--out", default=str(ROOT / "notes" / "lesion_summary.json"))
    args = ap.parse_args()
    R = Path(args.runs)

    donors = {rd.name: comm(rd)[0] for rd in sorted((R / "donor").glob("*"))}

    ctrl = [jaccard(comm(rd)[0], donors[rd.name])
            for rd in sorted((R / "ctrl_t500").glob("*"))]
    gate = float(np.median(ctrl)) if ctrl else None
    print(f"GATE ctrl_t500 median J_self = {gate}  "
          f"-> {'OK' if gate and gate >= 0.8 else 'VOID'}")

    out = {"gate": gate, "arms": {}}
    print(f"\n{'arm':>10} {'t':>4} {'n':>3} {'medJ_recover':>13} "
          f"{'null':>6} {'perm_p':>7} {'acc':>6}")
    for arm in ARMS:
        for t in TS:
            d = R / f"{arm}_t{t}"
            if not d.exists():
                continue
            js, accs, name_j = [], [], {}
            for rd in sorted(d.glob("*")):
                c, acc = comm(rd)
                js.append(jaccard(c, donors[rd.name]))
                accs.append(acc)
                name_j[rd.name] = c
            # shuffled null: lesioned run vs OTHER donors sharing the mask
            null = [jaccard(c, donors[o]) for n_, c in name_j.items()
                    for o in donors if o != n_
                    and o.split("_")[0] == n_.split("_")[0]]
            med = float(np.median(js))
            p = perm_p(med, js, null)
            out["arms"][f"{arm}_t{t}"] = {
                "J": js, "median": med, "null_median": float(np.median(null)),
                "perm_p": p, "acc": float(np.median(accs))}
            print(f"{arm:>10} {t:>4} {len(js):>3} {med:>13.2f} "
                  f"{np.median(null):>6.2f} {p:>7.3f} {np.median(accs):>6.3f}")

    a = out["arms"]
    g = lambda k, f: a.get(k, {}).get(f)
    print()
    p1 = (g("rand_we_t500", "median") or 0) >= 0.5 and (g("rand_we_t500", "perm_p") or 1) < 0.05
    print(f"P-L1 (downstream copy restores): {'PASS' if p1 else 'FAIL'}")
    p2 = (g("rand_rest_t500", "perm_p") or 1) < 0.05
    print(f"P-L2 (W_E copy restores): {'PASS' if p2 else 'no recovery -> downstream-only storage'}")
    p3 = all((g(f'{arm}_t500', 'median') or 0) >= (g(f'{arm}_t150', 'median') or 0)
             for arm in ARMS)
    print(f"P-L3 (redundancy grows 150->500): {'PASS' if p3 else 'FAIL'}")
    print(f"STRICT REDUNDANCY (both copies restore at t=500): "
          f"{'YES' if p1 and p2 else 'NO'}")
    out["verdicts"] = {"P-L1": p1, "P-L2": p2, "P-L3": p3,
                       "strict_redundancy": p1 and p2}

    o = Path(args.out)
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(out, indent=1, default=str))
    print(f"\nwrote {o}")


if __name__ == "__main__":
    main()
