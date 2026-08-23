"""Analyzer for the boost/jump double-dissociation cohort.

Prereg: boost/README.md. Usage:
    python3 boost/analyze_boost.py [--runs runs_boost] [--out notes/boost_summary.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dial"))
from analyze_dial import committee_from_coeffs, jaccard  # noqa: E402


def load(rd):
    z = np.load(rd / "spectra.npz")
    ep = np.asarray(z["epochs"])
    ok = np.asarray(z["test_acc"]) >= 0.99
    h = np.where(ok[:-1] & ok[1:])[0]
    return {"committee": committee_from_coeffs(z["coeffs"][-1]),
            "t_gen": int(ep[h[0]]) if len(h) else -1,
            "acc": float(z["test_acc"][-1])}


def med(v):
    v = [x for x in v if x is not None]
    return float(np.median(v)) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "runs_boost"))
    ap.add_argument("--out", default=str(ROOT / "notes" / "boost_summary.json"))
    args = ap.parse_args()
    R = Path(args.runs)
    meta = json.loads((R / "boost_meta.json").read_text())
    tb, tj = meta["t_boost"], meta["t_jump"]

    donors = {rd.name: load(rd) for rd in sorted((R / "donor").glob("*"))}
    out = {"arms": {}}

    def arm_rows(arm, offset):
        rows = []
        for rd in sorted((R / arm).glob("*")):
            r = load(rd)
            d = donors[rd.name]
            m = meta["runs"][rd.name]
            r["abs_t_gen"] = (r["t_gen"] + offset) if r["t_gen"] >= 0 else -1
            r["donor_t_gen"] = d["t_gen"]
            r["rel_dev"] = (abs(r["abs_t_gen"] - d["t_gen"]) / d["t_gen"]
                            if r["abs_t_gen"] >= 0 and d["t_gen"] > 0 else None)
            r["flip"] = m["target"] in r["committee"]
            r["J_donor"] = jaccard(r["committee"], d["committee"])
            r["name"] = rd.name
            rows.append(r)
        return rows

    print("arm         n  flips  medJ_donor  med|dt|/t  med_abs_tgen")
    arms = {}
    for arm, off in ([(f"boost_a{a:g}", tb) for a in (2, 4, 8)]
                     + [(f"sham_a{a:g}", tb) for a in (4, 8)]
                     + [("jump", tj), ("jumpgate", tj)]):
        if not (R / arm).exists():
            continue
        rows = arm_rows(arm, off)
        arms[arm] = rows
        out["arms"][arm] = rows
        fm = lambda v, s: format(v, s) if v is not None else "-"
        print(f"{arm:<10} {len(rows):>2} {sum(r['flip'] for r in rows):>6} "
              f"{fm(med([r['J_donor'] for r in rows]), '11.2f'):>11} "
              f"{fm(med([r['rel_dev'] for r in rows]), '10.3f'):>10} "
              f"{fm(med([r['abs_t_gen'] for r in rows if r['abs_t_gen'] >= 0]), '13.0f'):>13}")

    V = {}
    f2 = sum(r["flip"] for r in arms.get("boost_a2", []))
    f4 = sum(r["flip"] for r in arms.get("boost_a4", []))
    f8 = sum(r["flip"] for r in arms.get("boost_a8", []))
    V["P-B1"] = {"flips_by_alpha": [f2, f4, f8],
                 "result": ("KILL" if f8 <= 1 else
                            "PASS" if f8 >= 6 and f2 <= f4 <= f8 else "partial")}
    bdev = med([r["rel_dev"] for a in (2, 4, 8)
                for r in arms.get(f"boost_a{a:g}", [])])
    sdev = med([r["rel_dev"] for a in (4, 8)
                for r in arms.get(f"sham_a{a:g}", [])])
    V["P-B2"] = {"boost_med_rel_dev": bdev, "sham_med_rel_dev": sdev,
                 "result": "PASS" if bdev is not None and bdev <= 0.15
                 and (sdev is None or bdev <= sdev + 0.05) else "FAIL"}
    jr = [(r["abs_t_gen"], r["donor_t_gen"]) for r in arms.get("jump", [])
          if r["abs_t_gen"] >= 0]
    ratio = med([a / d for a, d in jr])
    V["P-B3"] = {"med_tgen_ratio": ratio,
                 "result": "PASS" if ratio is not None and ratio <= 0.5 else "FAIL"}
    jj = med([r["J_donor"] for r in arms.get("jump", [])])
    gj = med([r["J_donor"] for r in arms.get("jumpgate", [])])
    V["P-B4"] = {"J_jump": jj, "J_gate": gj,
                 "result": ("VOID (gate)" if gj is None or gj < 0.8 else
                            "PASS" if jj >= 0.8 else
                            "FAIL -> lock-in is data-conditional (finding)")}
    print()
    for k, v in V.items():
        print(f"{k}: {v}")
    ok = all(str(v.get("result", "")).startswith("PASS") for v in V.values())
    print(f"\nDOUBLE DISSOCIATION: {'STANDS' if ok else 'see failed arms above'}")
    out["verdicts"] = V

    o = Path(args.out)
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(out, indent=1, default=str))
    print(f"wrote {o}")


if __name__ == "__main__":
    main()
