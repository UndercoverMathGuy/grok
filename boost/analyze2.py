"""Round-2 analyzer (prereg: boost/README.md Round 2). Includes the
round-1 jump@200 numbers for the lock-hardening curve.

Usage: python3 boost/analyze2.py [--runs runs_boost] [--out notes/boost2_summary.json]
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
    ap.add_argument("--out", default=str(ROOT / "notes" / "boost2_summary.json"))
    args = ap.parse_args()
    R = Path(args.runs)
    meta = json.loads((R / "round2_meta.json").read_text())
    donors = {rd.name: load(rd) for rd in sorted((R / "donor").glob("*"))}

    def rows(arm, offset):
        rr = []
        for rd in sorted((R / arm).glob("*")):
            r = load(rd)
            d = donors[rd.name]
            m = meta["runs"].get(rd.name, {})
            r["abs_t_gen"] = (r["t_gen"] + offset) if r["t_gen"] >= 0 else -1
            r["rel_dev"] = (abs(r["abs_t_gen"] - d["t_gen"]) / d["t_gen"]
                            if r["abs_t_gen"] >= 0 and d["t_gen"] > 0 else None)
            r["ratio"] = (r["abs_t_gen"] / d["t_gen"]
                          if r["abs_t_gen"] >= 0 and d["t_gen"] > 0 else None)
            r["flip"] = m.get("target") in r["committee"] if m else None
            r["leader_out"] = (m.get("leader") not in r["committee"]) if m else None
            r["J_donor"] = jaccard(r["committee"], d["committee"])
            r["name"] = rd.name
            rr.append(r)
        return rr

    arms = {}
    print("arm        n  flips  ldr_out  medJ  med|dt|/t  med_ratio")
    plan = [("swap", 20), ("swapsham", 20), ("jump", 200),
            ("jump500", 500), ("jump1000", 1000), ("gate1000", 1000),
            ("jumpgate", 200)]
    for arm, off in plan:
        if not (R / arm).exists():
            continue
        rr = rows(arm, off)
        arms[arm] = rr
        fm = lambda v, s: format(v, s) if v is not None else "-"
        print(f"{arm:<10} {len(rr):>1} {sum(bool(r['flip']) for r in rr):>6} "
              f"{sum(bool(r['leader_out']) for r in rr):>8} "
              f"{fm(med([r['J_donor'] for r in rr]), '5.2f'):>5} "
              f"{fm(med([r['rel_dev'] for r in rr]), '10.3f'):>10} "
              f"{fm(med([r['ratio'] for r in rr]), '9.3f'):>9}")

    V = {}
    sw = arms.get("swap", [])
    V["P-B5"] = {"flips": sum(bool(r["flip"]) for r in sw), "n": len(sw),
                 "result": "PASS" if sum(bool(r["flip"]) for r in sw) >= 6
                 else "FAIL"}
    swd = med([r["rel_dev"] for r in sw])
    shd = med([r["rel_dev"] for r in arms.get("swapsham", [])])
    V["P-B6"] = {"swap_med_rel_dev": swd, "sham_med_rel_dev": shd,
                 "result": "PASS" if swd is not None and swd <= 0.15
                 and (shd is None or swd <= shd + 0.05) else "FAIL"}
    curve = {a: med([r["J_donor"] for r in arms[a]])
             for a in ("jump", "jump500", "jump1000") if a in arms}
    g = med([r["J_donor"] for r in arms.get("gate1000", [])])
    j1000 = curve.get("jump1000")
    V["P-B7"] = {"J_curve": curve, "gate1000": g,
                 "result": ("VOID (gate)" if g is None or g < 0.8 else
                            "PASS" if j1000 is not None and j1000 >= 0.8
                            else "FAIL")}
    r1000 = med([r["ratio"] for r in arms.get("jump1000", [])])
    V["P-B8"] = {"med_ratio": r1000,
                 "result": "PASS" if r1000 is not None and r1000 <= 0.7
                 else "FAIL"}
    print()
    for k, v in V.items():
        print(f"{k}: {v}")
    ok = all(str(v["result"]).startswith("PASS") for v in V.values())
    print(f"\nDOUBLE DISSOCIATION (round 2): {'STANDS' if ok else 'see failures'}")

    o = Path(args.out)
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps({"arms": {k: v for k, v in arms.items()},
                             "verdicts": V}, indent=1, default=str))
    print(f"wrote {o}")


if __name__ == "__main__":
    main()
