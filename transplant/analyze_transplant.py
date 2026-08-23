"""Analyzer for the transplant/handoff cohort (prereg: transplant/README.md).

Usage: python3 transplant/analyze_transplant.py [--runs runs_transplant] [--out notes/transplant_summary.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dial"))
from analyze_dial import committee_from_coeffs, jaccard  # noqa: E402

SWAP_TS = [0, 50, 150, 500]
CTRL_TS = [150, 500]


def comm(rd):
    z = np.load(rd / "spectra.npz")
    return committee_from_coeffs(z["coeffs"][-1]), float(z["test_acc"][-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "runs_transplant"))
    ap.add_argument("--out", default=str(ROOT / "notes" / "transplant_summary.json"))
    args = ap.parse_args()
    R = Path(args.runs)

    donors = {}
    for rd in sorted((R / "donor").glob("*")):
        donors[rd.name] = comm(rd)[0]           # ds{ds}_is{is}

    rows = []
    for t in SWAP_TS:
        for rd in sorted((R / f"hyb_t{t}").glob("*")):
            # A{a}toB{b}_ds{ds}
            name = rd.name
            a = name.split("to")[0][1:]
            b = name.split("toB")[1].split("_")[0]
            ds = name.split("_ds")[1]
            c, acc = comm(rd)
            ja = jaccard(c, donors[f"ds{ds}_is{a}"])
            jb = jaccard(c, donors[f"ds{ds}_is{b}"])
            rows.append({"t": t, "A": a, "B": b, "ds": ds, "acc": acc,
                         "J_A": ja, "J_B": jb, "adopts": "A" if ja > jb
                         else "B" if jb > ja else "tie"})
    ctrl = []
    for t in CTRL_TS:
        for rd in sorted((R / f"self_t{t}").glob("*")):
            c, acc = comm(rd)
            ctrl.append({"t": t, "run": rd.name, "acc": acc,
                         "J_self": jaccard(c, donors[rd.name])})

    print(f"{len(donors)} donors, {len(rows)} hybrids, {len(ctrl)} controls")
    gate = {t: float(np.median([c["J_self"] for c in ctrl if c["t"] == t]))
            for t in CTRL_TS}
    gate_ok = all(v >= 0.8 for v in gate.values())
    print(f"\nGATE  J_self medians {gate}  -> {'OK' if gate_ok else 'VOID'}")

    print(f"\n{'t':>5} {'n':>3} {'medJ_A':>7} {'medJ_B':>7} "
          f"{'A-wins':>7} {'B-wins':>7} {'medMax':>7}")
    curve = {}
    for t in SWAP_TS:
        rs = [r for r in rows if r["t"] == t]
        ja = float(np.median([r["J_A"] for r in rs]))
        jb = float(np.median([r["J_B"] for r in rs]))
        mx = float(np.median([max(r["J_A"], r["J_B"]) for r in rs]))
        aw = sum(r["adopts"] == "A" for r in rs)
        bw = sum(r["adopts"] == "B" for r in rs)
        curve[t] = {"J_A": ja, "J_B": jb, "A_wins": aw, "B_wins": bw,
                    "med_max": mx, "n": len(rs)}
        print(f"{t:>5} {len(rs):>3} {ja:>7.2f} {jb:>7.2f} {aw:>7} {bw:>7} {mx:>7.2f}")

    v1 = curve.get(0, {})
    px1 = (v1 and v1["J_A"] > v1["J_B"] and v1["A_wins"] >= 6)
    v2 = curve.get(500, {})
    px2 = ("PASS" if v2 and v2["med_max"] >= 0.6 else
           "KILL (ledger not portable)" if v2 and v2["med_max"] < 0.3
           else "GRAY")
    print(f"\nP-X1 (t=0 donor adoption): {'PASS' if px1 else 'FAIL'}  {v1}")
    print(f"P-X2 (decision survives, t=500): {px2}  med_max {v2.get('med_max')}")
    print("P-X3: the curve above IS the result (branch pre-declared)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"gate": gate, "gate_ok": gate_ok,
                               "curve": curve, "hybrids": rows,
                               "controls": ctrl}, indent=1, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
