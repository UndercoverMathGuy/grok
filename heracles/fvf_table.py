"""fvf_table.py -- collate the FVF controls against the archived constructed-start ladder and the programme.

Prints, per fixed-support arm, gamma at the last lam stage (mean over seeds), normalised to the arm's own maximum,
next to (i) the archived constructed-start ladder (FINAL_RUNS/NU4/fixed_prod_n2_p71_common) and (ii) the
free-amplitude programme Gamma_S (nu4_program_p71.json), also normalised to its own maximum.
Symbols: K = number of allowed frequencies; gamma = min margin / ||theta||^nu; N_.05 = frequencies holding the top
95% of the clean-logit energy; clean = 1 - junk share of the centred-logit energy."""
import json, sys
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parent
FVF = ROOT / "FINAL_RUNS" / "FVF"
NU4 = ROOT / "FINAL_RUNS" / "NU4"


def collect(d):
    rows = defaultdict(list)
    for f in sorted(d.glob("K*_seed*/metrics.json")):
        r = json.loads(f.read_text()); s = r["stages"][-1]
        rows[r["K"]].append((s["gamma"], s["clean_frac"], s["N_05"], r["m"]))
    return rows


def show(name, rows):
    if not rows:
        return
    ks = sorted(rows)
    best = max(np.mean([x[0] for x in rows[K]]) for K in ks)
    print(f"\n-- {name}  (best gamma {best:.4e})")
    print("   K | m   | gamma (mean)  | rel   | seeds spread      | clean | N_.05")
    for K in ks:
        g = [x[0] for x in rows[K]]
        print(f"  {K:2d} | {rows[K][0][3]:3d} | {np.mean(g):.4e} | {np.mean(g)/best:5.3f} | "
              f"[{min(g):.3e},{max(g):.3e}] | {np.mean([x[1] for x in rows[K]]):.3f} | "
              f"{','.join(str(x[2]) for x in rows[K])}")
    return {K: float(np.mean([x[0] for x in rows[K]])) for K in ks}


if __name__ == "__main__":
    out = {}
    for d in sorted(list(FVF.glob("*_prod_*")) + [NU4 / "fixed_prod_n2_p71_common", NU4 / "fixed_prod_n2_p71"]):
        if d.is_dir():
            out[d.name] = show(d.name, collect(d))
    prog = json.loads((ROOT / "nu4_program_p71.json").read_text())["3.0"]["ladder"]
    mx = max(v["free"] for v in prog.values())
    print("\n-- free-amplitude programme Gamma_S, p 71, nu 3 (prediction)")
    print("   K | Gamma_S  | rel")
    for K in sorted(prog, key=int):
        print(f"  {int(K):2d} | {prog[K]['free']:.5f} | {prog[K]['free']/mx:5.3f}")
    (FVF / "ladders.json").write_text(json.dumps(out, indent=1))
