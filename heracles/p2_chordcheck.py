"""P2: does the MEASURED Cl2_r satisfy P3A's chord criterion, and where is rho maximal?

P3A's closed form rho_need = (1 - 1/kappa0)/sqrt(2^n-1) is the value of
rho(eta) = [1 - Cl2_r(eta)^{1/r}/kappa0]/eta at the endpoint eta = sqrt(2^n-1), and is the
right constant ONLY if rho attains its maximum there.  This script tests that on the
reference curves measured by p2_measure.py (an independent implementation from agent 4's
gopt4_relief.py).
"""
import glob, json, os
import numpy as np

out = {}
for f in sorted(glob.glob("FINAL_RUNS/P2/cost_nu*_q*.json")):
    d = json.load(open(f))
    nu, q, r = d["nu"], d["q"], d["r"]
    n = nu - 1
    k0 = (n + 1) ** ((1 - r) / r)
    es = np.sqrt(2 ** n - 1)
    rho_need = (1 - 1 / k0) / es
    rows = [x for x in d["rows"] if x["eta"] > 1e-9]
    cur = [(x["eta"], x["val"] ** (1 / r),
            (1 - x["val"] ** (1 / r) / k0) / x["eta"]) for x in rows]
    eta_m, A_m, rho_m = max(cur, key=lambda t: t[2])
    zero = [x for x in d["rows"] if x["eta"] <= 1e-9]
    key = f"nu{nu}_q{q}"
    out[key] = {"Cl2_0": zero[0]["val"] if zero else None,
                "target_Cl2_0": (n + 1) ** (1 - r),
                "rho_need": float(rho_need), "rho_sup_measured": float(rho_m),
                "argmax_eta": float(eta_m), "etastar": float(es),
                "chord_holds": bool(rho_m <= rho_need * (1 + 1e-9)),
                "n_levels": len(cur)}
    print(f"nu={nu} q={q}: Cl2_r(0)={out[key]['Cl2_0']:.9f} "
          f"(target {(n+1)**(1-r):.9f}); sup rho = {rho_m:.9f} at eta={eta_m:.4f} "
          f"(etastar={es:.4f}); rho_need={rho_need:.9f}; "
          f"chord {'HOLDS' if out[key]['chord_holds'] else 'FAILS'} "
          f"over {len(cur)} levels")
os.makedirs("FINAL_RUNS/P2", exist_ok=True)
json.dump(out, open("FINAL_RUNS/P2/chordcheck.json", "w"), indent=1)
print("wrote FINAL_RUNS/P2/chordcheck.json")
