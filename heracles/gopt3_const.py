"""gopt3_const.py -- the constants of the peeling theorem, to full fp64 precision, and the final table.

    S_d      = max_{|z|=1} | 1 - ((1+z)/2)^d |            (d = 1..6; S_1 = 1, S_2 = 2/sqrt3, S_3 = 9/7)
    kappa_d  = (1 - 2^{-d}) / S_d                         (value of the d-slot kernel kill polynomial)
    D_h(n)   = max over compositions d_1+..+d_h = n of prod_i kappa_{d_i}      (h = 0..n, D_0 = 1)
    L_r(n)   = min { sum_l x_l^r : x_1 >= x_2 >= ... >= 0, sum_{l>h} x_l >= D_h(n), h = 0..n }
    factor   = ( (n+1)^{1-r} / L_r(n) )^{1/r}             (the factor by which (T) is missed)

The one-variable reduction of S_d is proved in gopt3_peel.py's docstring (all tan(psi_j) equal at any
critical point, and flipping psi_j -> psi_j + pi leaves prod_j (1+z_j)/2 unchanged).

Usage: uv run python heracles/gopt3_const.py --out heracles/FINAL_RUNS/GOPT3
"""
import argparse, json, math, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(ROOT))
from gopt3_peel import L_r, L_r_vertex, peel_numbers_kernel


def S_exact(d, G=4000001):
    psi = math.pi * (2 * np.arange(G) / (G - 1) - 1)
    m = np.cos(psi) ** d
    f = 1 - 2 * m * np.cos(d * psi) + m ** 2
    i = int(np.argmax(f))
    lo, hi = psi[max(i - 2, 0)], psi[min(i + 2, G - 1)]
    g = lambda t: -(1 - 2 * np.cos(t) ** d * np.cos(d * t) + np.cos(t) ** (2 * d))
    r = minimize_scalar(g, bounds=(lo, hi), method="bounded", options=dict(xatol=1e-15))
    return math.sqrt(max(float(f.max()), -float(r.fun)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(ROOT / "FINAL_RUNS" / "GOPT3"))
    a = ap.parse_args()
    out = dict(S={}, kappa={}, D={}, table=[])
    closed = {1: "1", 2: "2/sqrt(3)", 3: "9/7"}
    print("  d |        S_d        |      kappa_d      | closed form of S_d")
    K = {}
    for d in range(1, 7):
        s = S_exact(d); K[d] = (1 - 2.0 ** -d) / s
        out["S"][d] = s; out["kappa"][d] = K[d]
        print(f"  {d} | {s:.15f} | {K[d]:.15f} | {closed.get(d,'algebraic (root of a degree-d polynomial in cos^2 psi)')}")
    print(f"\n  check: 2/sqrt(3) = {2/math.sqrt(3):.15f},  9/7 = {9/7:.15f}, "
          f" kappa_2 = 3 sqrt(3)/8 = {3*math.sqrt(3)/8:.15f},  kappa_3 = 49/72 = {49/72:.15f}")

    print("\n  n  nu   q      r      D_1..D_n                         L_r(n)      (n+1)^(1-r)   ratio    FACTOR")
    for n in (2, 3, 4, 5):
        nu = n + 1
        D = peel_numbers_kernel(n)
        out["D"][n] = {h: D[h] for h in range(n + 1)}
        for q in (2.0, 2.5, 3.0, float(nu)):
            if q > nu:
                continue
            r = q / nu
            L, xv = L_r_vertex(n, D, r, Nmax=(8 if n < 4 else 7))     # exact: min of a concave
            Ls = min(L_r(n, D, r, starts=st, seed=sd)[0] for st, sd in ((60, 0), (120, 11)))  # function is at a vertex
            assert Ls > L - 1e-8, (n, q, L, Ls)
            tgt = nu ** (1 - r); fac = (tgt / L) ** (1 / r)
            out["table"].append(dict(n=n, nu=nu, q=q, r=r, L_r=L, L_r_slsqp=Ls, argmin=list(xv), target=tgt, ratio=L / tgt, factor=fac))
            print(f"  {n}  {nu}  {q:4g}  {r:.4f}  " + " ".join(f"{D[h]:.6f}" for h in range(1, n + 1)).ljust(32)
                  + f" {L:.8f}  {tgt:.8f}  {L/tgt:.6f}  {fac:.6f}")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "constants.json").write_text(json.dumps(out, indent=1, default=float))
    print("\nwrote", Path(a.out) / "constants.json")
