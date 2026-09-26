"""P2 proven minorants for Cl2_r, and the chord test of P3A.

WHAT HAS TO BE PROVED (P3A's reduction).  Write

    A(eta) := Cl2_r(eta)^{1/r}          ("certified amplitude per unit cost")
    kappa0 := (n+1)^{(1-r)/r} = A(0)    (Key Lemma / P1)
    etastar := sqrt(2^n - 1)            (largest possible junk, A(etastar) = 1)

P3A needs   1 - phi(eta)^{-1} <= rho_need * eta   with rho_need = (1 - 1/kappa0)/etastar,
which is exactly

    (CHORD)   A(eta)  >=  kappa0 - (kappa0 - 1) * eta / etastar     for all eta in [0, etastar].

i.e. A must lie above the straight line joining its two endpoints.

WHAT IS PROVED HERE.  For a population of N = n+1 atoms with C_empty = 1 and junk
energy sum_{T != empty}|C_T|^2 <= eta^2, put p_l = |c_l|, delta := sqrt(2^n-1) eta
(Cauchy-Schwarz: |g(u)| = |sum_{T!=0} C_T u^T| <= sqrt(2^n-1) eta everywhere, so
R := min_{T^n} Re h >= 1 - delta).  Then

    (C1)  sum_l p_l >= 1                              [C_empty = 1, triangle inequality]
    (C2)  prod_l p_l >= ((1-delta)/(n+1))^{n+1}       [Thm B (4.2) with R >= 1-delta]
    (C3)  p_l >= (1-delta)/2^n  for every l           [Thm B (4.1), chord d <= 2]

and the value of  min{ sum_l p_l^r : (C1),(C2),(C3) }  is a PROVEN minorant of Cl2_r
restricted to N = n+1.  This script computes it exactly (KKT / grid) and compares with
the chord.
"""
import argparse, json, os
import numpy as np
from scipy.optimize import minimize


def prog_value(n, r, delta, ngrid=0):
    """min sum_{l=1}^{n+1} p_l^r  s.t. sum p >= 1, prod p >= G^{n+1}, p_l >= m."""
    nb = n + 1
    G = max(1.0 - delta, 0.0) / nb
    m = max(1.0 - delta, 0.0) / (2 ** n)
    if G <= 0:
        # only (C1) survives: infimum is 1 (one atom of modulus 1)
        return 1.0
    logG = nb * np.log(G)

    def f(x):          # x = log p
        p = np.exp(x)
        return float(np.sum(p ** r)), r * np.exp(r * x)

    cons = [
        {"type": "ineq", "fun": lambda x: np.sum(np.exp(x)) - 1.0,
         "jac": lambda x: np.exp(x)},
        {"type": "ineq", "fun": lambda x: np.sum(x) - logG,
         "jac": lambda x: np.ones_like(x)},
    ]
    bnds = [(np.log(m) if m > 0 else -40.0, 5.0)] * nb
    best = np.inf
    rng = np.random.default_rng(0)
    starts = [np.full(nb, np.log(1.0 / nb))]
    for _ in range(60):
        starts.append(np.log(np.maximum(rng.dirichlet(np.full(nb, 0.6)), 1e-8)))
    for x0 in starts:
        try:
            res = minimize(f, x0, jac=True, bounds=bnds, constraints=cons,
                           method="SLSQP", options={"maxiter": 300, "ftol": 1e-14})
        except Exception:
            continue
        x = res.x
        p = np.exp(x)
        if np.sum(p) < 1 - 1e-9 or np.sum(x) < logG - 1e-9 or np.any(p < m - 1e-9):
            continue
        best = min(best, float(np.sum(p ** r)))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=["3:2", "3:2.5", "4:2", "4:3"])
    ap.add_argument("--npts", type=int, default=41)
    a = ap.parse_args()
    out = {}
    for cell in a.cells:
        nu, q = cell.split(":")
        nu, q = int(nu), float(q)
        n, r = nu - 1, q / nu
        k0 = (n + 1) ** ((1 - r) / r)
        es = np.sqrt(2 ** n - 1)
        rho_need = (1 - 1 / k0) / es
        rows = []
        for eta in np.linspace(0, es, a.npts):
            v = prog_value(n, r, es * eta)
            A = v ** (1.0 / r)
            chord = k0 - (k0 - 1) * eta / es
            rows.append({"eta": float(eta), "cl2": v, "A": A, "chord": float(chord),
                         "ok": bool(A >= chord - 1e-12)})
        sl = None
        e0 = es * 1e-4
        sl = (k0 - prog_value(n, r, es * e0) ** (1.0 / r)) / e0
        out[cell] = {"n": n, "r": r, "kappa0": k0, "etastar": float(es),
                     "rho_need": float(rho_need),
                     "slope_proved_A": float(sl),
                     "slope_needed_A": float((k0 - 1) / es),
                     "shortfall_factor": float(sl / ((k0 - 1) / es)),
                     "rows": rows}
        print(f"{cell}: kappa0={k0:.6f} etastar={es:.6f} rho_need={rho_need:.9f} "
              f"| proved slope {sl:.6f} vs needed {(k0-1)/es:.6f} "
              f"-> short by {sl/((k0-1)/es):.3f}x")
        nb = sum(1 for x in rows if not x["ok"])
        print(f"    chord satisfied at {len(rows)-nb}/{len(rows)} grid points")
    os.makedirs("FINAL_RUNS/P2", exist_ok=True)
    json.dump(out, open("FINAL_RUNS/P2/bound_prog.json", "w"), indent=1)
    print("wrote FINAL_RUNS/P2/bound_prog.json")


if __name__ == "__main__":
    main()
