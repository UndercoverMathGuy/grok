"""P1B: independent numerical evidence and structural tests around P1.

Blocks (choose with --blocks):
  C   the first open case n = 2, N = 4 as a global search over the 6 gauge-fixed angles
      (the square case: the rule z is the unique solution of a 4x4 system).
  H   the sign ("Hadamard") configuration v_lj = +-1 with N = 2^n atoms: an EXACT second
      family of multi-affine cubature rules, weights 2^-n.  It refutes the product
      conjecture P1-prod (see below) in closed form.
  G   the gap window: for the cells where the two-branch proof does NOT close, search
      configurations whose tail E lies in the uncovered window and test P1 there.
  R   does the multiplicative structure matter?  Replace the constraint rows
      (e, a, b, a*b) by (e, a, b, d) with d an unrelated unimodular vector and see whether
      the constant is still 3^{1-r}.

P1-prod (REFUTED here): "some n+1 atoms have prod |c_l| >= (|A|/(n+1))^{n+1}".  It would imply
P1 in one line via AM-GM.  The sign configuration at n = 2 has all |c_l| = 1/4, so every triple
has product 1/64 < 1/27 = (1/3)^3.
"""

import argparse
import json
import os

import numpy as np
from scipy.optimize import minimize

import p1b_core as C

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_RUNS", "P1B")


# ------------------------------------------------------------------ block C

def ratio_angles(theta, n, N, r):
    v = C.angles_to_v(theta, N, n)
    try:
        z = C.rule_from_config(v)
    except np.linalg.LinAlgError:
        return 1e6
    if not np.all(np.isfinite(z)):
        return 1e6
    return float(np.sum(np.abs(z) ** r)) / (n + 1) ** (1.0 - r)


def global_search(n, r, starts=600, seed=0):
    """n = 2, N = 4 (square).  Nelder-Mead + Powell from random and structured starts."""
    N = 1 << n
    rng = np.random.default_rng(seed)
    d = C.n_free_angles(N, n)
    best, bestth = np.inf, None
    vb = C.block(n)
    for i in range(starts):
        if i < 12:  # block + one free atom, jittered
            th = []
            for l in range(1, n + 1):
                th += [float(np.angle(vb[l, j])) for j in range(n)]
            th += list(rng.uniform(0, 2 * np.pi, n))
            th = np.array(th) + rng.normal(0, 0.02 * i, d)
        else:
            th = rng.uniform(0, 2 * np.pi, d)
        res = minimize(ratio_angles, th, args=(n, N, r), method="Nelder-Mead",
                       options=dict(maxiter=20000, maxfev=20000, xatol=1e-13, fatol=1e-15))
        res = minimize(ratio_angles, res.x, args=(n, N, r), method="Powell",
                       options=dict(maxiter=20000, xtol=1e-13, ftol=1e-15))
        if res.fun < best:
            best, bestth = float(res.fun), res.x.copy()
    v = C.angles_to_v(bestth, N, n)
    z = C.rule_from_config(v)
    return dict(n=n, N=N, r=r, min_ratio=best,
                moduli=sorted(np.abs(z).tolist(), reverse=True))


# ------------------------------------------------------------------ block H

def sign_family(nmax=5):
    rows = []
    for n in range(1, nmax + 1):
        v = C.hadamard_config(n)
        z = C.rule_from_config(v)
        N = 1 << n
        mod = np.abs(z)
        best_prod = max(np.prod(np.sort(mod)[::-1][:n + 1]), 0.0)
        rows.append(dict(n=n, N=N, max_dev_from_2mn=float(np.max(np.abs(mod - 2.0 ** -n))),
                         cost_over_target_r2over3=float(np.sum(mod ** (2 / 3.)) / (n + 1) ** (1 / 3.)),
                         best_top_np1_product=float(best_prod),
                         block_product=float((n + 1) ** -(n + 1)),
                         P1prod_holds=bool(best_prod >= (n + 1) ** -(n + 1) - 1e-12)))
    return rows


# ------------------------------------------------------------------ block G

def gap_window_search(n, q, e0, e1, trials=4000, seed=0):
    """Sample (KL) systems with N = 2^n by perturbing the block, keep those whose tail E
    lands in the uncovered window (e0, e1), and report the worst cost/target seen."""
    r = q / (n + 1)
    rng = np.random.default_rng(seed)
    N = 1 << n
    tgt = (n + 1) ** (1.0 - r)
    worst, worstE, hits = np.inf, None, 0
    allworst = np.inf
    for _ in range(trials):
        s = 10 ** rng.uniform(-3.0, -0.3)
        vb = C.block(n) * np.exp(1j * rng.normal(0, s, (n + 1, n)))
        extra = np.exp(1j * rng.uniform(0, 2 * np.pi, (N - n - 1, n)))
        v = np.vstack([vb, extra])
        try:
            z = C.rule_from_config(v)
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite(z)):
            continue
        x = np.sort(np.abs(z))[::-1]
        E = float(np.sum(x[n + 1:]))
        ratio = float(np.sum(x ** r)) / tgt
        allworst = min(allworst, ratio)
        if e0 < E < e1:
            hits += 1
            if ratio < worst:
                worst, worstE = ratio, E
    return dict(n=n, q=q, r=r, window=[e0, e1], samples=trials, in_window=hits,
                worst_ratio_in_window=worst, worst_E=worstE, worst_ratio_overall=allworst)


# ------------------------------------------------------------------ block R

def general_rows(n=2, trials=20000, r=2.0 / 3.0, seed=0):
    """Rows (e, a, b, d): a, b, d unimodular, d NOT forced to equal a*b.  If the minimum of
    sum|c|^r drops below 3^{1-r}, the multiplicative (multi-affine) structure is essential."""
    rng = np.random.default_rng(seed)
    worst_prod_struct = np.inf
    worst_free = np.inf
    for _ in range(trials):
        a = np.exp(1j * rng.uniform(0, 2 * np.pi, 4))
        b = np.exp(1j * rng.uniform(0, 2 * np.pi, 4))
        d = np.exp(1j * rng.uniform(0, 2 * np.pi, 4))
        for rows, key in ((np.array([np.ones(4), a, b, a * b]), "struct"),
                          (np.array([np.ones(4), a, b, d]), "free")):
            rhs = np.zeros(4, dtype=complex); rhs[0] = 1.0
            try:
                z = np.linalg.solve(rows, rhs)
            except np.linalg.LinAlgError:
                continue
            if not np.all(np.isfinite(z)):
                continue
            val = float(np.sum(np.abs(z) ** r)) / 3 ** (1 - r)
            if key == "struct":
                worst_prod_struct = min(worst_prod_struct, val)
            else:
                worst_free = min(worst_free, val)
    return dict(trials=trials, r=r, min_ratio_multiaffine=worst_prod_struct,
                min_ratio_free_third_row=worst_free)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default="CHGR")
    ap.add_argument("--starts", type=int, default=400)
    ap.add_argument("--trials", type=int, default=4000)
    ap.add_argument("--tag", default="search")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    out = {}

    if "C" in args.blocks:
        out["C"] = [global_search(2, r, starts=args.starts, seed=7) for r in (2 / 3., 0.5, 5 / 6.)]
        for row in out["C"]:
            print("C  n=2 N=4 r=%.4f  min cost/target = %.12f   moduli %s"
                  % (row["r"], row["min_ratio"], np.round(row["moduli"], 8)))

    if "H" in args.blocks:
        out["H"] = sign_family(5)
        for row in out["H"]:
            print("H  n=%d N=%d  |z_l| all 2^-n (dev %.2e)  cost/target(r=2/3)=%.9f  "
                  "best top-(n+1) product %.6g vs block %.6g -> P1-prod %s"
                  % (row["n"], row["N"], row["max_dev_from_2mn"],
                     row["cost_over_target_r2over3"], row["best_top_np1_product"],
                     row["block_product"], "holds" if row["P1prod_holds"] else "FAILS"))

    if "G" in args.blocks:
        out["G"] = []
        for (n, q, e0, e1) in [(5, 2.0, 4.437e-3, 1.754e-2), (3, 3.0, 1.928e-4, 7.111e-3),
                               (4, 3.0, 6.965e-4, 3.703e-2)]:
            row = gap_window_search(n, q, e0, e1, trials=args.trials)
            out["G"].append(row)
            print("G  n=%d q=%.1f window (%.3g, %.3g): %d/%d samples in window, "
                  "worst cost/target in window = %s (overall %.9f)"
                  % (n, q, e0, e1, row["in_window"], row["samples"],
                     ("%.9f" % row["worst_ratio_in_window"]) if row["in_window"] else "-",
                     row["worst_ratio_overall"]))

    if "R" in args.blocks:
        out["R"] = general_rows(trials=args.trials * 5)
        print("R  multi-affine rows (e,a,b,ab): min ratio %.9f ; free third row (e,a,b,d): %.9f"
              % (out["R"]["min_ratio_multiaffine"], out["R"]["min_ratio_free_third_row"]))

    with open(os.path.join(OUT, args.tag + ".json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
