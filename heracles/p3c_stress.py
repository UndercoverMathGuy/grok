"""P3C step 5: stress tests of the criterion (PFP) around its tight point.

The `(n+1)`-block gives R_P = 1 for EVERY partition and every A (its g_P is the
constant C_empty), so the criterion `sum_P lam_P R_P^r <= 1` is an EQUALITY there
and the whole question is whether the block is the maximiser.  This script
hammers that point:

  --block local   dense random perturbations of the block, at many radii, with
                  and without an extra small atom;
  --block dirs    the first-order picture: for a perturbation
                  c_l = (1+eps_l)/(n+1), v_lj = w^{-l} e^{i delta_lj}, the junk
                  C_T is first order while the cost gain is second order, so the
                  block is stable unless EVERY C_T vanishes to first order --
                  and the singleton partition sees every T.  Checked by
                  measuring the two orders separately.
  --block sweep   a fine bisection for A_lam(n,r) using the pool + LP of
                  p3c_part.py, seeded with the m-families and block perturbations.

  uv run python p3c_stress.py --cells nu3q2 nu4q2 --block local dirs sweep
"""
import argparse, json, os
import numpy as np
from scipy.optimize import linprog, minimize

import p3c_core as C
import p3c_part as PP

CELLS = PP.CELLS


def block_pop(n, N=None, rng=None, eps=0.0, extra=0.0):
    """The (n+1)-block, optionally perturbed and optionally with an extra atom."""
    nb = n + 1
    w = np.exp(2j * np.pi / nb)
    c = np.full(nb, 1.0 / nb, dtype=complex)
    th = np.array([[-2 * np.pi * l / nb] * n for l in range(nb)])
    if eps and rng is not None:
        c = c * (1 + eps * (rng.standard_normal(nb) + 1j * rng.standard_normal(nb)))
        th = th + eps * rng.standard_normal((nb, n))
    if extra and rng is not None:
        c = np.concatenate([c, [extra * (rng.standard_normal() +
                                         1j * rng.standard_normal())]])
        th = np.vstack([th, rng.uniform(0, 2 * np.pi, (1, n))])
    return c, np.exp(1j * th)


def lam_value(lam, R, r):
    return float(np.dot(lam, np.maximum(R, 0.0) ** r))


def block_local(out, n, r, A, lam, parts, rng, trials=4000):
    worst, wrec = -np.inf, None
    for _ in range(trials):
        e = 10.0 ** rng.uniform(-4, -0.3)
        ex = 0.0 if rng.random() < 0.5 else 10.0 ** rng.uniform(-4, -0.7)
        c, V = block_pop(n, rng=rng, eps=e, extra=ex)
        R = PP.R_all(c, V, A, parts, r, n, restarts=8, sweeps=40, rng=rng)
        v = lam_value(lam, R, r)
        if v > worst:
            worst, wrec = v, (e, ex, R.copy())
    out["local"] = {"worst": worst, "eps": wrec[0], "extra": wrec[1],
                    "R": wrec[2].tolist()}
    print(f"  block perturbations ({trials}): max sum_P lam_P R_P^r = {worst:.12f} "
          f"(eps={wrec[0]:.2e}, extra atom={wrec[1]:.2e})", flush=True)


def block_dirs(out, n, r, A, parts, rng, trials=600):
    """Separate the two orders: |C_T| (first order in the perturbation) against
    the cost surplus cost^{1/r}/kappa0 - 1 (second order)."""
    rows = []
    for _ in range(trials):
        t = 10.0 ** rng.uniform(-4, -1.0)
        c, V = block_pop(n, rng=rng, eps=t)
        Ct = C.coeffs(c, V)
        cl = abs(Ct[0])
        c = c / Ct[0]                                     # renormalise C_empty = 1
        Ct = C.coeffs(c, V)
        junk = float(np.max(np.abs(Ct[1:])))
        cost = float(np.sum(np.abs(c) ** r))
        sur = (cost ** (1 / r)) / C.kappa0(n, r) - 1.0
        rows.append((t, junk, sur))
    a = np.array(rows)
    # fit log|junk| and log|surplus| against log t
    s1 = np.polyfit(np.log(a[:, 0]), np.log(np.maximum(a[:, 1], 1e-300)), 1)[0]
    s2 = np.polyfit(np.log(a[:, 0]), np.log(np.maximum(np.abs(a[:, 2]), 1e-300)), 1)[0]
    out["dirs"] = {"slope_max_junk": float(s1), "slope_cost_surplus": float(s2)}
    print(f"  order check: d log(max_T |C_T|)/d log t = {s1:.4f} (expect 1), "
          f"d log|cost surplus|/d log t = {s2:.4f} (expect 2)", flush=True)


def sweep_A(n, r, parts, rng, lo, hi, npop=6000, starts=6, rounds=4, it=7):
    """Bisection on A for the LP value = 1."""
    hist = []
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        val, lam, pool, _ = PP.cutting_plane(n, r, mid, starts=starts, rounds=rounds,
                                             npop=npop, N_list=(1, 2, 3, 4, 5))
        hist.append((mid, val, lam.tolist()))
        print(f"   A={mid:.6f}: LP {val:.9f}", flush=True)
        if val <= 1.0:
            hi = mid
        else:
            lo = mid
    return lo, hi, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=["nu3q2"])
    ap.add_argument("--block", nargs="*", default=["local", "dirs"])
    ap.add_argument("--A", type=float, default=0.5)
    ap.add_argument("--lo", type=float, default=0.2)
    ap.add_argument("--hi", type=float, default=0.6)
    ap.add_argument("--trials", type=int, default=4000)
    ap.add_argument("--out", default="FINAL_RUNS/P3C/stress.json")
    a = ap.parse_args()
    res = {}
    for key in a.cells:
        nu, q = CELLS[key]
        n, r = nu - 1, q / nu
        parts = PP.partitions(n)
        rng = np.random.default_rng(99)
        cell = {"nu": nu, "q": q, "r": r, "n": n, "A": a.A}
        if "sweep" in a.block:
            lo, hi, hist = sweep_A(n, r, parts, rng, a.lo, a.hi)
            cell["A_lam_bracket"] = [lo, hi]
            cell["sweep"] = hist
            print(f"  A_lam({n},{r}) in [{lo:.6f}, {hi:.6f}]", flush=True)
        if "local" in a.block or "dirs" in a.block:
            # the lam used for the local tests: uniform unless a sweep gave one
            lam = np.full(len(parts), 1.0 / len(parts))
            if "sweep" in a.block and cell.get("sweep"):
                lam = np.array(cell["sweep"][-1][2])
            print(f"[{key}] A={a.A} lam={np.round(lam,4)}")
            if "local" in a.block:
                block_local(cell, n, r, a.A, lam, parts, rng, a.trials)
            if "dirs" in a.block:
                block_dirs(cell, n, r, a.A, parts, rng)
        res[key] = cell
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
