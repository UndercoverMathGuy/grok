"""eleg2_det.py -- is the determinant form (C3) refuted only at n = 1?

(C3)   |det M| >= hmin^D ,  M = sum_l c_l m_l m_l^*,  D = n+1,  hmin = min_{T^n} |h| .
       (and more generally |det(I + sM)| >= (1 + s hmin)^D for s >= 0.)

At n = 1 the infimum of |det M|/hmin^D is 0 (eleg2_routes.py R1).  Here the same quantity is
attacked at n = 2, 3 from starts that are guaranteed to have hmin > 0 -- random JUNK-FREE
populations of N atoms (for which h == A, hmin = |A| > 0) plus perturbations -- so the search is
not biased toward the block the way eleg2_extra.py E3 is.

Run: uv run --with scipy python eleg2_det.py --ns 1 2 3 --Ns 3 4 5 6
"""
import argparse
import json
import os
import sys
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eleg2_core import rand_pop, chain_matrix
from eleg2_thmA import min_abs_h

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_RUNS", "ELEG2")
os.makedirs(OUT, exist_ok=True)


def obj(x, N, n, rng, starts, s=None):
    D = n + 1
    c = np.exp(x[:N]) * np.exp(1j * x[N:2 * N])
    v = np.exp(1j * x[2 * N:].reshape(N, n))
    hm = min_abs_h(c, v, starts=starts, rng=rng)
    if hm <= 1e-10:
        return 1e6
    M = chain_matrix(c, v)
    if s is None:
        return float(abs(np.linalg.det(M)) / hm ** D)
    return float(abs(np.linalg.det(np.eye(D) + s * M)) / (1 + s * hm) ** D)


def run(ns, Ns, restarts, starts):
    rng = np.random.default_rng(31)
    out = {}
    for n in ns:
        for N in Ns:
            if N < n + 1:
                continue
            dim = 2 * N + N * n
            best, bx = np.inf, None
            for t in range(restarts):
                c0, v0 = rand_pop(rng, N, n, "junkfree")
                sc = max(np.abs(c0).max(), 1e-12)
                c0 = c0 / sc
                x0 = np.concatenate([np.log(np.abs(c0) + 1e-9), np.angle(c0),
                                     np.angle(v0).ravel()])
                x0 = x0 + (0.02 + 0.8 * rng.random()) * rng.normal(size=dim)
                res = minimize(lambda z: obj(z, N, n, rng, starts), x0,
                               method="Nelder-Mead",
                               options=dict(maxiter=500 + 90 * dim, maxfev=500 + 90 * dim,
                                            xatol=1e-10, fatol=1e-13))
                val = obj(res.x, N, n, rng, 45)
                if val < best:
                    best, bx = val, res.x.copy()
            # also check the full s-family at the minimiser
            sfam = min(obj(bx, N, n, rng, 45, s=s)
                       for s in [0.05, 0.2, 1.0, 5.0, 50.0])
            out[f"n{n}_N{N}"] = dict(min_detM_over_hmin_D=float(best),
                                     min_over_s_family=float(sfam))
            print(f"  n={n} N={N}: min |det M|/hmin^D = {best:.9f}   "
                  f"min_s |det(I+sM)|/(1+s hmin)^D = {sfam:.9f}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--Ns", type=int, nargs="+", default=[3, 4, 5, 6])
    ap.add_argument("--restarts", type=int, default=14)
    ap.add_argument("--starts", type=int, default=8)
    a = ap.parse_args()
    r = run(a.ns, a.Ns, a.restarts, a.starts)
    with open(os.path.join(OUT, "det.json"), "w") as f:
        json.dump(r, f, indent=1)
