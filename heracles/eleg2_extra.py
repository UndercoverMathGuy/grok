"""eleg2_extra.py -- Theorem B's ingredients, the k=1 rung, and R1 with good starts.

E1  Theorem B (n = 1, every N):  S >= hmin ;  S - max_l x_l >= hmin/2 ;  e_2 >= (hmin/2)^2 ;
    and the conclusion sum|c_l|^r >= 2^{1-r} hmin^r.
E2  The k = 1 rung for general n:  is  hmin <= max_T |C_T| ?  (would give e_1 >= hmin, since
    every |C_T| <= sum_l |c_l|).  Probed randomly and adversarially.
E3  R1 (the determinant route) at n = 2, 3 with BLOCK-based starts, so that m > 0 is actually
    reached -- the random starts of eleg2_routes.py mostly return m <= 0 there.

Run: uv run --with scipy python eleg2_extra.py
"""
import json
import os
import sys
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eleg2_core import rand_pop, min_re_h, chain_matrix, CT, elem_sym
from eleg2_thmA import min_abs_h

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_RUNS", "ELEG2")
os.makedirs(OUT, exist_ok=True)
rec = {}
RS = [0.05, 0.25, 0.5, 0.75, 0.95, 1.0]


# ------------------------------------------------------------------ E1
def e1(trials=400):
    rng = np.random.default_rng(21)
    w = dict(rung1=np.inf, peel=np.inf, rung2=np.inf, thm=np.inf)
    for N in [2, 3, 4, 5, 8, 12]:
        for t in range(trials):
            c, v = rand_pop(rng, N, 1, "generic")
            c = c / max(np.abs(c).max(), 1e-30)
            x = np.abs(c); S = x.sum()
            A = c.sum(); C1 = (c * v[:, 0]).sum()
            hmin = abs(abs(A) - abs(C1))
            if hmin < 1e-12:
                continue
            w["rung1"] = min(w["rung1"], S / hmin)
            w["peel"] = min(w["peel"], (S - x.max()) / (hmin / 2))
            e = elem_sym(x, kmax=2)
            w["rung2"] = min(w["rung2"], e[2] / (hmin / 2) ** 2)
            for r in RS:
                w["thm"] = min(w["thm"], float(np.sum(x ** r) / (2 ** (1 - r) * hmin ** r)))
    return {k: float(v) for k, v in w.items()}


# ------------------------------------------------------------------ E2
def e2(nmax=4, trials=250):
    rng = np.random.default_rng(22)
    out = {}
    for n in range(1, nmax + 1):
        worst_max = np.inf      # hmin <= max_T |C_T|  ->  ratio max|C_T| / hmin >= 1
        worst_S = np.inf        # hmin <= S
        for N in [n, n + 1, n + 2, 2 * (n + 1)]:
            for kind in ["generic", "junkfree", "nearjf"]:
                for t in range(trials):
                    c, v = rand_pop(rng, N, n, kind)
                    c = c / max(np.abs(c).max(), 1e-30)
                    hm = min_abs_h(c, v, starts=14, rng=rng)
                    if hm < 1e-10:
                        continue
                    Cv = np.abs(CT(c, v))
                    worst_max = min(worst_max, float(Cv.max() / hm))
                    worst_S = min(worst_S, float(np.abs(c).sum() / hm))
        out[n] = dict(min_maxCT_over_hmin=worst_max, min_S_over_hmin=worst_S)
        print(f"  E2 n={n}: min max_T|C_T|/hmin = {worst_max:.9f}   min S/hmin = {worst_S:.9f}")
    return out


# ------------------------------------------------------------------ E3
def e3(ns=(2, 3), Ns=(3, 4, 5, 6), restarts=12):
    rng = np.random.default_rng(23)
    out = {}
    for n in ns:
        D = n + 1
        for N in Ns:
            dim = 2 * N + N * n

            def f(x, starts=10):
                c = np.exp(x[:N]) * np.exp(1j * x[N:2 * N])
                v = np.exp(1j * x[2 * N:].reshape(N, n))
                m, _ = min_re_h(c, v, 1.0, starts=starts, rng=rng)
                if m <= 1e-9:
                    return 1e6
                return float(abs(np.linalg.det(chain_matrix(c, v))) / m ** D)
            best = np.inf
            for t in range(restarts):
                g = -np.log(D) * np.ones(N)
                if N > D:
                    g[D:] = np.log(10.0 ** rng.uniform(-3, -0.5, size=N - D))
                th = np.zeros(N)
                bb = np.zeros((N, n))
                for l in range(min(N, D)):
                    bb[l, :] = -2 * np.pi * l / D
                if N > D:
                    bb[D:] = 2 * np.pi * rng.random((N - D, n))
                x0 = np.concatenate([g, th, bb.ravel()])
                x0 = x0 + (0.05 + 0.7 * rng.random()) * rng.normal(size=dim)
                res = minimize(f, x0, method="Nelder-Mead",
                               options=dict(maxiter=400 + 60 * dim, maxfev=400 + 60 * dim,
                                            xatol=1e-9, fatol=1e-12))
                val = f(res.x, starts=40)
                best = min(best, val)
            out[f"n{n}_N{N}"] = float(best)
            print(f"  E3 n={n} N={N}: min |det M|/m^D = {best:.9f}")
    return out


if __name__ == "__main__":
    print("E1  Theorem B ingredients (n = 1)")
    rec["E1"] = e1()
    print("   ", rec["E1"])
    print("E2  the k = 1 rung")
    rec["E2"] = e2()
    print("E3  determinant route with block starts")
    rec["E3"] = e3()
    with open(os.path.join(OUT, "extra.json"), "w") as f:
        json.dump(rec, f, indent=1)
