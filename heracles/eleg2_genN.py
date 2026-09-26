"""eleg2_genN.py -- the general-N conjecture and the aiming-depth thresholds.

CONJECTURE C1 (general N).   sum_l |c_l|^r >= (n+1)^{1-r} hmin^r,  hmin = min_{T^n} |h|.
CONJECTURE C2 (the ladder).  e_k(|c|) >= binom(D,k) (hmin/D)^k,  k = 1..D = n+1.
                             C2 => (Bernstein) => C1, uniformly in N and r.

DEPTH.  For a network on Z_p^n a probe can only aim to depth A = A*(p,S) < 1, so the object
that actually enters (P3C Theorem P3C-1, singleton partition) is
    m_A := min over |u_j| = A of Re h(u),     R(pop, A) = (n+1)^{(1-r)/r} (m_A)_+ / cost^{1/r}.
This script measures the threshold  A_1(n, r, N) := inf { A : R(pop,A) <= 1 for all N-atom pops }.

Run:  uv run --with scipy python eleg2_genN.py --mode genN
      uv run --with scipy python eleg2_genN.py --mode depth
"""
import argparse
import json
import os
import sys
import numpy as np
from math import comb
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eleg2_core import min_re_h, elem_sym, rand_pop

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_RUNS", "ELEG2")
os.makedirs(OUT, exist_ok=True)


def unpack(x, N, n):
    c = np.exp(x[:N]) * np.exp(1j * x[N:2 * N])
    v = np.exp(1j * x[2 * N:].reshape(N, n))
    return c, v


def mk_obj(N, n, r, A, which, starts, rng):
    D = n + 1

    def f(x):
        c, v = unpack(x, N, n)
        mm, _ = min_re_h(c, v, A, starts=starts, rng=rng)
        if mm <= 1e-12:
            return 1e6
        ac = np.abs(c)
        if which == "C1":
            return float(np.sum(ac ** r) / (D ** (1 - r) * mm ** r))
        e = elem_sym(ac, kmax=D)
        return float(min(e[k] / (comb(D, k) * (mm / D) ** k) for k in range(1, D + 1)))
    return f


def descend(N, n, r, A, which, restarts, starts, seed, polish_starts=60):
    rng = np.random.default_rng(seed)
    f = mk_obj(N, n, r, A, which, starts, rng)
    fp = mk_obj(N, n, r, A, which, polish_starts, rng)
    D = n + 1
    dim = 2 * N + N * n
    best, bx = np.inf, None
    for t in range(restarts):
        if t % 3 == 0:                       # perturbed block (+ dust atoms)
            g = -np.log(D) * np.ones(N)
            if N > D:
                g[D:] = np.log(10.0 ** rng.uniform(-4, -0.7, size=N - D))
            th = np.zeros(N)
            bb = np.zeros((N, n))
            for l in range(min(N, D)):
                bb[l, :] = -2 * np.pi * l / D
            if N > D:
                bb[D:] = 2 * np.pi * rng.random((N - D, n))
            x0 = np.concatenate([g, th, bb.ravel()])
            x0 += (0.03 + 0.5 * rng.random()) * rng.normal(size=dim)
        elif t % 3 == 1:                     # junk-free random solve
            c0, v0 = rand_pop(rng, N, n, "junkfree")
            c0 = c0 / max(np.abs(c0).max(), 1e-12)
            x0 = np.concatenate([np.log(np.abs(c0) + 1e-9), np.angle(c0),
                                 np.angle(v0).ravel()])
            x0 += 0.25 * rng.normal(size=dim)
        else:                                # free random
            x0 = np.concatenate([0.4 * rng.normal(size=N), 2 * np.pi * rng.random(N),
                                 2 * np.pi * rng.random(N * n)])
        res = minimize(f, x0, method="Nelder-Mead",
                       options=dict(maxiter=500 + 110 * dim, maxfev=500 + 110 * dim,
                                    xatol=1e-9, fatol=1e-12))
        val = fp(res.x)                       # re-evaluate with a careful inner solve
        if val < best:
            best, bx = float(val), res.x.copy()
    c, v = unpack(bx, N, n)
    return best, c, v


def mode_genN(a):
    out = []
    for n in a.ns:
        D = n + 1
        for N in a.Ns:
            if N < 1:
                continue
            for which in a.which:
                for r in (a.rs if which == "C1" else [None]):
                    b, c, v = descend(N, n, r or 0.5, 1.0, which, a.restarts,
                                      a.starts, seed=97 * n + 13 * N + int(100 * (r or 0)))
                    ac = np.abs(c) / np.abs(c).max()
                    supp = int(np.sum(ac > 1e-3))
                    out.append(dict(n=n, N=N, r=r, which=which, best=b,
                                    support=supp, absc=ac.round(6).tolist()))
                    print(f"n={n} D={D} N={N} {which} r={r}: min = {b:.12f}  "
                          f"support={supp}  |c|/max={np.round(ac,4).tolist()}")
    with open(os.path.join(OUT, "genN.json"), "w") as f:
        json.dump(out, f, indent=1)


def mode_depth(a):
    """Bisect the singleton-partition aiming threshold A_1(n,r,N)."""
    out = []
    for n in a.ns:
        for r in a.rs:
            for N in a.Ns:
                lo, hi = 0.05, 1.0
                for _ in range(a.bisect):
                    mid = 0.5 * (lo + hi)
                    b, c, v = descend(N, n, r, mid, "C1", a.restarts, a.starts,
                                      seed=7 * n + N + int(1000 * mid))
                    if b >= 1.0 - 1e-7:
                        hi = mid
                    else:
                        lo = mid
                out.append(dict(n=n, r=r, N=N, A_lo=lo, A_hi=hi))
                print(f"n={n} r={r:.4f} N={N}: singleton threshold A_1 in [{lo:.4f}, {hi:.4f}]")
    with open(os.path.join(OUT, "depth.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="genN")
    ap.add_argument("--ns", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--Ns", type=int, nargs="+", default=[3, 4, 5, 6])
    ap.add_argument("--rs", type=float, nargs="+", default=[1/3, 2/3])
    ap.add_argument("--which", nargs="+", default=["C1", "C2"])
    ap.add_argument("--restarts", type=int, default=9)
    ap.add_argument("--starts", type=int, default=6)
    ap.add_argument("--bisect", type=int, default=7)
    a = ap.parse_args()
    (mode_genN if a.mode == "genN" else mode_depth)(a)
