"""eleg2_adv.py -- adversarial attack on the ELEGANT2 conjectures.

Minimise, over populations (c, v),

    ratio_C1(r) = sum_l |c_l|^r  /  [ (n+1)^{1-r} * m_+^r ] ,     m = min_{|u_j|=A} Re h(u)

(and the same for the symmetric ladder C2 and the determinant form C3).  A value
below 1 refutes; the block is the known equality case, so the target is 1.

Parametrisation: |c_l| = exp(g_l), arg c_l = t_l, arg v_lj = b_lj.  Global scale and
the three gauges (global phase of c, per-slot phase rotation of v) are left free --
they only add flat directions.

Run:  uv run python eleg2_adv.py --n 2 --Ns 3 4 5 6 --A 1.0
"""
import argparse
import json
import os
import sys
import numpy as np
from math import comb
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eleg2_core import min_re_h, chain_matrix, elem_sym, rand_pop

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_RUNS", "ELEG2")
os.makedirs(OUT, exist_ok=True)


def unpack(x, N, n):
    g = x[:N]
    t = x[N:2 * N]
    b = x[2 * N:].reshape(N, n)
    c = np.exp(g) * np.exp(1j * t)
    v = np.exp(1j * b)
    return c, v


def ratio_C1(x, N, n, r, A, starts, rng):
    c, v = unpack(x, N, n)
    D = n + 1
    m, _ = min_re_h(c, v, A, starts=starts, rng=rng)
    if m <= 1e-14:
        return 1e6
    return float(np.sum(np.abs(c) ** r) / (D ** (1 - r) * m ** r))


def ratio_C2(x, N, n, A, starts, rng):
    c, v = unpack(x, N, n)
    D = n + 1
    m, _ = min_re_h(c, v, A, starts=starts, rng=rng)
    if m <= 1e-14:
        return 1e6
    e = elem_sym(np.abs(c), kmax=D)
    return float(min(e[k] / (comb(D, k) * (m / D) ** k) for k in range(1, D + 1)))


def ratio_C3(x, N, n, A, starts, rng, SS=(0.1, 0.3, 1.0, 3.0, 10.0, 100.0)):
    c, v = unpack(x, N, n)
    D = n + 1
    m, _ = min_re_h(c, v, A, starts=starts, rng=rng)
    if m <= 1e-14:
        return 1e6
    M = chain_matrix(c, v)
    return float(min(abs(np.linalg.det(np.eye(D) + s * M)) / (1 + s * m) ** D for s in SS))


def block_start(N, n, rng, jitter):
    D = n + 1
    g = np.zeros(N) - np.log(D)
    t = np.zeros(N)
    b = np.zeros((N, n))
    for l in range(min(N, D)):
        b[l, :] = -2 * np.pi * l / D
    if N > D:
        g[D:] = np.log(1e-3)
        b[D:] = 2 * np.pi * rng.random((N - D, n))
    x = np.concatenate([g, t, b.ravel()])
    return x + jitter * rng.normal(size=x.size)


def attack(n, N, r, A, which, restarts, starts, seed):
    rng = np.random.default_rng(seed)
    fun = {"C1": lambda x: ratio_C1(x, N, n, r, A, starts, rng),
           "C2": lambda x: ratio_C2(x, N, n, A, starts, rng),
           "C3": lambda x: ratio_C3(x, N, n, A, starts, rng)}[which]
    best, bestx = np.inf, None
    dim = 2 * N + N * n
    for t in range(restarts):
        if t % 3 == 0:
            x0 = block_start(N, n, rng, jitter=0.05 + 0.5 * rng.random())
        elif t % 3 == 1:
            c0, v0 = rand_pop(rng, N, n, "junkfree")
            c0 = c0 / np.abs(c0).max()
            x0 = np.concatenate([np.log(np.abs(c0) + 1e-9), np.angle(c0),
                                 np.angle(v0).ravel()])
            x0 = x0 + 0.3 * rng.normal(size=dim)
        else:
            x0 = np.concatenate([rng.normal(size=N) * 0.5, 2 * np.pi * rng.random(N),
                                 2 * np.pi * rng.random(N * n)])
        res = minimize(fun, x0, method="Nelder-Mead",
                       options=dict(maxiter=4000 + 400 * dim, xatol=1e-10, fatol=1e-12))
        if res.fun < best:
            best, bestx = float(res.fun), res.x.copy()
    return best, bestx


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--Ns", type=int, nargs="+", default=[3, 4, 5])
    ap.add_argument("--rs", type=float, nargs="+", default=[1/3, 2/3])
    ap.add_argument("--A", type=float, default=1.0)
    ap.add_argument("--which", nargs="+", default=["C1"])
    ap.add_argument("--restarts", type=int, default=12)
    ap.add_argument("--starts", type=int, default=16)
    ap.add_argument("--tag", default="adv")
    a = ap.parse_args()
    out = []
    for N in a.Ns:
        for which in a.which:
            rs = a.rs if which == "C1" else [None]
            for r in rs:
                b, bx = attack(a.n, N, r, a.A, which, a.restarts, a.starts,
                               seed=1000 * N + int(1000 * (r or 0)))
                c, v = unpack(bx, N, a.n)
                rec = dict(n=a.n, N=N, r=r, A=a.A, which=which, best=b,
                           absc=np.abs(c).tolist(),
                           argv=(np.angle(v) / np.pi).round(4).tolist())
                out.append(rec)
                print(f"n={a.n} N={N} A={a.A} {which} r={r}: min ratio = {b:.12f}")
                print(f"      |c| = {np.round(np.abs(c)/np.abs(c).max(),4).tolist()}")
    with open(os.path.join(OUT, f"{a.tag}_n{a.n}_A{a.A}.json"), "w") as f:
        json.dump(out, f, indent=1)
