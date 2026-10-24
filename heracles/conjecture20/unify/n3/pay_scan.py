"""Numerical scan of single-feature payment constants at n=3 (nu=4).

Atoms beta_i in C, xi_i in T^3.  A = sum beta_i, beta_T = sum_i beta_i prod_{t in T} xi_it,
Q = sum |beta_i|^r, r = 1/s, bonus = |A| - 4^{1-s} Q^s.
  cL2(s)  = sup bonus / sqrt(sum_T |beta_T|^2)          (L2 currency, 7 sets T)
  c11(s)  = sup bonus / max_T |beta_T|                     (pointwise single-T currency)
Local search with random restarts (evidence only).
usage: python pay_scan.py <which: L2|max> <s> <Nmax> <restarts> <seed>
"""
import sys, itertools
import numpy as np
from scipy.optimize import minimize

n = 3
TS = [T for k in range(1, n + 1) for T in itertools.combinations(range(n), k)]
MASK = np.array([[1 if t in T else 0 for t in range(n)] for T in TS], dtype=float)  # 7 x 3


def unpack(x, N):
    br = x[:N]; bi = x[N:2 * N]; ang = x[2 * N:].reshape(N, n)
    return br + 1j * bi, ang


def quantities(x, N, s):
    beta, ang = unpack(x, N)
    A = beta.sum()
    phase = np.exp(1j * (ang @ MASK.T))       # N x 7
    bT = (beta[:, None] * phase).sum(axis=0)  # 7
    Q = np.sum(np.abs(beta) ** (1.0 / s))
    bon = np.abs(A) - 4.0 ** (1 - s) * Q ** s
    return bon, bT, A, Q


def obj_L2(x, N, s):
    bon, bT, A, Q = quantities(x, N, s)
    R = np.sqrt(np.sum(np.abs(bT) ** 2))
    return -bon / max(R, 1e-12)


def obj_max(x, N, s, k=60.0):
    bon, bT, A, Q = quantities(x, N, s)
    m = np.abs(bT)
    M = m.max()
    sm = M + np.log(np.sum(np.exp(k * (m - M) / max(M, 1e-12)))) * M / k
    return -bon / max(sm, 1e-12)


def exact_ratio(x, N, s, which):
    bon, bT, A, Q = quantities(x, N, s)
    if which == 'L2':
        return bon / np.sqrt(np.sum(np.abs(bT) ** 2))
    return bon / np.abs(bT).max()


def run(which, s, Nmax, restarts, seed):
    rng = np.random.default_rng(seed)
    best = (-np.inf, None, None)
    for N in range(1, Nmax + 1):
        for rr in range(restarts):
            x0 = np.concatenate([rng.normal(size=2 * N), rng.uniform(0, 2 * np.pi, size=N * n)])
            x0[:N] += 1.0
            f = obj_L2 if which == 'L2' else obj_max
            try:
                res = minimize(f, x0, args=(N, s), method='L-BFGS-B', options=dict(maxiter=3000))
                if which != 'L2':
                    for k in (120.0, 400.0, 2000.0):
                        res = minimize(obj_max, res.x, args=(N, s, k), method='Nelder-Mead',
                                       options=dict(maxiter=4000, xatol=1e-10, fatol=1e-12))
                val = exact_ratio(res.x, N, s, which)
            except Exception as ex:
                continue
            if val > best[0]:
                best = (val, N, res.x)
    return best


if __name__ == '__main__':
    which = sys.argv[1]; s = float(sys.argv[2]); Nmax = int(sys.argv[3]); restarts = int(sys.argv[4])
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    val, N, x = run(which, s, Nmax, restarts, seed)
    beta, ang = unpack(x, N)
    bon, bT, A, Q = quantities(x, N, s)
    sc = 1.0 / np.abs(A)
    print(f"which={which} s={s} best={val:.6f} N={N} single-atom L2={(1-4**(1-s))/np.sqrt(7):.6f} single-atom max={(1-4**(1-s)):.6f}")
    rot = np.conj(A) / np.abs(A)
    print("  |beta_i| (A=1):", np.round(np.abs(beta) * sc, 5))
    print("  arg beta_i (rel A):", np.round(np.angle(beta * rot), 5))
    print("  angles xi:", np.round(np.mod(ang, 2 * np.pi), 5).tolist())
    print("  |beta_T| (A=1):", dict(zip([''.join(str(t + 1) for t in T) for T in TS], np.round(np.abs(bT) * sc, 5))))
    print("  bonus/|A| =", bon * sc, " 4^{1-s}Q^s/|A| =", 4 ** (1 - s) * Q ** s * sc)
