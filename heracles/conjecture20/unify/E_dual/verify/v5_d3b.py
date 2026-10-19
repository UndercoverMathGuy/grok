"""Referee check of D3b: nu^{-nu/2} ||E_q g||_sigma >= B_p Gamma_1 = (p/(p-1)) B_p for every q, with equality at
q uniform.  ALS (alternating maximisation, many starts) gives LOWER bounds on the spectral norm; for q uniform we
also compare with the analytic upper bound (p/(p-1)) B_p nu^{nu/2}."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlib import *
rng = np.random.default_rng(9)


def Gtensor(q, p, n):
    """tensor G[x_1..x_n, c] of the linear functional f -> sum_{x,delta} q(x,delta) g(x,delta)."""
    idx = np.indices((p,) * n).reshape(n, -1).T; y = idx.sum(1) % p
    G = np.zeros((p,) * (n + 1))
    for k, x in enumerate(idx):
        for d in range(1, p):
            G[tuple(x) + (y[k],)] += q[k, d - 1]; G[tuple(x) + ((y[k] - d) % p,)] -= q[k, d - 1]
    return G


def als(G, starts=30, iters=200):
    k = G.ndim; p = G.shape[0]; best = 0
    for st in range(starts):
        vs = [rng.standard_normal(p) for _ in range(k)]; vs = [v / np.linalg.norm(v) for v in vs]
        for it in range(iters):
            for a in range(k):
                T = G
                for b in reversed(range(k)):
                    if b != a: T = np.tensordot(T, vs[b], axes=([b], [0]))
                vs[a] = T / max(np.linalg.norm(T), 1e-300)
        T = G
        for b in reversed(range(k)): T = np.tensordot(T, vs[b], axes=([b], [0]))
        best = max(best, abs(float(T)))
    return best


for (p, n) in [(3, 2), (5, 2), (7, 2), (5, 3)]:
    nu = n + 1; N = p ** n
    target = p / (p - 1) * Bp(p, n)
    qu = np.full((N, p - 1), 1.0 / (N * (p - 1)))
    su = als(Gtensor(qu, p, n)) * nu ** (-nu / 2)
    ratios = []
    for t in range(8):
        q = rng.dirichlet(np.ones(N * (p - 1)) * rng.uniform(0.05, 2)).reshape(N, p - 1)
        ratios.append(als(Gtensor(q, p, n), starts=15) * nu ** (-nu / 2) / target)
    print('p=%d n=%d: q uniform ALS/(B_p Gamma_1) = %.8f (analytic upper bound 1) ; random q: min ALS lower bound/(B_p Gamma_1) = %.4f'
          % (p, n, su / target, min(ratios)), flush=True)
