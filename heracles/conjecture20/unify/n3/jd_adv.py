"""Adversarial check of the S_123 and S_k slicing bounds (and line slicings): minimise (AA' - bound)/E by local search.
A violation would be a value < 0.  usage: python jd_adv.py p restarts seed
"""
import sys
import numpy as np
from scipy.optimize import minimize
from jd_check import setup, check, TS, IX

p = int(sys.argv[1]); R = int(sys.argv[2]); seed = int(sys.argv[3])
S = setup(p)
rng = np.random.default_rng(seed)
d = 1


def build(x):
    f = x.reshape(7, p - 1)
    return np.concatenate([f, -f.sum(axis=1, keepdims=True)], axis=1)


def smooth_obj(x, key, k=60.0):
    D = build(x)
    Z = sum(D[a][S[:, a]] for a in range(7))
    Zd = sum(D[a][(S[:, a] + d) % p] for a in range(7))
    E = (D ** 2).mean(axis=1).sum()
    sc = np.sqrt(E)
    A = (-Z / sc).max() + np.log(np.sum(np.exp(k * ((-Z / sc) - (-Z / sc).max())))) / k
    Ap = (Zd / sc).max() + np.log(np.sum(np.exp(k * ((Zd / sc) - (Zd / sc).max())))) / k
    _, _, out = check(p, D, d, S)
    return A * Ap - out[key] / E


worst = {}
for key in ['S1', 'S123', 'L1', 'L12']:
    best = np.inf
    for r in range(R):
        x = rng.normal(size=7 * (p - 1)) * rng.exponential(1, size=7 * (p - 1))
        for k in (10.0, 40.0, 160.0):
            res = minimize(smooth_obj, x, args=(key, k), method='L-BFGS-B', options=dict(maxiter=1500))
            x = res.x
        D = build(x)
        prod, E, out = check(p, D, d, S)
        best = min(best, (prod - out[key]) / E)
    worst[key] = best
    print(f"p={p} bound {key}: min (AA' - bound)/E found = {best:.5f}", flush=True)
