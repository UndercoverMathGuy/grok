"""Sup over own data d = (A,B,C,D) (A = e^{i alpha}) of the L^2 relaxation ratio with the proved payment lower bound:
   ratio(d) = max_R h_L2(d;R) / (Gamma_s sqrt(P_lb(d)^2 + 3^{2-2s} R^2/4)),
   P_lb(d) = max(|A| - c_1 M, |A| - R_T/(2 sqrt 3), 3^{1-s} max(|A|, M)),  M = max|B,C,D|, R_T^2 = |B|^2+|C|^2+|D|^2.
Usage: python l2plb.py s restarts seed
"""
import sys, time
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from p52lib import *
from l2relax import h_L2

s = float(sys.argv[1]); RS = int(sys.argv[2]); seed = int(sys.argv[3])
G = Gamma(s); c1 = 1 - 3 ** (1 - s); cR = 3 ** (2 - 2 * s) / 4


def P_lb(A, bT):
    M = max(abs(z) for z in bT); RT = np.sqrt(sum(abs(z) ** 2 for z in bT))
    return max(abs(A) - c1 * M, abs(A) - RT / (2 * np.sqrt(3)), 3 ** (1 - s) * max(abs(A), M))


def best_R(A, bT, P1):
    f = lambda lr: -h_L2(A, bT, np.exp(lr)) / (G * np.sqrt(P1 ** 2 + cR * np.exp(2 * lr)))
    grid = np.linspace(np.log(1e-3), np.log(20), 12)
    vals = [f(v) for v in grid]
    k = int(np.argmin(vals))
    lo = grid[max(k - 1, 0)]; hi = grid[min(k + 1, len(grid) - 1)]
    res = minimize_scalar(f, bounds=(lo, hi), method='bounded', options={'xatol': 1e-3})
    return -res.fun, np.exp(res.x)


def ratio(v):
    A = np.exp(1j * v[0])
    bT = [v[1] + 1j * v[2], v[3] + 1j * v[4], v[5] + 1j * v[6]]
    try:
        val, R = best_R(A, bT, P_lb(A, bT))
    except Exception:
        return 0.0
    return val


rng = np.random.default_rng(seed)
best = (-1, None); t0 = time.time()
for k in range(RS):
    v0 = np.concatenate([[rng.normal(0, 0.1)], rng.normal(0, 0.5, 6)])
    res = minimize(lambda v: -ratio(v), v0, method='Nelder-Mead', options={'maxiter': 1500, 'xatol': 1e-5, 'fatol': 1e-7})
    val = -res.fun
    if val > best[0]:
        best = (val, res.x)
    v = res.x
    print(f"  start {k}: {val:.5f}   best {best[0]:.5f}  d=({np.round(v[0],3)}; {np.round(v[1:],3)})  ({time.time()-t0:.0f}s)", flush=True)
np.save(f"l2plb_s{s}_seed{seed}.npy", best[1])
print(f"s={s}: sup ratio (L2 relaxation, P_lb) >= {best[0]:.5f}")
