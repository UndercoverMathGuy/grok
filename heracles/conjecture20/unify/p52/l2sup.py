"""Supremum over atom systems of the L^2-paid relaxation ratio at (5,2).
ratio(atoms) = max_R h_L2(d(atoms); R) / (Gamma_s sqrt(P_1^2 + 3^{2-2s} R^2/4)),  P_1 = 3^{1-s} Q^s (exact atom cost)
Usage: python l2sup.py s natoms restarts seed [mode]   mode = exact (Q) or plb (proved payment lower bound)
"""
import sys, time
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from p52lib import *
from l2relax import h_L2

s = float(sys.argv[1]); m = int(sys.argv[2]); RS = int(sys.argv[3]); seed = int(sys.argv[4])
mode = sys.argv[5] if len(sys.argv) > 5 else 'exact'
r = 1 / s; G = Gamma(s); c1 = 1 - 3 ** (1 - s)
cR = 3 ** (2 - 2 * s) / 4


def P_lb(A, bT):
    M = max(abs(z) for z in bT); RT = np.sqrt(sum(abs(z) ** 2 for z in bT))
    return max(abs(A) - c1 * M, abs(A) - RT / (2 * np.sqrt(3)), 3 ** (1 - s) * abs(A))


def best_R(A, bT, P1):
    f = lambda lr: -h_L2(A, bT, np.exp(lr)) / (G * np.sqrt(P1 ** 2 + cR * np.exp(2 * lr)))
    # coarse grid then refine
    grid = np.linspace(np.log(1e-3 * P1 + 1e-12), np.log(20 * P1), 14)
    vals = [f(v) for v in grid]
    k = int(np.argmin(vals))
    lo = grid[max(k - 1, 0)]; hi = grid[min(k + 1, len(grid) - 1)]
    res = minimize_scalar(f, bounds=(lo, hi), method='bounded', options={'xatol': 1e-4})
    return -res.fun, np.exp(res.x)


def unpack(v):
    beta = v[0:m] * np.exp(1j * v[m:2 * m])
    xi = np.exp(1j * v[2 * m:4 * m].reshape(m, 2))
    return beta, xi


def ratio(v):
    beta, xi = unpack(v)
    Q = np.sum(np.abs(beta) ** r)
    if Q < 1e-9:
        return 0.0
    A, bT = own_from_atoms(beta, xi)
    P1 = 3 ** (1 - s) * Q ** s if mode == 'exact' else P_lb(A, bT)
    try:
        val, R = best_R(A, bT, P1)
    except Exception:
        return 0.0
    return val


rng = np.random.default_rng(seed)
best = (-1, None); t0 = time.time()
for k in range(RS):
    v0 = np.concatenate([rng.uniform(0.2, 1, m), rng.normal(0, 0.3, m), rng.uniform(0, 2 * np.pi, 2 * m)])
    if k == 0 and m >= 3:
        v0 = np.concatenate([np.ones(m) / m, np.zeros(m), np.array([[2 * np.pi * i / 3] * 2 for i in range(m)]).ravel()])
    res = minimize(lambda v: -ratio(v), v0, method='Nelder-Mead', options={'maxiter': 400 * m, 'xatol': 1e-6, 'fatol': 1e-8})
    val = -res.fun
    if val > best[0]:
        best = (val, res.x)
    print(f"  start {k}: {val:.5f}   best {best[0]:.5f}   ({time.time()-t0:.0f}s)", flush=True)
beta, xi = unpack(best[1])
A, bT = own_from_atoms(beta, xi)
Q = np.sum(np.abs(beta) ** r)
print(f"s={s} m={m} mode={mode}: best L2 ratio {best[0]:.5f};  |beta|={np.round(np.abs(beta),4)}  |A|/(3^(1-s)Q^s)={abs(A)/(3**(1-s)*Q**s):.4f}  |BCD|/|A|={np.round(np.abs(bT)/abs(A),3)}")
np.save(f"l2sup_s{s}_m{m}_{mode}_seed{seed}.npy", best[1])
