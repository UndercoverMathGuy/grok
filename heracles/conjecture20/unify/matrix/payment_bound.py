"""Upper bound for V_A (single feature A={1}) at a dense prime via PROVED payments only.

For own data (beta, beta_1, beta_2, beta_12) of feature 1 (n=2), every atom system realising it has IF-price amplitude
P = 3^{1-s} Q^s satisfying (proved):
   P >= beta - c1 * max_T |beta_T|          (Note, Thm c1sharp, c1 = 1-3^{1-s}; computer-assisted)
   P >= beta - R/(2 sqrt 3),  R^2 = sum_T |beta_T|^2      (Part I, Theorem A)
   P >= 3^{1-s} beta                          (Q^s >= sum|beta_i| >= beta)
   P >= beta - c12 |beta_T|  if the other two sign flips vanish  (P12)   [not used in the max below unless exact]
V_A <= sup over (beta_T) of  h_free(beta, beta_T) / (Gamma_s(p) * P_lb),   h_free = exact LP with all non-own modes free
(fields Phi_j for every j in F; features other than 1 have zero own coefficients).
We maximise the ratio over beta_T in C^3 (beta = 1) by random starts + Nelder-Mead.  Usage: python payment_bound.py p b restarts
"""
import sys
import numpy as np
from scipy.optimize import minimize
from mrel import Model, subsets
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/mechanism_scripts')
from prog import solve as prog_solve

p = int(sys.argv[1]); b = float(sys.argv[2]); restarts = int(sys.argv[3])
n = 2; nu = 3; s = nu / b; r = 1 / s
F = (p - 1) // 2
G, ystar, _ = prog_solve(p, s, restarts=8)
M = Model(p, n, list(range(1, F + 1)), mode='free', s=s)
Ts = subsets(n)
c1 = 1 - 3 ** (1 - s)


def P_lb(bT):
    m = max(abs(z) for z in bT); R = np.sqrt(sum(abs(z) ** 2 for z in bT))
    return max(1 - c1 * m, 1 - R / (2 * np.sqrt(3)), 3 ** (1 - s))


def ratio(v):
    bT = [v[0] + 1j * v[1], v[2] + 1j * v[3], v[4] + 1j * v[5]]
    a = {j: {T: 0j for T in Ts} for j in range(1, F + 1)}
    a[1][()] = 1.0 + 0j
    for T, z in zip([T for T in Ts if T], bT):
        a[1][T] = np.conj(z)
    h, _ = M.margin(a)
    return h / (G * P_lb(bT))


rng = np.random.default_rng(0)
best = (-1, None)
print(f"p={p} b={b} s={s:.4f} Gamma={G:.5f} c1={c1:.4f}; single-feature IF ratio = kappa_min/Gamma = {(1-np.cos(2*np.pi/p))/G:.4f}", flush=True)
for k in range(restarts):
    v0 = rng.normal(size=6) * rng.uniform(0.05, 1.5)
    res = minimize(lambda v: -ratio(v), v0, method='Nelder-Mead', options={'maxiter': 4000, 'xatol': 1e-8, 'fatol': 1e-10})
    val = -res.fun
    if val > best[0]:
        best = (val, res.x)
    if k % 10 == 9:
        print(f"  restart {k}: {val:.5f}  best {best[0]:.6f}  at |beta_T| = {np.round(np.abs(best[1][0::2] + 1j*best[1][1::2]),4)}", flush=True)
v = best[1]
print(f"BEST bound on V_1: {best[0]:.6f};  beta_T = {np.round(v[0::2] + 1j*v[1::2], 5)}")
