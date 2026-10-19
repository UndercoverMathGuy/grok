"""Minimal atom cost Q*(d) = inf sum |beta_i|^r over atom systems realising own data d = (A,B,C,D) (n=2):
A = sum beta_i, B = sum beta_i xi_i1, C = sum beta_i xi_i2, D = sum beta_i xi_i1 xi_i2.
Penalty method + Nelder-Mead/L-BFGS over m atoms.  Prints P_1 = 3^{1-s} Q*^s against the proved lower bound P_lb.
Usage: python qstar.py s m restarts  A_re A_im B_re B_im C_re C_im D_re D_im
"""
import sys
import numpy as np
from scipy.optimize import minimize

s = float(sys.argv[1]); m = int(sys.argv[2]); RS = int(sys.argv[3])
v = [float(t) for t in sys.argv[4:12]]
d = np.array([v[0] + 1j * v[1], v[2] + 1j * v[3], v[4] + 1j * v[5], v[6] + 1j * v[7]])
r = 1 / s; c1 = 1 - 3 ** (1 - s)


def own(x):
    beta = x[:m] + 1j * x[m:2 * m]
    ph = x[2 * m:].reshape(m, 2)
    xi = np.exp(1j * ph)
    return np.array([beta.sum(), (beta * xi[:, 0]).sum(), (beta * xi[:, 1]).sum(), (beta * xi[:, 0] * xi[:, 1]).sum()]), beta


def obj(x, lam):
    o, beta = own(x)
    return np.sum((np.abs(beta) ** 2 + 1e-12) ** (r / 2)) + lam * np.sum(np.abs(o - d) ** 2)


rng = np.random.default_rng(0)
best = np.inf
for k in range(RS):
    x = np.concatenate([rng.normal(0, 0.5, 2 * m), rng.uniform(0, 2 * np.pi, 2 * m)])
    for lam in [1, 10, 100, 1e3, 1e4, 1e5, 1e6]:
        res = minimize(lambda z: obj(z, lam), x, method='L-BFGS-B', options={'maxiter': 5000})
        x = res.x
    o, beta = own(x)
    err = np.max(np.abs(o - d))
    Q = np.sum(np.abs(beta) ** r)
    if err < 1e-4 and Q < best:
        best = Q; bx = x
A, B, C, D = d
M = max(abs(B), abs(C), abs(D)); RT = np.sqrt(abs(B) ** 2 + abs(C) ** 2 + abs(D) ** 2)
plb = max(abs(A) - c1 * M, abs(A) - RT / (2 * np.sqrt(3)), 3 ** (1 - s) * max(abs(A), M))
print(f"Q*(d) <= {best:.5f}:  P_1 = 3^(1-s) Q^s = {3**(1-s)*best**s:.5f};  P_lb = {plb:.5f}")
o, beta = own(bx)
print("atoms |beta| =", np.round(np.abs(beta), 4))
