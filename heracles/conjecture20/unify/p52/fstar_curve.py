"""True phase-blind payment curve (numerical): F*(x; s) = inf { 3^{1-s} (sum_i |beta_i|^{1/s})^s : |A| = 1, max_T |beta_T| <= x }
over atom systems (beta_i, xi_i, eta_i), beta_T = sum_i beta_i prod_{t in T} (xi_it).  Penalty method + Nelder-Mead / SLSQP,
random restarts, up to m atoms.  Usage: python fstar_curve.py s m restarts"""
import sys
import numpy as np
from scipy.optimize import minimize

s = float(sys.argv[1]); m = int(sys.argv[2]); RS = int(sys.argv[3]); r = 1 / s
rng = np.random.default_rng(1)


def unpack(v):
    a = np.abs(v[:m]); ph = v[m:2 * m]; t1 = v[2 * m:3 * m]; t2 = v[3 * m:4 * m]
    beta = a * np.exp(1j * ph)
    A = beta.sum(); B = (beta * np.exp(1j * t1)).sum(); C = (beta * np.exp(1j * t2)).sum(); D = (beta * np.exp(1j * (t1 + t2))).sum()
    return a, A, B, C, D


def obj(v, x, pen):
    a, A, B, C, D = unpack(v)
    P1 = 3 ** (1 - s) * np.sum(a ** r) ** s
    viol = max(0, 1 - abs(A)) ** 2 + sum(max(0, abs(z) - x) ** 2 for z in (B, C, D))
    return P1 + pen * viol


for x in (0.2, 0.4, 0.5, 0.6, 0.656, 0.7, 0.8, 0.9, 1.0):
    best = np.inf
    for k in range(RS):
        v = np.concatenate([rng.uniform(0.1, 1, m) / m * 2, rng.uniform(-0.3, 0.3, m), rng.uniform(0, 2 * np.pi, 2 * m)])
        for pen in (1e2, 1e4, 1e6):
            res = minimize(obj, v, args=(x, pen), method='Nelder-Mead', options={'maxiter': 6000, 'xatol': 1e-9, 'fatol': 1e-12})
            v = res.x
        a, A, B, C, D = unpack(v)
        # rescale to make |A| = 1 exactly and check feasibility
        sc = 1 / abs(A)
        if max(abs(B), abs(C), abs(D)) * sc <= x + 1e-6:
            P1 = 3 ** (1 - s) * np.sum((a * sc) ** r) ** s
            best = min(best, P1)
    c1 = 1 - 3 ** (1 - s)
    print(f"s={s} m={m} x={x}: F*~{best:.4f}   c1sharp {max(1 - c1 * x, 3 ** (1 - s)):.4f}", flush=True)
