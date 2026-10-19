"""Fast numerical sup over own data of  min(trivial bound, R-eliminated HS certificate ratio), payment F = max(P_lb, tw2).
Usage: python hs_sup2.py restarts seed
"""
import sys, time
import numpy as np
from scipy.optimize import minimize
from p52lib import *
from hs_elim import elim_ratio
from tw2 import pay_tw2

s = 1.5; G = Gamma(s); c1 = 1 - 3 ** (1 - s)
RS = int(sys.argv[1]); seed = int(sys.argv[2])


def F_of(m1, m2, m12, cphi):
    M = max(m1, m2, m12); RT = np.sqrt(m1 ** 2 + m2 ** 2 + m12 ** 2)
    plb = max(1 - c1 * M, 1 - RT / (2 * np.sqrt(3)), 3 ** (1 - s) * max(1, M))
    return max(plb, pay_tw2(1.0, m1, m2, m12, cphi, s, nL=150))


def ratio(v):
    al, m1, m2, m12, phi = v
    m1, m2, m12 = abs(m1), abs(m2), abs(m12)
    F = F_of(m1, m2, m12, np.cos(phi + al))
    triv = KA * max(np.cos(al), 0) / (G * F)
    if triv < 0.8:
        return triv
    try:
        return min(triv, elim_ratio(al, m1, m2, m12, phi, F))
    except Exception:
        return triv


rng = np.random.default_rng(seed)
best = (-1, None); t0 = time.time()
for k in range(RS):
    t = rng.uniform(0.6, 1.2)
    v0 = np.array([rng.normal(0, 0.15), t + rng.normal(0, 0.1), t + rng.normal(0, 0.1), t + rng.normal(0, 0.1), rng.normal(0, 0.3)])
    res = minimize(lambda v: -ratio(v), v0, method='Nelder-Mead', options={'maxiter': 400, 'xatol': 1e-4, 'fatol': 1e-6})
    if -res.fun > best[0]:
        best = (-res.fun, res.x)
    print(f"  start {k}: {-res.fun:.5f} at {np.round(res.x,3)}   best {best[0]:.5f}  ({time.time()-t0:.0f}s)", flush=True)
print("BEST", best[0], np.round(best[1], 4))
