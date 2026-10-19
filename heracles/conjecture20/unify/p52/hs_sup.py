"""Numerical sup of the HS-certificate ratio with payment F = max(P_lb, twisted) over own data (torus-normalised):
A = e^{i alpha}, B = m1, C = m2, D = m12 e^{i phi}.  ratio = max_R B_HS(d,R) / (Gamma sqrt(F^2 + R^2/12)) at s = 1.5.
Usage: python hs_sup.py restarts seed
"""
import sys, time
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from p52lib import *
from hs_cert import B_HS
from twisted_pay import twisted_lb

s = 1.5; G = Gamma(s); c1 = 1 - 3 ** (1 - s); cR = 3 ** (2 - 2 * s) / 4
RS = int(sys.argv[1]); seed = int(sys.argv[2])


def F_of(A, B, C, D):
    m = np.array([abs(B), abs(C), abs(D)]); M = m.max(); RT = np.sqrt((m ** 2).sum())
    plb = max(abs(A) - c1 * M, abs(A) - RT / (2 * np.sqrt(3)), 3 ** (1 - s) * max(abs(A), M))
    tw = twisted_lb(A, B, C, D, s, tries=6)
    return max(plb, tw)


def ratio(v, verbose=False):
    al, m1, m2, m12, phi = v
    A = np.exp(1j * al); B = abs(m1); C = abs(m2); D = abs(m12) * np.exp(1j * phi)
    F = F_of(A, B, C, D)
    if KA / (G * F) < 0.75:      # trivially fine
        return KA / (G * F)
    f = lambda lr: -(B_HS(A, [B, C, D], np.exp(lr)) or -1e9) / (G * np.sqrt(F ** 2 + cR * np.exp(2 * lr)))
    grid = np.linspace(np.log(0.02), np.log(5), 10)
    vals = [f(t) for t in grid]
    k = int(np.argmin(vals))
    res = minimize_scalar(f, bounds=(grid[max(k - 1, 0)], grid[min(k + 1, 9)]), method='bounded', options={'xatol': 1e-3})
    val = max(-res.fun, (B_HS(A, [B, C, D], 0.0) or -1e9) / (G * F))
    if verbose:
        print(f"   F={F:.4f} R*={np.exp(res.x):.3f}")
    return val


rng = np.random.default_rng(seed)
best = (-1, None); t0 = time.time()
for k in range(RS):
    t = rng.uniform(0.75, 1.1)
    v0 = np.array([rng.normal(0, 0.15), t + rng.normal(0, 0.05), t + rng.normal(0, 0.05), t + rng.normal(0, 0.05), rng.normal(0, 0.2)])
    res = minimize(lambda v: -ratio(v), v0, method='Nelder-Mead', options={'maxiter': 300, 'xatol': 1e-4, 'fatol': 1e-6})
    if -res.fun > best[0]:
        best = (-res.fun, res.x)
    print(f"  start {k}: {-res.fun:.5f} at {np.round(res.x,3)}   best {best[0]:.5f}  ({time.time()-t0:.0f}s)", flush=True)
print("BEST", best[0], np.round(best[1], 4))
ratio(best[1], verbose=True)
