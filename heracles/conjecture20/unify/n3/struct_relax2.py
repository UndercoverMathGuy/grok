"""Faster structured relaxation: moduli at their bounds (m on the exact 4-cycle, g on the 2 diameters), minimise
M_r over the 3 gauge-invariant phases (fluxes); compare with full 12-parameter search at a few points.
Then scan (a,m) on a grid near the extremal region.  Floating point exploration.
usage: python struct_relax2.py s
"""
import sys
import numpy as np
from scipy.optimize import minimize
from kbound import G, Fr
from struct_relax import minMr as minMr_full

# vertices 0..3 = f1,f2,f3,f4 ; exact cycle edges (1,0),(2,0),(3,1),(3,2); diameters (2,1),(3,0)
def Hmat(ph, m, g):
    # gauge: edges (1,0),(2,0),(3,1) real positive; phases on (3,2),(2,1),(3,0)
    H = np.eye(4, dtype=complex)
    vals = {(1, 0): m, (2, 0): m, (3, 1): m, (3, 2): m * np.exp(1j * ph[0]),
            (2, 1): g * np.exp(1j * ph[1]), (3, 0): g * np.exp(1j * ph[2])}
    for (i, j), z in vals.items():
        H[i, j] = z; H[j, i] = np.conj(z)
    return H


def Mr_of(H, r):
    nu = np.linalg.eigvalsh(H)
    if nu.min() < -1e-12:
        return 10.0 + 100 * (-nu.min())
    nu = np.clip(nu, 0, None)
    return np.mean(nu ** r) ** (1 / r)


def minMr_phase(m, g, r, starts=None):
    best = np.inf
    grid = np.linspace(-np.pi, np.pi, 9)[:-1]
    for p0 in grid:
        for p1 in grid:
            for p2 in grid:
                v = Mr_of(Hmat((p0, p1, p2), m, g), r)
                if v < best:
                    best, arg = v, (p0, p1, p2)
    res = minimize(lambda ph: Mr_of(Hmat(ph, m, g), r), arg, method='Nelder-Mead',
                   options=dict(xatol=1e-10, fatol=1e-12, maxiter=2000))
    return min(best, res.fun), res.x


if __name__ == '__main__':
    s = float(sys.argv[1]); r = 1 / s
    rng = np.random.default_rng(1)
    # 1) compare with full search at a few points
    for (a, m) in [(0.9757, 0.2747), (0.95, 0.35), (0.9, 0.15), (0.97, 0.45)]:
        g = G(a, m)
        v1, ph = minMr_phase(m, g, r)
        v2 = minMr_full(m, g, r, rng, 8)
        print(f"check a={a} m={m} g={g:.4f}: phase-only {v1:.5f}  full {v2:.5f}  phases {np.round(ph,3)}", flush=True)
    # 2) scan
    best = (-np.inf, None)
    for a in np.linspace(0.88, 1.0, 25):
        for m in np.linspace(0.02, 0.7, 35):
            g = G(a, m)
            v, _ = minMr_phase(m, g, r)
            ratio = (a - v) / m
            if ratio > best[0]:
                best = (ratio, (a, m, g, v))
    print("K_struct(s=%.4f) ~ %.4f at (a,m,g,minMr)=%s" % (s, best[0], np.round(best[1], 4)), flush=True)
