"""Structured relaxation: min M_r(spec Hhat) over 4x4 correlation matrices with
   |H21|,|H31|,|H42|,|H43| <= m (exact 4-cycle 1-2-4-3-1) and |H32|,|H41| <= g (the two diagonals),
then K_struct(s) = sup_{a,m} (a - minMr(m, G(a,m)))/m.   Floating-point exploration only.
"""
import sys
import numpy as np
from scipy.optimize import minimize
from kbound import G, Fr

POS_EX = [(1, 0), (2, 0), (3, 1), (3, 2)]
POS_NE = [(2, 1), (3, 0)]


def build(v, m, g):
    # v: 6 moduli in (0,1) via sigmoid-ish scaling, 6 phases
    H = np.eye(4, dtype=complex)
    mods = 1 / (1 + np.exp(-v[:6]))
    ph = v[6:12]
    for k, (i, j) in enumerate(POS_EX + POS_NE):
        bound = m if k < 4 else g
        z = bound * mods[k] * np.exp(1j * ph[k])
        H[i, j] = z; H[j, i] = np.conj(z)
    return H


def obj(v, m, g, r):
    H = build(v, m, g)
    nu = np.linalg.eigvalsh(H)
    if nu.min() < 0:
        return 10 + (-nu.min()) * 100   # infeasible
    return np.mean(nu ** r) ** (1 / r)


def minMr(m, g, r, rng, starts=40):
    best = np.inf
    for t in range(starts):
        v = np.concatenate([rng.normal(3, 1, 6), rng.uniform(-np.pi, np.pi, 6)])
        res = minimize(obj, v, args=(m, g, r), method='Nelder-Mead', options=dict(maxiter=4000, xatol=1e-8, fatol=1e-10))
        best = min(best, res.fun)
    return best


if __name__ == '__main__':
    s = float(sys.argv[1]); r = 1 / s
    rng = np.random.default_rng(0)
    best = (-np.inf, None)
    for a in np.linspace(0.80, 1.0, 11):
        for m in np.linspace(0.05, 0.6, 12):
            g = G(a, m)
            mm = minMr(m, g, r, rng, 12)
            val = (a - mm) / m
            pon = (a - Fr(4 * m * m + 2 * g * g, r)) / m
            print(f"a={a:.3f} m={m:.3f} g={g:.3f}: struct minMr={mm:.5f} ratio={val:.4f}  (P-only ratio {pon:.4f})", flush=True)
            if val > best[0]:
                best = (val, (a, m))
    print("K_struct(%.4f) ~ %.4f at a,m=%s" % (s, best[0], best[1]))
