"""Floating-point scan of the two-piece relaxation for the n=3 pointwise payment:
   theta Q^s / L >= max( F^V_r(Pbar), F^pert_r(a,m) ),
   F^V_r(P)  = min{ M_r(nu) : nu>=0, sum nu = 4, sum (nu-1)^2 <= 2P }        (variance-only)
   F^pert_r  = ( [(1+G)^r+(1-G)^r]/2 - r(1-r)(1-G-2m)^{r-2} m^2 )^{1/r}       (block perturbation; needs 1-G-2m>0)
   Pbar = 4 m^2 + 2 G^2 (pointwise currency) or 3 m^2 + 2 G^2 (pair-family currency 'cf').
K(s) = sup_{a,m} (a - max(F^V, F^pert))/m.     usage: python vp_scan.py s [cf]
"""
import sys
import numpy as np
from kbound import Fr, G

s = float(sys.argv[1]); r = 1 / s
cf = len(sys.argv) > 2 and sys.argv[2] == 'cf'


def Fpert(a, m):
    g = G(a, m)
    lo = 1 - g - 2 * m
    if lo <= 0:
        return -np.inf
    br = ((1 + g) ** r + (1 - g) ** r) / 2 - r * (1 - r) * lo ** (r - 2) * m * m
    return br ** (1 / r) if br > 0 else -np.inf


best = (-np.inf, None)
A = np.linspace(0.0, 1.0, 501)
Ms = np.linspace(0.002, 1.0, 500)
for a in A:
    for m in Ms:
        g = G(a, m)
        Pb = (3 if cf else 4) * m * m + 2 * g * g
        F = max(Fr(Pb, r), Fpert(a, m))
        val = (a - F) / m
        if val > best[0]:
            best = (val, (a, m, g, Fr(Pb, r), Fpert(a, m)))
print(f"s={s} cf={cf}: K ~ {best[0]:.4f} at (a,m,G,FV,Fpert) = {np.round(best[1], 4)}")
