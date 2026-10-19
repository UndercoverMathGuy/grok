"""HS certificate with the R-dimension eliminated (single certificate valid for all R >= 0).

For any certificate G (Hermitian squares, see hs_cert.py) with own part Bnd_0(G; d) and leakage n(G):
   sup_{R>=0} (Bnd_0 + n R) / (Gamma sqrt(F^2 + c R^2)) <= || ( max(Bnd_0,0)/F , n/sqrt(c) ) ||_2 / Gamma.
So  ratio <= min_G || (max(Bnd_0(G),0)/F, n(G)/sqrt c) || / Gamma   -- a convex program in G.
Data in torus-normal form: A = e^{i alpha}, B = m1, C = m2 (real >= 0), D = m12 e^{i phi}.
"""
import numpy as np
import cvxpy as cp
from p52lib import *
from hs_cert import what_expr, c1v, c2v, FR, FT

_c = {}


def build(deg=1):
    if deg in _c:
        return _c[deg]
    Gs = [cp.Variable((4, 4), hermitian=True) for _ in DELS]
    Are = cp.Parameter(); Aim = cp.Parameter()
    bre = cp.Parameter(3); bim = cp.Parameter(3)
    invF = cp.Parameter(nonneg=True); invsc = cp.Parameter(nonneg=True)
    cons = [G >> 0 for G in Gs]
    cons.append(sum(cp.real(cp.trace(G)) for G in Gs) == 1)
    W = {f: [what_expr(G, f) for G in Gs] for f in FR}

    def dsum(cv, f):
        return sum(cv[k] * W[f][k] for k in range(4))
    mean = dsum(c1v, (0, 0))
    bnd0 = Are * cp.real(mean) + Aim * cp.imag(mean)
    for k, f in enumerate(FT):
        X = dsum(c1v, (-f[0], -f[1]))
        bnd0 = bnd0 + bre[k] * cp.real(X) + bim[k] * cp.imag(X)
    leak = []
    for f in FT:
        X = dsum(c1v, f); leak += [cp.real(X), cp.imag(X)]
    for f in [(1, -1), (-1, 1)]:
        X = dsum(c1v, f); leak += [cp.real(X), cp.imag(X)]
    for f in FR:
        if f == (0, 0):
            continue
        X = dsum(c2v, f); leak += [cp.real(X), cp.imag(X)]
    u = cp.Variable(nonneg=True)
    cons.append(u >= bnd0)
    obj = cp.norm(cp.hstack([u * invF, invsc * cp.norm(cp.hstack(leak))]))
    prob = cp.Problem(cp.Minimize(obj), cons)
    _c[deg] = dict(p=prob, G=Gs, Are=Are, Aim=Aim, bre=bre, bim=bim, invF=invF, invsc=invsc)
    return _c[deg]


def elim_ratio(alpha, m1, m2, m12, phi, F, s=1.5, ret=False):
    C = build()
    A = np.exp(1j * alpha); bT = [m1, m2, m12 * np.exp(1j * phi)]
    C['Are'].value = float(A.real); C['Aim'].value = float(A.imag)
    C['bre'].value = np.array([np.real(b) for b in bT], float); C['bim'].value = np.array([np.imag(b) for b in bT], float)
    C['invF'].value = 1.0 / F
    cR = 3 ** (2 - 2 * s) / 4
    C['invsc'].value = 1.0 / np.sqrt(cR)
    C['p'].solve(solver=cp.CLARABEL)
    val = C['p'].value / Gamma(s)
    if ret:
        return val, [G.value for G in C['G']]
    return val


if __name__ == '__main__':
    s = 1.5
    for al in [0.0, 0.1, 0.2, 0.3]:
        print(f"single atom alpha={al}: R-eliminated HS ratio {elim_ratio(al, 1, 1, 1, -al, 3**(1-s)):.4f}")
