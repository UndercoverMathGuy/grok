"""Degree-d Hermitian-square certificates (torus-equivariant), R eliminated.  d = 1 or 2.

w_delta(theta) = v^* G_delta v,  v = (e^{i<e,theta>})_{e in {0..d}^2},  theta_t = 2 pi (2 x_t)/5.
Frequency f in {-d..d}^2 <-> character chi = 2 f (mod 5) (injective for d <= 2).
Own pairs (feature 1): clean f=0 with conj(A);  sign flips: what(-f_T) pairs with conj(beta_T), f_T in {(1,0),(0,1),(1,1)}.
Non-own: Phi_1 at every other character; Phi_2 at every character except 0.
ratio <= min_G || (max(Bnd_0,0)/F , n/sqrt(c)) || / Gamma.
"""
import numpy as np
import cvxpy as cp
from p52lib import *

c1v = np.array([1 - rho ** (-d) for d in DELS])
c2v = np.array([1 - rho ** (-2 * d) for d in DELS])
FT = [(1, 0), (0, 1), (1, 1)]
_c = {}


def chi_of(f):
    return ((2 * f[0]) % 5, (2 * f[1]) % 5)


def build(deg):
    if deg in _c:
        return _c[deg]
    E = [(a, b) for a in range(deg + 1) for b in range(deg + 1)]
    nE = len(E)
    FR = sorted(set((e2[0] - e[0], e2[1] - e[1]) for e in E for e2 in E))
    Gs = [cp.Variable((nE, nE), hermitian=True) for _ in DELS]
    # coefficient maps: what_delta(f) = sum_{e'-e=f} G[e,e']
    idx = {f: [(i, j) for i, e in enumerate(E) for j, e2 in enumerate(E) if (e2[0] - e[0], e2[1] - e[1]) == f] for f in FR}
    Are = cp.Parameter(); Aim = cp.Parameter()
    bre = cp.Parameter(3); bim = cp.Parameter(3)
    invF = cp.Parameter(nonneg=True); invsc = cp.Parameter(nonneg=True)
    cons = [G >> 0 for G in Gs]
    cons.append(sum(cp.real(cp.trace(G)) for G in Gs) == 1)

    def W(k, f):
        return sum(Gs[k][i, j] for (i, j) in idx[f]) if f in idx else 0

    def dsum(cv, f):
        return sum(cv[k] * W(k, f) for k in range(4))
    mean = dsum(c1v, (0, 0))
    bnd0 = Are * cp.real(mean) + Aim * cp.imag(mean)
    for k, f in enumerate(FT):
        X = dsum(c1v, (-f[0], -f[1]))
        bnd0 = bnd0 + bre[k] * cp.real(X) + bim[k] * cp.imag(X)
    own_pairs = {(0, 0)} | {(-f[0], -f[1]) for f in FT}     # what-frequencies that pair with own coefficients of Phi_1
    leak = []
    for f in FR:
        if f not in own_pairs:
            X = dsum(c1v, f); leak += [cp.real(X), cp.imag(X)]
        if f != (0, 0):
            X = dsum(c2v, f); leak += [cp.real(X), cp.imag(X)]
    u = cp.Variable(nonneg=True)
    cons.append(u >= bnd0)
    obj = cp.norm(cp.hstack([u * invF, invsc * cp.norm(cp.hstack(leak))]))
    prob = cp.Problem(cp.Minimize(obj), cons)
    _c[deg] = dict(p=prob, G=Gs, Are=Are, Aim=Aim, bre=bre, bim=bim, invF=invF, invsc=invsc, E=E, FR=FR, idx=idx)
    return _c[deg]


def elim_ratio(alpha, m1, m2, m12, phi, F, deg=2, s=1.5, ret=False):
    C = build(deg)
    A = np.exp(1j * alpha); bT = [m1, m2, m12 * np.exp(1j * phi)]
    C['Are'].value = float(A.real); C['Aim'].value = float(A.imag)
    C['bre'].value = np.array([np.real(b) for b in bT], float); C['bim'].value = np.array([np.imag(b) for b in bT], float)
    C['invF'].value = 1.0 / F
    C['invsc'].value = 1.0 / np.sqrt(3 ** (2 - 2 * s) / 4)
    C['p'].solve(solver=cp.CLARABEL)
    val = C['p'].value / Gamma(s)
    if ret:
        return val, [G.value for G in C['G']]
    return val


if __name__ == '__main__':
    s = 1.5
    for (al, m1, m2, m12, phi, F) in [(0, 1, 1, 1, 0, 3 ** (1 - s)), (0, 0.555, 0.582, 0.856, 0.269, 0.7663),
                                      (0, 0.628, 0.663, 0.881, 0.021, None)]:
        if F is None:
            from tw2 import pay_tw2
            F = pay_tw2(1.0, m1, m2, m12, np.cos(phi + al), s)
        print(f"data {(al,m1,m2,m12,phi)} F={F:.4f}:  deg1 {elim_ratio(al,m1,m2,m12,phi,F,deg=1):.4f}   deg2 {elim_ratio(al,m1,m2,m12,phi,F,deg=2):.4f}")
