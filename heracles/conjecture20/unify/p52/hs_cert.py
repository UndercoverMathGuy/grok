"""Hermitian-square certificates at (5,2) (torus-equivariant family).

For each delta, w_delta(theta) = v(theta)^* G_delta v(theta), v = (1, e^{i th1}, e^{i th2}, e^{i(th1+th2)}), G_delta PSD (4x4),
evaluated at theta_t = 2 pi (2 x_t)/5.  25 q(x,delta) = w_delta >= 0, sum_delta tr G_delta = 1.
Fourier: frequency f in {-1,0,1}^2 <-> character chi = 2f (mod 5).
Rigorous bound for every network with beta_2 = 0, own data (A, beta_T) and non-own l2 <= R:
  h <= Re(conj(A) sum_d c1(d) tr G_d) + sum_T Re(conj(beta_T) sum_d c1(d) what_d(-f_T)) + R n(G),
  n^2 = sum_T |sum_d c1(d) what_d(f_T)|^2 + sum_{+-}|sum_d c1(d) what_d(+-(1,-1))|^2 + sum_{f != 0} |sum_d c2(d) what_d(f)|^2.
"""
import itertools
import numpy as np
import cvxpy as cp
from p52lib import *

c1v = np.array([1 - rho ** (-d) for d in DELS])
c2v = np.array([1 - rho ** (-2 * d) for d in DELS])
E = [(0, 0), (1, 0), (0, 1), (1, 1)]
FR = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)]
FT = [(1, 0), (0, 1), (1, 1)]          # own frequencies f_T for T = {1}, {2}, {1,2}
_c = {}


def what_expr(G, f):
    """Fourier coefficient at frequency f of v^* G v: sum over e' - e = f of G[e, e'] (complex cvxpy expression)"""
    terms = []
    for i, e in enumerate(E):
        for j, e2 in enumerate(E):
            if (e2[0] - e[0], e2[1] - e[1]) == f:
                terms.append(G[i, j])
    return sum(terms) if terms else 0


def build():
    if 'p' in _c:
        return _c
    Gs = [cp.Variable((4, 4), hermitian=True) for _ in DELS]
    Are = cp.Parameter(); Aim = cp.Parameter()
    bre = cp.Parameter(3); bim = cp.Parameter(3)
    Rpar = cp.Parameter(nonneg=True)
    cons = [G >> 0 for G in Gs]
    cons.append(sum(cp.real(cp.trace(G)) for G in Gs) == 1)
    W = {f: [what_expr(G, f) for G in Gs] for f in FR}
    def dsum(cv, f):
        return sum(cv[k] * W[f][k] for k in range(4))
    mean = dsum(c1v, (0, 0))
    obj = Are * cp.real(mean) + Aim * cp.imag(mean)       # Re(conj(A) X) = ReA ReX + ImA ImX
    for k, f in enumerate(FT):
        X = dsum(c1v, (-f[0], -f[1]))
        obj = obj + bre[k] * cp.real(X) + bim[k] * cp.imag(X)
    leak = []
    for f in FT:
        X = dsum(c1v, f); leak += [cp.real(X), cp.imag(X)]
    for f in [(1, -1), (-1, 1)]:
        X = dsum(c1v, f); leak += [cp.real(X), cp.imag(X)]
    for f in FR:
        if f == (0, 0):
            continue
        X = dsum(c2v, f); leak += [cp.real(X), cp.imag(X)]
    obj = obj + Rpar * cp.norm(cp.hstack(leak))
    prob = cp.Problem(cp.Minimize(obj), cons)
    _c.update(dict(p=prob, G=Gs, Are=Are, Aim=Aim, bre=bre, bim=bim, R=Rpar))
    return _c


def B_HS(A, bT, R, ret=False):
    C = build()
    C['Are'].value = float(np.real(A)); C['Aim'].value = float(np.imag(A))
    C['bre'].value = np.array([np.real(b) for b in bT]); C['bim'].value = np.array([np.imag(b) for b in bT])
    C['R'].value = R
    C['p'].solve(solver=cp.CLARABEL)
    if ret:
        return C['p'].value, [G.value for G in C['G']]
    return C['p'].value


if __name__ == '__main__':
    from l2relax import h_L2
    s = 1.5
    pts = [(1, -0.312 + 0.364j, -0.03 - 0.465j, 0.493 + 0.384j),
           (1, 0.03 + 0.651j, -0.482 + 0.029j, -0.063 - 0.492j),
           (1, -0.42 + 0.28j, 0.502 + 0.036j, -0.569 + 0.347j)]
    K = 48
    xi = np.exp(2j * np.pi * 26 / K)
    pts.append((1, xi, xi, xi * xi))
    for (A, B, C, D) in pts:
        for R in [0.0, 0.3, 1.0]:
            print(f"|B,C,D|=({abs(B):.3f},{abs(C):.3f},{abs(D):.3f}) R={R}: B_HS={B_HS(A,[B,C,D],R):.4f}  h_L2={h_L2(A,[B,C,D],R):.4f}")
    # torus equivariance check: rotate B by e^{i t1}, C by e^{i t2}, D by e^{i(t1+t2)}
    A, B, C, D = pts[0]
    for t1, t2 in [(0.3, -1.1), (2.0, 0.7)]:
        print("rotated:", B_HS(A, [B * np.exp(1j*t1), C * np.exp(1j*t2), D * np.exp(1j*(t1+t2))], 0.3), " h_L2 rotated:",
              h_L2(A, [B * np.exp(1j*t1), C * np.exp(1j*t2), D * np.exp(1j*(t1+t2))], 0.3))
