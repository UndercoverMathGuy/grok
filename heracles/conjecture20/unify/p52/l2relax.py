"""L^2-paid relaxation at (5,2).

h_L2(d; R) = max h  s.t.  g(x,delta) >= h,  E_x g(.,delta) = Re(conj(beta1) c_1(delta)),
             own coefficients of Phi_1 at chi_{1,T} = conj(beta_{1,T}),
             E_x|Phi_1 - Phi_1^own|^2 + E_x|Phi_2|^2 <= R^2      (all non-own modes, Phi_2 has mean 0)
where Phi_j(x) = -(2/5) sum_delta g(x,delta) rho^{j delta}.
Valid because every network with beta_2 = 0 has ||non-own modes||_2 <= R := 2 sum_i sqrt(e_i^2 - |beta_i1|^2), and
3^{1-s} B_5 P^s >= sqrt(P_1^2 + 3^{2-2s} R^2 / 4),  P_1 = 3^{1-s} Q^s.
"""
import numpy as np
import cvxpy as cp
from p52lib import *

N = 25
XS = X  # 25 x 2
_cache = {}


def _build():
    if 'prob' in _cache:
        return _cache
    g = cp.Variable((4, N))
    h = cp.Variable()
    mpar = cp.Parameter(4)
    ownre = cp.Parameter(3); ownim = cp.Parameter(3)
    # fixed own field Phi_1^own(x) real/imag parts as parameters
    phiown_re = cp.Parameter(N); phiown_im = cp.Parameter(N)
    Rpar = cp.Parameter(nonneg=True)
    cons = [g >= h]
    for di in range(4):
        cons.append(cp.sum(g[di]) / N == mpar[di])
    for k, chi in enumerate(OWN1):
        ph = rho ** (-(chi[0] * XS[:, 0] + chi[1] * XS[:, 1])) / N
        expr_re = 0; expr_im = 0
        for di, d in enumerate(DELS):
            c = rho ** d * ph
            expr_re = expr_re + c.real @ g[di]
            expr_im = expr_im + c.imag @ g[di]
        cons += [expr_re == ownre[k], expr_im == ownim[k]]
    # fields
    re1 = 0; im1 = 0; re2 = 0; im2 = 0
    for di, d in enumerate(DELS):
        re1 = re1 - (2 / 5) * np.cos(2 * np.pi * d / 5) * g[di]
        im1 = im1 - (2 / 5) * np.sin(2 * np.pi * d / 5) * g[di]
        re2 = re2 - (2 / 5) * np.cos(4 * np.pi * d / 5) * g[di]
        im2 = im2 - (2 / 5) * np.sin(4 * np.pi * d / 5) * g[di]
    vec = cp.hstack([re1 - phiown_re, im1 - phiown_im, re2, im2]) / np.sqrt(N)
    cons.append(cp.norm(vec, 2) <= Rpar)
    prob = cp.Problem(cp.Maximize(h), cons)
    _cache.update(dict(prob=prob, g=g, h=h, mpar=mpar, ownre=ownre, ownim=ownim, phre=phiown_re, phim=phiown_im, R=Rpar))
    return _cache


def h_L2(beta1, bT, R):
    C = _build()
    C['mpar'].value = np.array([np.real(np.conj(beta1) * (1 - rho ** (-d))) for d in DELS])
    vals = [-(5.0 / 2.0) * np.conj(b) for b in bT]
    C['ownre'].value = np.array([v.real for v in vals]); C['ownim'].value = np.array([v.imag for v in vals])
    own = np.conj(beta1) * np.ones(N, dtype=complex)
    for k, chi in enumerate(OWN1):
        own = own + np.conj(bT[k]) * rho ** (chi[0] * XS[:, 0] + chi[1] * XS[:, 1])
    C['phre'].value = own.real; C['phim'].value = own.imag
    C['R'].value = R
    C['prob'].solve(solver=cp.CLARABEL)
    return C['h'].value


if __name__ == '__main__':
    import sys
    s = 1.5
    # checks: R = 0 should reproduce class-C single-feature value; R = large -> h_free
    beta = np.array([1.0 + 0j]); K = 48
    xi = np.array([[np.exp(2j * np.pi * 26 / K), np.exp(2j * np.pi * 26 / K)]])
    A, bT = own_from_atoms(beta, xi)
    print("h_free =", h_free(A, bT))
    for R in [0, 0.1, 0.2, 0.5, 1, 2, 5, 20]:
        hv = h_L2(A, bT, R)
        P1 = 3 ** (1 - s) * 1.0
        print(f"R={R}: h_L2={hv:.5f}  ratio={hv / (Gamma(s) * np.sqrt(P1**2 + 3**(2-2*s)*R**2/4)):.5f}")
