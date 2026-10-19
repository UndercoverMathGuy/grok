"""max-T relaxation at (5,2).

h_maxT(A, Tstar, M, phi, R) = max h  s.t. g >= h, E g(.,delta) = Re(conj(A) c_1(delta)),
   coefficient of Phi_1 at chi_{1,T*}  = M e^{-i phi}          (conj(beta_{1,T*}) with |beta_{1,T*}| = M = max_T |beta_{1,T}|)
   |coefficient of Phi_1 at chi_{1,T}| <= M   for T != T*      (other sign flips: free phase, modulus <= M)
   ||non-own modes of (Phi_1, Phi_2)||_2 <= R.
Ratio bound: h / (Gamma_s sqrt(P_lb^2 + 3^{2-2s} R^2/4)),  P_lb = max(|A| - c_1 M, 3^{1-s}|A|).
"""
import numpy as np
import cvxpy as cp
from p52lib import *

N = 25
XS = X
_c = {}


def chi_row(chi):
    return rho ** (-(chi[0] * XS[:, 0] + chi[1] * XS[:, 1])) / N


def build(Tstar):
    if Tstar in _c:
        return _c[Tstar]
    g = cp.Variable((4, N)); h = cp.Variable()
    mpar = cp.Parameter(4); sre = cp.Parameter(); sim = cp.Parameter(); Mpar = cp.Parameter(nonneg=True)
    Rpar = cp.Parameter(nonneg=True)
    cons = [g >= h]
    for di in range(4):
        cons.append(cp.sum(g[di]) / N == mpar[di])
    # Phi_1 coefficient at chi: -(2/5) sum_d rho^d ghat_d(chi)
    def phi1coef(chi):
        ph = chi_row(chi); re = 0; im = 0
        for di, d in enumerate(DELS):
            c = -(2 / 5) * rho ** d * ph
            re = re + c.real @ g[di]; im = im + c.imag @ g[di]
        return re, im
    own_re = {}; own_im = {}
    for k, chi in enumerate(OWN1):
        re, im = phi1coef(chi)
        own_re[k] = re; own_im[k] = im
        if k == Tstar:
            cons += [re == sre, im == sim]
        else:
            cons.append(cp.norm(cp.hstack([re, im]), 2) <= Mpar)
    # fields and their own parts
    re1 = 0; im1 = 0; re2 = 0; im2 = 0
    for di, d in enumerate(DELS):
        re1 = re1 - (2 / 5) * np.cos(2 * np.pi * d / 5) * g[di]
        im1 = im1 - (2 / 5) * np.sin(2 * np.pi * d / 5) * g[di]
        re2 = re2 - (2 / 5) * np.cos(4 * np.pi * d / 5) * g[di]
        im2 = im2 - (2 / 5) * np.sin(4 * np.pi * d / 5) * g[di]
    # own part of Phi_1: mean + three own characters (coefficients are affine in g)
    mre = 0; mim = 0
    for di, d in enumerate(DELS):
        c = -(2 / 5) * rho ** d
        mre = mre + c.real * cp.sum(g[di]) / N; mim = mim + c.imag * cp.sum(g[di]) / N
    ore = mre * np.ones(N); oim = mim * np.ones(N)
    for k, chi in enumerate(OWN1):
        ch = rho ** (chi[0] * XS[:, 0] + chi[1] * XS[:, 1])
        ore = ore + own_re[k] * ch.real - own_im[k] * ch.imag
        oim = oim + own_re[k] * ch.imag + own_im[k] * ch.real
    vec = cp.hstack([re1 - ore, im1 - oim, re2, im2]) / np.sqrt(N)
    cons.append(cp.norm(vec, 2) <= Rpar)
    prob = cp.Problem(cp.Maximize(h), cons)
    _c[Tstar] = dict(prob=prob, h=h, mpar=mpar, sre=sre, sim=sim, M=Mpar, R=Rpar, g=g)
    return _c[Tstar]


def h_maxT(A, Tstar, M, phi, R):
    C = build(Tstar)
    C['mpar'].value = np.array([np.real(np.conj(A) * (1 - rho ** (-d))) for d in DELS])
    v = M * np.exp(-1j * phi)
    C['sre'].value = v.real; C['sim'].value = v.imag
    C['M'].value = M; C['R'].value = R
    C['prob'].solve(solver=cp.CLARABEL)
    if C['prob'].status not in ('optimal', 'optimal_inaccurate'):
        return -np.inf
    return C['h'].value


if __name__ == '__main__':
    import sys
    from scipy.optimize import minimize
    s = float(sys.argv[1]) if len(sys.argv) > 1 else 1.5
    G = Gamma(s); c1 = 1 - 3 ** (1 - s); cR = 3 ** (2 - 2 * s) / 4
    rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    for Tstar in [2, 0]:
        best = (-1, None)
        def ratio(v):
            alpha, M, phi, lR = v
            M = abs(M)
            if M > 3 ** (s - 1):
                return -1
            A = np.exp(1j * alpha)
            P = max(1 - c1 * M, 3 ** (1 - s))
            R = np.exp(lR)
            hv = h_maxT(A, Tstar, M, phi, R)
            return hv / (G * np.sqrt(P ** 2 + cR * R ** 2))
        for k in range(12):
            v0 = np.array([rng.normal(0, 0.1), rng.uniform(0, 1.7), rng.uniform(0, 2 * np.pi), rng.uniform(-3, 0.5)])
            res = minimize(lambda v: -ratio(v), v0, method='Nelder-Mead', options={'maxiter': 600, 'xatol': 1e-5, 'fatol': 1e-7})
            if -res.fun > best[0]:
                best = (-res.fun, res.x)
            print(f"T*={Tstar} start {k}: {-res.fun:.5f}  best {best[0]:.5f} at {np.round(best[1],4)}", flush=True)
        print(f"s={s} T*={OWN1[Tstar]}: sup ratio (max-T relaxation) >= {best[0]:.5f}")
