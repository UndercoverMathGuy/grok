"""Sum-of-singles certificates (moduli-only bound) at (5,2).

For symmetric mu and z_{delta,T} in C with sum_T |z_{delta,T}| <= mu_delta, the density
   25 q(x,delta) = mu_delta - sum_T Re(z_{delta,T} rho^{<chi_T,x>})  >= 0
gives, for every network with beta_2 = 0, |A| = 1, |beta_{1,T}| = m_T, non-own l2 <= R:
   h <= K(mu) - sum_T (m_T/2) Re(sum_d c_1(d) conj(z_{d,T})) + R * || l(z) ||,
   l(z) = (1/2) ( sum_d c_1(d) z_{d,T},  sum_d c_2(d) z_{d,T},  sum_d c_2(d) conj(z_{d,T}) )_{T}.
B*(m,R) = min over (mu, z) of the right side: an SOCP (phase of z_T WLOG aligned).
"""
import numpy as np
import cvxpy as cp
from p52lib import *

c1v = np.array([1 - rho ** (-d) for d in DELS])
c2v = np.array([1 - rho ** (-2 * d) for d in DELS])
kap = np.array([1 - np.cos(2 * np.pi * d / 5) for d in DELS])
_c = {}


def build():
    if 'p' in _c:
        return _c
    a = cp.Variable(nonneg=True)   # mu_{+-1} = a/2, mu_{+-2} = (1-a)/2
    zr = cp.Variable((4, 3)); zi = cp.Variable((4, 3))
    mpar = cp.Parameter(3, nonneg=True); Rpar = cp.Parameter(nonneg=True)
    mu = cp.hstack([a / 2, (1 - a) / 2, (1 - a) / 2, a / 2])
    cons = [a <= 1]
    for di in range(4):
        cons.append(cp.sum(cp.hstack([cp.norm(cp.hstack([zr[di, T], zi[di, T]])) for T in range(3)])) <= mu[di])
    K = kap @ mu
    det = 0
    leak = []
    for T in range(3):
        # Re( sum_d c1(d) conj(z_dT) ) = sum_d (Re c1 zr + Im c1 zi)
        det = det + mpar[T] / 2 * (c1v.real @ zr[:, T] + c1v.imag @ zi[:, T])
        # sum c1 z: re = Re c1 zr - Im c1 zi ; im = Re c1 zi + Im c1 zr
        for cv, conjz in [(c1v, False), (c2v, False), (c2v, True)]:
            s_ = -1 if conjz else 1
            re = cv.real @ zr[:, T] - cv.imag @ (s_ * zi[:, T])
            im = cv.real @ (s_ * zi[:, T]) + cv.imag @ zr[:, T]
            leak += [re / 2, im / 2]
    obj = K - det + Rpar * cp.norm(cp.hstack(leak))
    prob = cp.Problem(cp.Minimize(obj), cons)
    _c.update(dict(p=prob, a=a, zr=zr, zi=zi, m=mpar, R=Rpar))
    return _c


def Bstar(m, R, ret=False):
    C = build()
    C['m'].value = np.array(m, float); C['R'].value = R
    C['p'].solve(solver=cp.CLARABEL)
    if ret:
        return C['p'].value, C['a'].value, C['zr'].value + 1j * C['zi'].value
    return C['p'].value


if __name__ == '__main__':
    from twisted_pay import twisted_lb
    s = 1.5; G = Gamma(s); c1 = 1 - 3 ** (1 - s); cR = 3 ** (2 - 2 * s) / 4
    # quick test at the critical points (moduli only) with the phase-specific twisted payment as a reference
    for m in [(0.479, 0.466, 0.625), (0.652, 0.483, 0.496), (0.505, 0.503, 0.666), (1, 1, 1), (0, 0, 0), (0.5, 0.5, 0.5)]:
        best = 0
        for R in np.concatenate([[0], np.geomspace(1e-3, 5, 25)]):
            b = Bstar(m, R)
            M = max(m); RT = np.sqrt(sum(t * t for t in m))
            plb = max(1 - c1 * M, 1 - RT / (2 * np.sqrt(3)), 3 ** (1 - s) * max(1, M))
            best = max(best, b / (G * np.sqrt(plb ** 2 + cR * R ** 2)))
        print(m, "sum-of-singles ratio with P_lb:", round(best, 4), " B*(m,0)=", round(Bstar(m, 0), 4))
