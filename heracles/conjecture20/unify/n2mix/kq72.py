"""Neuron-level paid norm of a test at (7,2), L = {1,2}:
   K(q) = sup_omega tau_q^paid(omega) / sqrt(e(omega)^2 - b_L(omega)^2),
   tau_q^paid(omega) = Re sum_{paid (j,k)} m(j;k) T_j(k + j 1).
Compare with the l2 ledger constant 2 n(q) and the family ledger constant 2 n''(q).
"""
import sys
import numpy as np
from scipy.optimize import minimize
from socp72 import *

p = 7
s = float(sys.argv[1]) if len(sys.argv) > 1 else 1.5
r = 1 / s


def Tvec(q):
    """T_j(chi) for all (jidx, chi1, chi2) as array [3,7,7]"""
    T = np.zeros((3, p, p), dtype=complex)
    for (ji, c1, c2), row in ROWS.items():
        T[ji, c1, c2] = row @ q
    return T


PAIDMASK = np.zeros((3, p, p), dtype=bool)
for k, v in LAB.items():
    if v == 'paid':
        PAIDMASK[k] = True


def tau_paid(omega, T):
    u1, u2, w = omega[:7], omega[7:14], omega[14:21]
    U1, U2, Wh = np.fft.fft(u1), np.fft.fft(u2), np.fft.fft(w)
    tot = 0.0
    for ji, j in enumerate(G.F):
        # modes m(j;k) at chi = k + j: T_j(chi) -> index chi = k + j
        Tj = np.roll(np.roll(T[ji] * PAIDMASK[ji], -j, axis=0), -j, axis=1)   # Tj[k1,k2] = T_j(k + j1)
        tot += np.real(2 * p ** (-3) * Wh[j] * (U1 @ Tj @ U2))
    return tot


def eb(omega):
    u1, u2, w = omega[:7], omega[7:14], omega[14:21]
    U1, U2, Wh = np.fft.fft(u1), np.fft.fft(u2), np.fft.fft(w)
    e = G.Bp * (u1 @ u1 + u2 @ u2 + w @ w) ** 1.5
    b1 = abs(2 * p ** (-3) * np.conj(Wh[1]) * U1[1] * U2[1])
    b2 = abs(2 * p ** (-3) * np.conj(Wh[2]) * U1[2] * U2[2])
    bL = (b1 ** (2 / 3) + b2 ** (2 / 3)) ** 1.5
    return e, bL


def Kq(T, starts=60, seed=0):
    rng = np.random.default_rng(seed)
    best = (0, None)
    def f(om):
        e, bL = eb(om)
        d = e * e - bL * bL
        if d <= 1e-300:
            return 0.0
        return -tau_paid(om, T) / np.sqrt(d)
    for k in range(starts):
        om = rng.normal(size=21)
        if k % 3 == 1:   # near an L-neuron with a small paid part
            a = np.arange(7)
            om = np.concatenate([np.cos(2*np.pi*a/7 + rng.uniform(0, 6)) + 0.3*rng.normal(size=7),
                                 np.cos(2*np.pi*a/7 + rng.uniform(0, 6)) + 0.3*rng.normal(size=7),
                                 np.cos(2*np.pi*a/7 + rng.uniform(0, 6)) + 0.3*rng.normal(size=7)])
        res = minimize(f, om, method='BFGS', options={'maxiter': 3000, 'gtol': 1e-10})
        if -res.fun > best[0]:
            best = (-res.fun, res.x)
    return best


if __name__ == '__main__':
    bf = BoundFam(s); bd = Bound(s)
    A2 = 2 * np.cos(np.pi / 7)
    xi = np.exp(1j * np.array([[0.3, 1.1]]))
    a1, bt1 = own_from_atoms(np.array([1.0 + 0j]), xi)
    F = ((3 ** (1 - s)) ** r + A2 ** r) ** s
    c = 3.0 ** (2 - 2 * s) / 4
    for name, b in (('l2', bd), ('family', bf)):
        v, q = b([a1, A2], [bt1, [0, 0, 0]], F)
        T = Tvec(q)
        a = own_functional([a1, A2], [bt1, [0, 0, 0]]) @ q
        n2 = np.sqrt(np.sum(np.abs(T[PAIDMASK]) ** 2))
        K, om = Kq(T)
        bnd_K = np.sqrt(max(a, 0) ** 2 / F ** 2 + K ** 2 / (4 * c)) / Gamma7(s)
        print(f"{name}-optimal test: bound {v:.5f}; a/F = {a/F:.5f}; 2n = {2*n2:.4f}; K(q) = {K:.4f}; bound with K: {bnd_K:.5f}")
