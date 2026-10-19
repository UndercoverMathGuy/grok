"""Polish the E2b non-aligned example (printed to 8 digits) to an exact IF configuration and recheck H < log 3."""
import itertools, math
import numpy as np
import mpmath as mp
from scipy.optimize import least_squares
mp.mp.dps = 40
mod = np.array([0.49144832, 0.10252906, 0.40602261]); arg = np.array([4.59829238, 3.03313222, 1.07566332])
xa = np.array([[6.24048247, 2.48936183], [5.18952748, 1.43840683], [0.21132144, 2.7433861]])
Ts = [(0,), (1,), (0, 1)]
def res(x):
    md = x[:3]; ag = x[3:6]; ph = x[6:].reshape(3, 2)
    beta = md * np.exp(1j * ag); xi = np.exp(1j * ph)
    R = []
    for T in Ts:
        v = (beta * np.prod(xi[:, list(T)], axis=1)).sum(); R += [v.real, v.imag]
    return np.array(R + [md.sum() - 1])
x0 = np.concatenate([mod, arg, xa.ravel()])
x = least_squares(res, x0, xtol=1e-15, ftol=1e-15, gtol=1e-15).x
# Newton refinement in mpmath on the 7 equations with the 12 unknowns: fix 5 of them (min-norm step)
xm = [mp.mpf(v) for v in x]
def resm(xm):
    md = xm[:3]; ag = xm[3:6]; ph = xm[6:]
    beta = [md[l] * mp.expj(ag[l]) for l in range(3)]
    R = []
    for T in Ts:
        v = mp.fsum(beta[l] * mp.fprod([mp.expj(ph[2 * l + t]) for t in T]) for l in range(3)); R += [mp.re(v), mp.im(v)]
    return mp.matrix(R + [mp.fsum(md) - 1])
for it in range(8):
    r = resm(xm)
    J = mp.matrix(7, 12)
    for k in range(12):
        e = [mp.mpf(0)] * 12; e[k] = mp.mpf('1e-25')
        rp = resm([xm[i] + e[i] for i in range(12)])
        for q in range(7):
            J[q, k] = (rp[q] - r[q]) / mp.mpf('1e-25')
    # min-norm Newton step: dx = -J^T (J J^T)^{-1} r
    dx = -(J.T * mp.inverse(J * J.T) * r)
    xm = [xm[i] + dx[i] for i in range(12)]
r = resm(xm)
md = xm[:3]
w = [v / mp.fsum(md) for v in md]
H = -mp.fsum(v * mp.log(v) for v in w)
print("max residual (40 digits):", mp.nstr(max(abs(v) for v in r), 5))
print("moduli:", [mp.nstr(v, 10) for v in md], " min modulus > 0:", min(md) > 0)
print("H =", mp.nstr(H, 12), " log 3 =", mp.nstr(mp.log(3), 12), " H < log 3:", H < mp.log(3))
print("|beta| =", mp.nstr(abs(mp.fsum(md[l] * mp.expj(xm[3 + l]) for l in range(3))), 8))
print("shift from printed point:", np.abs(np.array([float(v) for v in xm]) - x0).max())
