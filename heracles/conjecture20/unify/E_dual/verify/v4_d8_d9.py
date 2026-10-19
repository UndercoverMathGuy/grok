"""Referee re-computation of the D8 / D9 counterexample data from the SPLIT violator theta_v at (p,n,b)=(5,2,2), s=3/2.
Independent code: exact rational margin and cost (12-significant-digit rationalisation of the stored weights),
60-digit clean coefficients by direct DFT, exact two-block IF frontier."""
import itertools
from fractions import Fraction as Fr
import numpy as np, mpmath as mp
mp.mp.dps = 60
p, n, nu = 5, 2, 3
s = mp.mpf(3) / 2; r = 1 / s
d = np.load('/home/user/grok/heracles/conjecture20/unify/split/viol_p5_n2_b2.npz')
print('npz keys:', sorted(d.keys()))
for key in ['0', '1', '2']:
    U = d['U' + key]; W = d['W' + key]; m = len(W)
    q = lambda x: Fr(repr(float('%.12g' % x)))
    Uq = [[[q(U[i, t, a]) for a in range(p)] for t in range(n)] for i in range(m)]
    Wq = [[q(W[i, a]) for a in range(p)] for i in range(m)]
    h = min(min(sum(Wq[i][(x1 + x2) % p] * Uq[i][0][x1] * Uq[i][1][x2] for i in range(m))
                - sum(Wq[i][c] * Uq[i][0][x1] * Uq[i][1][x2] for i in range(m))
                for c in range(p) if c != (x1 + x2) % p) for x1 in range(p) for x2 in range(p))
    S = [sum(a * a for t in range(n) for a in Uq[i][t]) + sum(a * a for a in Wq[i]) for i in range(m)]
    P = sum(S)
    tomp = lambda fr: mp.mpf(fr.numerator) / fr.denominator
    gam = tomp(h) / tomp(P) ** s
    ka = 1 - mp.cos(2 * mp.pi / p); kb = 1 - mp.cos(4 * mp.pi / p)
    Gam = (ka + kb) * 2 ** (-s)                       # K* = 2, y* uniform at s=3/2 (< 1.8552)
    Bp = 2 * (2 * p * nu) ** (-mp.mpf(nu) / 2)
    gIF = 3 ** (1 - s) * Bp * Gam
    lev = gam / gIF
    w = [tomp(x) / tomp(P) for x in S]               # b = 2: cost shares ||omega_i||^2 / P
    H = -sum(x * mp.log(x) for x in w if x > 0)
    # clean coefficients beta_j = 2 p^{-nu} sum_{x,c} f(x)[c] e^{-2 pi i (j x1 + j x2 - j c)/p}
    beta = []
    for j in (1, 2):
        tot = mp.mpc(0)
        for x1 in range(p):
            for x2 in range(p):
                for c in range(p):
                    fv = sum(tomp(Wq[i][c]) * tomp(Uq[i][0][x1]) * tomp(Uq[i][1][x2]) for i in range(m))
                    tot += fv * mp.expj(-2 * mp.pi * (j * x1 + j * x2 - j * c) / p)
        beta.append(2 * mp.mpf(p) ** (-nu) * tot)
    A = [abs(b_) for b_ in beta]; yh = [A[0] / (A[0] + A[1]), A[1] / (A[0] + A[1])]
    Mv = min(ka * yh[0] + kb * yh[1], kb * yh[0] + ka * yh[1])
    DC = Mv / (yh[0] ** r + yh[1] ** r) ** s / Gam
    print('violator %s: m=%d h>0? %s  gamma/gIF = %s  H(w_s) = %s  profile = (%s, %s)  D_{yhat}/gIF = %s  excess = %s'
          % (key, m, h > 0, mp.nstr(lev, 12), mp.nstr(H, 10), mp.nstr(yh[0], 8), mp.nstr(yh[1], 8), mp.nstr(DC, 10),
             mp.nstr(lev / DC - 1, 6)))
    if key == '0':
        lev0 = lev; H0 = H

# two-block IF frontier: R(t) = M(1,t)/(1+t^r)^s/Gamma on t in [0,1]; least entropy log3 + H(z(t)) reaching lev0
R = lambda t: min(ka + t * kb, kb + t * ka) / (1 + t ** r) ** s / Gam
ts = [mp.mpf(k) / 20000 for k in range(20001)]
vals = [R(t) for t in ts]
first = next(t for t, v in zip(ts, vals) if v >= lev0)
t0 = mp.findroot(lambda t: R(t) - lev0, first)
z2 = t0 ** r / (1 + t0 ** r)
Hdec = mp.log(3) + (-(1 - z2) * mp.log(1 - z2) - z2 * mp.log(z2))
imin = int(np.argmin([float(v) for v in vals]))
print('R(0) = %s ; argmin R on grid t = %s (R = %s) ; first t with R >= level: t0 = %s ; eta_dec = %s vs H(theta_v) = %s'
      % (mp.nstr(R(0), 10), mp.nstr(ts[imin], 6), mp.nstr(vals[imin], 8), mp.nstr(t0, 10), mp.nstr(Hdec, 10), mp.nstr(H0, 10)))
# monotonicity of R after the minimum and below level before t0 (grid)
print('grid: all R(t) < level for t < t0: %s ; R increasing after argmin: %s'
      % (all(v < lev0 for t, v in zip(ts, vals) if t < t0 - mp.mpf('1e-4')),
         all(vals[k + 1] >= vals[k] for k in range(imin, len(vals) - 1))))
# mirror t>1 equivalent: R(1/t) with roles swapped gives the same frontier (symmetry of kappa at p=5)
