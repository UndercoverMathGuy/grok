"""High-precision (50-digit) evaluation, from the 12-digit rational weights certified in split/certify_viol.py, of
 (i)  the cost-share entropy H(w_s) of the (5,2,2) violator theta_v (s = 3/2) and its gamma/gamma_IF (exact margin);
 (ii) the least entropy of an IF-block decoupled network reaching the same gamma level (solved exactly in t = y2/y1);
 (iii) its clean-amplitude profile and the pinned-profile decoupled value D_C = pi B_p M(yhat)/||yhat||_r.
pi_{2,b}(5) = 3^{1-s} (Note Thm q1n2).  Run: uv run --with numpy --with mpmath python certify_E.py"""
import itertools
from fractions import Fraction as Fr
import numpy as np, mpmath as mp
mp.mp.dps = 50
p, n = 5, 2; nu = 3; s = mp.mpf(3) / 2; r = 1 / s
d = np.load('../split/viol_p5_n2_b2.npz'); U = d['U0']; W = d['W0']; m = len(W)
q = lambda x: Fr(float('%.12g' % x))
Uq = [[[q(U[i, t, a]) for a in range(p)] for t in range(n)] for i in range(m)]
Wq = [[q(W[i, a]) for a in range(p)] for i in range(m)]
# exact margin
h = None
for x in itertools.product(range(p), repeat=n):
    y = sum(x) % p
    z = [Uq[i][0][x[0]] * Uq[i][1][x[1]] for i in range(m)]
    f = [sum(Wq[i][c] * z[i] for i in range(m)) for c in range(p)]
    g = min(f[y] - f[c] for c in range(p) if c != y)
    h = g if h is None or g < h else h
S = [sum(a * a for t in range(n) for a in Uq[i][t]) + sum(a * a for a in Wq[i]) for i in range(m)]   # ||omega_i||^2
P = sum(S)                                               # b = 2 cost
kap = lambda j, dl: 1 - mp.cos(2 * mp.pi * j * dl / p)
ka, kb = kap(1, 1), kap(1, 2)
Gam = (mp.mpf(5) / 2) * 2 ** (-s)
Bp = 2 * (2 * p * nu) ** (-mp.mpf(nu) / 2)
gIF = 3 ** (1 - s) * Bp * Gam
gam = mp.mpf(h.numerator) / h.denominator / mp.mpf(P.numerator / P.denominator) ** s
lev = gam / gIF
c = [mp.mpf(x.numerator) / x.denominator for x in S]     # ||omega||^b with b = 2
w = [x / sum(c) for x in c if x > 0]
H = -sum(x * mp.log(x) for x in w)
print('theta_v: gamma/gamma_IF = %s   H(w_s) = %s' % (mp.nstr(lev, 12), mp.nstr(H, 12)))


def ratio(t):   # IF two-block value M(1,t)/||(1,t)||_r / Gamma_s
    M = min(ka + t * kb, kb + t * ka)
    return M / (1 + t ** r) ** s / Gam


t0 = mp.findroot(lambda t: ratio(t) - lev, (mp.mpf('0.05'), mp.mpf('0.6')), solver='anderson')
z2 = t0 ** r / (1 + t0 ** r)
Hdec = mp.log(3) + (-(1 - z2) * mp.log(1 - z2) - z2 * mp.log(z2))
print('IF-block decoupled: level %.6f first reached at y2/y1 = %s with entropy log3 + H(z) = %s' %
      (float(lev), mp.nstr(t0, 10), mp.nstr(Hdec, 12)))
print('   (on t in (0,t0) the two-block value is below the level: ratio(t0/2) = %s, ratio(0) = kappa_a/Gamma = %s)'
      % (mp.nstr(ratio(t0 / 2), 8), mp.nstr(ratio(0), 8)))
# clean amplitudes beta_j = 2 p^{-nu} sum_i conj(what_i(j)) prod_t uhat_it(j)
rho = lambda k: [mp.exp(-2j * mp.pi * k * a / p) for a in range(p)]
def hat(v, k):
    return sum(mp.mpf(v[a].numerator) / v[a].denominator * rho(k)[a] for a in range(p))
beta = []
for j in (1, 2):
    bj = sum(mp.conj(hat(Wq[i], j)) * hat(Uq[i][0], j) * hat(Uq[i][1], j) for i in range(m)) * 2 * mp.mpf(p) ** (-nu)
    beta.append(abs(bj))
yh = [b_ / (beta[0] + beta[1]) for b_ in beta]
M = min(ka * yh[0] + kb * yh[1], kb * yh[0] + ka * yh[1])
DC = M / (yh[0] ** r + yh[1] ** r) ** s / Gam
print('profile yhat = (%s, %s);  D_C/gamma_IF = %s  <  gamma(theta_v)/gamma_IF = %s  (excess %s)'
      % (mp.nstr(yh[0], 8), mp.nstr(yh[1], 8), mp.nstr(DC, 10), mp.nstr(lev, 10), mp.nstr(lev / DC - 1, 5)))
# Unimodality (exact): on t in [0,1], R(t) = (ka + t kb)(1+t^{2/3})^{-3/2}/Gamma; with u = t^{1/3},
# d log R/dt = kb/(ka + u^3 kb) - 1/(u (1+u^2)) vanishes only at u = ka/kb.  So R decreases on [0,(ka/kb)^3] and increases
# after; R(0) < level, hence R < level on [0,t0) and the least entropy over IF-block decoupled networks at that level is
# log 3 + H(z(t0)) (H(z(t)) increases on [0,1]; t > 1 is the mirror case).
print('turning point (ka/kb)^3 = %s, R there = %s' % (mp.nstr((ka / kb) ** 3, 8), mp.nstr(ratio((ka / kb) ** 3), 8)))
