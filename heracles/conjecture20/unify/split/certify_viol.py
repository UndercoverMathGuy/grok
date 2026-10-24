"""Rigorous (exact rational margin + interval arithmetic) certificate that the mixed network stored in
viol_p5_n2_b2.npz (net 0; p = 5, n = 2, b = 2, width 6; weights rounded to 12 significant digits) violates
  (D_Q)      h <= nu^{1-s} (C_b P / sum_l Q_l)^s min_delta sum_l Q_l^s kappa_l(delta)
  (D_split)  h <= nu^{1-s} B_p (P / P_feat)^s min_delta sum_j P_j^s kappa_j(delta)   (live frequency-piece costs)
while its split theta_split satisfies both, and gamma(theta) < gamma_IF.  At n = 2 the price of a feature equals nu^{1-s}
for every prime (Note, Thm 'price of a feature at n = 2'), so (D_Q)/(D_split) with the true price are refuted for
general networks; for class-C networks (D_Q) is the Note's Theorem 'decoupling when few features are bought cheaply'.
Run: uv run --with numpy --with mpmath python certify_viol.py
"""
import itertools
from fractions import Fraction as Fr
import numpy as np
from mpmath import iv, mpf
iv.dps = 50
p, n, b = 5, 2, 2
nu = n + 1; s = Fr(nu, b); r = 1 / s
d = np.load('viol_p5_n2_b2.npz')
U = d['U0']; W = d['W0']
m = len(W)
q = lambda x: Fr(float('%.12g' % x))
Uq = [[[q(U[i, t, a]) for a in range(p)] for t in range(n)] for i in range(m)]
Wq = [[q(W[i, a]) for a in range(p)] for i in range(m)]


def margin(Uq, Wq):
    h = None
    for x in itertools.product(range(p), repeat=n):
        y = sum(x) % p
        z = [Uq[i][0][x[0]] * Uq[i][1][x[1]] for i in range(len(Wq))]
        f = [sum(Wq[i][c] * z[i] for i in range(len(Wq))) for c in range(p)]
        g = min(f[y] - f[c] for c in range(p) if c != y)
        h = g if h is None or g < h else h
    return h


def cost(Uq, Wq):
    return sum(sum(a * a for t in range(n) for a in Uq[i][t]) + sum(a * a for a in Wq[i]) for i in range(len(Wq)))


def ivf(x):
    return iv.mpf(x.numerator) / iv.mpf(x.denominator)


def fhat(v, k):   # interval DFT coefficient sum_a v(a) rho^{-k a}, returned as (re, im) intervals
    re = sum(ivf(v[a]) * iv.cos(2 * iv.pi * k * a / p) for a in range(p))
    im = -sum(ivf(v[a]) * iv.sin(2 * iv.pi * k * a / p) for a in range(p))
    return re, im


def cmul(a, b_):
    return (a[0] * b_[0] - a[1] * b_[1], a[0] * b_[1] + a[1] * b_[0])


def proj_energy(v, k):   # ||P_k v||^2 = (2/p)|vhat(k)|^2 for k != 0
    re, im = fhat(v, k)
    return 2 * (re ** 2 + im ** 2) / p


def analyse(Uq, Wq, tag):
    m = len(Wq)
    h = margin(Uq, Wq); P = cost(Uq, Wq)
    F = [1, 2]
    Q = {}; Pj = {}
    for l in F:
        Ql = iv.mpf(0); Pl = iv.mpf(0)
        for i in range(m):
            w_ = fhat(Wq[i], l); w_ = (w_[0], -w_[1])                # conj
            prod = cmul(w_, cmul(fhat(Uq[i][0], l), fhat(Uq[i][1], l)))
            mod2 = (prod[0] ** 2 + prod[1] ** 2) * (2 * iv.mpf(p) ** (-nu)) ** 2
            Ql += iv.exp(ivf(r) / 2 * iv.log(mod2))               # |beta_il|^r
            e = proj_energy(Wq[i], l) + proj_energy(Uq[i][0], l) + proj_energy(Uq[i][1], l)
            Pl += e                                                   # b = 2: piece cost = energy
        Q[l] = Ql; Pj[l] = Pl
    Bp = 2 * (2 * iv.mpf(p) * nu) ** (-iv.mpf(nu) / 2)
    Cb = iv.exp(iv.log(Bp) / ivf(s))
    kap = lambda l, dd: 1 - iv.cos(2 * iv.pi * l * dd / p)
    Ps = lambda x: iv.exp(ivf(s) * iv.log(x))
    covQ = [sum(Ps(Q[l]) * kap(l, dd) for l in F) for dd in range(1, p)]
    covP = [sum(Ps(Pj[l]) * kap(l, dd) for l in F) for dd in range(1, p)]
    minQ = min(covQ, key=lambda x: x.b); minP = min(covP, key=lambda x: x.b)   # rigorous min needs care: use lower of uppers
    # rigorous enclosure of the minimum: [min of lower ends, min of upper ends]
    minQ = iv.mpf([min(x.a for x in covQ), min(x.b for x in covQ)]); minP = iv.mpf([min(x.a for x in covP), min(x.b for x in covP)])
    nu1s = iv.exp((1 - ivf(s)) * iv.log(iv.mpf(nu)))
    Qs = sum(Q.values()); Pf = sum(Pj.values())
    rhsQ = nu1s * Ps(Cb * ivf(P) / Qs) * minQ
    rhsP = nu1s * Bp * Ps(ivf(P) / Pf) * minP
    Gam = iv.mpf('0.883883476483184405501055452631')   # Gamma_{3/2}(5) = 0.8838834764... (= 1/(2 sqrt 2) * 2.5)
    gIF = nu1s * Bp * Gam
    ratioQ = ivf(h) / rhsQ; ratioP = ivf(h) / rhsP; g = ivf(h) / Ps(ivf(P)) / gIF
    print(f"{tag}: h = {float(h):+.6e} (exact rational), P = {float(P):.6f}")
    print(f"   h / RHS(D_Q)     in [{float(ratioQ.a):.6f}, {float(ratioQ.b):.6f}]")
    print(f"   h / RHS(D_split) in [{float(ratioP.a):.6f}, {float(ratioP.b):.6f}]")
    print(f"   gamma/gamma_IF   in [{float(g.a):.6f}, {float(g.b):.6f}]")
    return ratioQ, ratioP, g


# Gamma_{3/2}(5): K* = 2, y = (t, t), t^{2/3} * 2 = 1 -> t = 2^{-3/2}; min_delta t (kappa_1 + kappa_2) = t * 2.5 (both deltas)
assert abs(2 ** -1.5 * 2.5 - 0.883883476483184405501055452631) < 1e-15
rQ, rP, g = analyse(Uq, Wq, 'theta (mixed, width 6)')
# split: project every factor on each frequency plane (exact rationals via real DFT basis is irrational -> use float split
# rounded to 12 digits; the split certificate only needs to show ratio < 1, done with the rounded split network itself)
def proj(v, k):
    v = np.array([float(x) for x in v]); a = np.arange(p)
    c = np.cos(2 * np.pi * k * a / p); s_ = np.sin(2 * np.pi * k * a / p)
    return c * (v @ c) / (c @ c) + s_ * (v @ s_) / (s_ @ s_)
Us, Ws = [], []
for i in range(m):
    for k in (1, 2):
        Us.append([[q(x) for x in proj(Uq[i][t], k)] for t in range(n)]); Ws.append([q(x) for x in proj(Wq[i], k)])
rQs, rPs, gs = analyse(Us, Ws, 'theta_split (class C, 12 pieces)')
print("CERTIFIED: theta violates (D_Q) and (D_split):", rQ.a > 1 and rP.a > 1, "; gamma(theta) < gamma_IF:", g.b < 1,
      "; split satisfies both:", rQs.b < 1 and rPs.b < 1)
