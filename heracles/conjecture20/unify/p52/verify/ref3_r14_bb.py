"""Referee-3: independent re-verification of R14 (the (7,2) two-feature bound) with a DIFFERENT branch-and-bound.

Claim re-verified:  for all tau = A_2/(A_1+A_2) in [0,1], x_j = M_j/A_j in [0, inf], s in [s_lo, s_hi]:
   min_mu  [K_1(mu) A_1 + K_2(mu) A_2 - (7/2) min(mu) max_j M_j] / [(7/2) 3^{-s} (sum_j (F_j A_j)^{1/s})^s]  <=  TARGET,
   F_j = max(1 - c_1(s) x_j, 3^{1-s} max(1, x_j)),  c_1 = 1 - 3^{1-s},  K_j(mu) = sum_{d in Z_7^*} mu_d (1 - cos(2 pi j d/7)).
Differences from simple72_rig.py (deliberately):
  * s is a 4th B&B dimension (bisected adaptively), not fixed s-cells;
  * the x = infinity slab starts at XMAX = 2.0 (not 1.75), with F_j >= 3^{1-s} * 2 >= 1 checked rigorously;
  * bounds come from explicit monotonicity, evaluated with mpmath interval arithmetic at box corners:
      numerator <= max_{tau in {t0,t1}} (K1(1-tau) + K2 tau) - 3.5 m max(a0 (1-t1), b0 t0)
      F_j >= max(1 - c1(s1) x_hi, 3^{1-s1} max(1, x_lo))
      (sum y_j^r)^{1/r} decreasing in r  =>  den >= 3.5 * 3^{-s1} * (y1^{1/s0} + y2^{1/s0})^{s0},  y_j >= F_j^lo A_j^lo;
  * mu: own LP at the box centre, then rounded to dyadic rationals and renormalised exactly (Fractions).
Usage: python ref3_r14_bb.py s_lo s_hi TARGET
"""
import sys, time
from fractions import Fraction as Fr
import numpy as np
import mpmath
from scipy.optimize import linprog

iv = mpmath.iv
iv.prec = 110
s_lo = float(sys.argv[1]); s_hi = float(sys.argv[2]); TARGET = float(sys.argv[3])
XMAX = 2.0
def widen(x):
    # mpmath 1.3.0 iv.exp / ** are not strictly outward (observed excess <= 3.2e-28 relative); widen by 1e-20 relative
    return x * (1 + iv.mpf([-1, 1]) * iv.mpf('1e-20'))


def pow_iv(base, ex):
    # base > 0 interval, ex interval: exp(ex * log(base)), widened
    return widen(iv.exp(ex * iv.log(base)))


COS = {k: widen(iv.cos(2 * iv.pi * k / 7)) for k in range(7)}
KAP = {(j, d): 1 - COS[(j * d) % 7] for j in (1, 2) for d in (1, 2, 3)}
kapf = {(j, d): float(1 - np.cos(2 * np.pi * j * d / 7)) for j in (1, 2) for d in (1, 2, 3)}


def lp_mu(tau, x1, x2, inf1, inf2):
    A1, A2 = 1 - tau, tau
    M = max(0.0 if inf1 else A1 * x1, 0.0 if inf2 else A2 * x2)
    c = [2 * (kapf[(1, d)] * A1 + kapf[(2, d)] * A2) for d in (1, 2, 3)] + [-3.5 * M]
    res = linprog(c, A_ub=[[-1, 0, 0, 1], [0, -1, 0, 1], [0, 0, -1, 1]], b_ub=[0, 0, 0],
                  A_eq=[[2, 2, 2, 0]], b_eq=[1], bounds=[(0, None)] * 4, method='highs')
    mu = np.maximum(res.x[:3], 0)
    # dyadic rounding + exact renormalisation (any nonneg symmetric mu with 2 sum = 1 is admissible)
    q = [Fr(int(round(v * 2 ** 24)), 2 ** 24) for v in mu]
    Z = 2 * sum(q)
    return [v / Z for v in q]


def to_iv(fr):
    return iv.mpf(fr.numerator) / fr.denominator


def bound(mu, t0, t1, a0, a1, b0, b1, inf1, inf2, s0, s1):
    K1 = sum(2 * to_iv(mu[d - 1]) * KAP[(1, d)] for d in (1, 2, 3))
    K2 = sum(2 * to_iv(mu[d - 1]) * KAP[(2, d)] for d in (1, 2, 3))
    m = to_iv(min(mu))
    T0 = iv.mpf(t0); T1 = iv.mpf(t1)
    lin = iv.mpf([max((K1 * (1 - T0) + K2 * T0).a, (K1 * (1 - T1) + K2 * T1).a),
                  max((K1 * (1 - T0) + K2 * T0).b, (K1 * (1 - T1) + K2 * T1).b)])
    Mlo1 = iv.mpf(0) if inf1 else iv.mpf(a0) * (1 - T1)
    Mlo2 = iv.mpf(0) if inf2 else iv.mpf(b0) * T0
    Mlo = iv.mpf([max(Mlo1.a, Mlo2.a), max(Mlo1.a, Mlo2.a)])
    num_hi = (lin - 3.5 * m * Mlo).b
    if num_hi <= 0:
        return float(num_hi)
    S1 = iv.mpf(s1); S0 = iv.mpf(s0)
    th1 = pow_iv(iv.mpf(3), 1 - S1)                  # 3^{1-s} smallest at s1
    c1 = 1 - th1                                 # c_1(s) largest at s1

    def Flo(xl, xh, inf):
        if inf:
            return iv.mpf(1)
        f1 = 1 - c1 * iv.mpf(xh)
        f2 = th1 * iv.mpf(max(1.0, xl))
        return iv.mpf(max(f1.a, f2.a))
    y1 = Flo(a0, a1, inf1) * (1 - T1)
    y2 = Flo(b0, b1, inf2) * T0
    r0 = 1 / S0

    def pw(y):
        if y.a <= 0:
            return iv.mpf(0)
        return pow_iv(iv.mpf(y.a), r0)
    base = pw(y1) + pw(y2)
    if base.a <= 0:
        return np.inf
    nrm = pow_iv(iv.mpf(base.a), S0)
    den_lo = (iv.mpf(3.5) * pow_iv(iv.mpf(3), -S1) * nrm).a
    if den_lo <= 0:
        return np.inf
    return float((iv.mpf(num_hi) / iv.mpf(den_lo)).b) * (1 + 1e-15)


def run():
    # slab validity: F_j >= 3^{1-s} x >= 3^{1-s_hi} XMAX >= 1
    assert (pow_iv(iv.mpf(3), 1 - iv.mpf(s_hi)) * XMAX).a >= 1
    t_start = time.time()
    stack = []
    for inf1 in (False, True):
        for inf2 in (False, True):
            stack.append((0.0, 1.0, 0.0, XMAX, 0.0, XMAX, inf1, inf2, s_lo, s_hi, 0))
    nbox = 0; fails = 0; worst = -np.inf; maxdep = 0
    while stack:
        t0, t1, a0, a1, b0, b1, inf1, inf2, s0, s1, dep = stack.pop()
        maxdep = max(maxdep, dep)
        mu = lp_mu(0.5 * (t0 + t1), 0.5 * (a0 + a1), 0.5 * (b0 + b1), inf1, inf2)
        r = bound(mu, t0, t1, a0, a1, b0, b1, inf1, inf2, s0, s1)
        nbox += 1
        if r <= TARGET:
            worst = max(worst, r); continue
        if dep > 60:
            fails += 1; print("FAIL", t0, t1, a0, a1, b0, b1, inf1, inf2, s0, s1, r); continue
        # widths (s scaled by 10 so that s gets split when it matters)
        w = [t1 - t0, 0 if inf1 else (a1 - a0) / 2, 0 if inf2 else (b1 - b0) / 2, 10 * (s1 - s0)]
        i = int(np.argmax(w))
        if i == 0:
            m_ = 0.5 * (t0 + t1); kids = [(t0, m_, a0, a1, b0, b1), (m_, t1, a0, a1, b0, b1)]; ss = [(s0, s1)] * 2
        elif i == 1:
            m_ = 0.5 * (a0 + a1); kids = [(t0, t1, a0, m_, b0, b1), (t0, t1, m_, a1, b0, b1)]; ss = [(s0, s1)] * 2
        elif i == 2:
            m_ = 0.5 * (b0 + b1); kids = [(t0, t1, a0, a1, b0, m_), (t0, t1, a0, a1, m_, b1)]; ss = [(s0, s1)] * 2
        else:
            m_ = 0.5 * (s0 + s1); kids = [(t0, t1, a0, a1, b0, b1)] * 2; ss = [(s0, m_), (m_, s1)]
        for k, (sa, sb) in zip(kids, ss):
            stack.append(k + (inf1, inf2, sa, sb, dep + 1))
    print(f"REF3 R14 B&B: s in [{s_lo},{s_hi}] TARGET={TARGET}: boxes={nbox} fails={fails} worst={worst:.6f} "
          f"maxdepth={maxdep} time={time.time()-t_start:.0f}s")


if __name__ == '__main__':
    run()
