"""Rigorous verification of the simple two-feature bound at (7,2) on a range of s (interval B&B).

For networks with features in S = {1,2} (beta_3 = 0), A_j = |beta_j|, M_j = max_T |beta_{j,T}|, x_j = M_j / A_j:
  h <= sum_j K_j(mu) A_j - (7/2) min_d mu_d * max_j M_j          (free relaxation, rigid zero-leakage tests)
  3^{1-s} B_7 P^s >= ( sum_j (F_j A_j)^r )^s,   F_j = max(1 - c_1(s) x_j, 3^{1-s} max(1, x_j))     (c1sharp + trivial)
  gamma_IF = 3^{1-s} B_7 Gamma_s(7),  Gamma_s(7) = (7/2) 3^{-s}  (uniform allocation is optimal: sum_j kappa_j = 7/2).
Claim: ratio <= TARGET < 1 for all (A_1 : A_2), x_1, x_2 >= 0 and s in [s_lo, s_hi].
Variables: tau = A_2/(A_1+A_2) in [0,1];  x_j in [0, XMAX] plus the 'x_j = infinity' slab (F_j >= 1, M_j unused).
Prover's mu chosen per box by an LP at the box centre.  All box evaluations in interval arithmetic (iv.py).
Usage: python simple72_rig.py s_lo s_hi TARGET [s_step]
Rigour notes: mu is renormalised exactly (division by the interval 2(mu_1+mu_2+mu_3)); y^R is enclosed with 0 allowed;
the returned ratio bound is rounded upward; the x = infinity slab is valid because 3^{1-s} XMAX >= 1 (asserted).
"""
import sys
import numpy as np
from scipy.optimize import linprog
from iv import I, dn, up

s_lo = float(sys.argv[1]); s_hi = float(sys.argv[2]); TARGET = float(sys.argv[3])
SSTEP = float(sys.argv[4]) if len(sys.argv) > 4 else 0.005
p = 7
XMAX = 1.75


def cos2pi(k):
    """enclosure of cos(2 pi k/7) via mpmath interval arithmetic"""
    import mpmath
    mpmath.mp.prec = 80
    v = mpmath.iv.cos(2 * mpmath.iv.pi * k / 7)
    return I(dn(float(v.a)), up(float(v.b)))


KAP = {(j, d): I(1.0) - cos2pi((j * d) % 7) for j in (1, 2) for d in (1, 2, 3)}


def pw(x, e):
    """enclosure of x^e for x>0 interval and float exponent e via mpmath (outward)"""
    import mpmath
    mpmath.mp.prec = 80
    xl = float(x.lo); xh = float(x.hi)
    lo = mpmath.iv.mpf(xl) ** mpmath.iv.mpf(e) if xl > 0 else mpmath.iv.mpf(0)
    hi = mpmath.iv.mpf(xh) ** mpmath.iv.mpf(e)
    if e < 0:
        lo, hi = hi, lo
    return I(dn(float(mpmath.mpf(lo.a))), up(float(mpmath.mpf(hi.b))))


def box_ratio(mu, tau, x1, x2, inf1, inf2, s0, s1):
    """rigorous upper bound of the ratio over the box (tau, x1, x2 intervals) for all s in [s0, s1];
    every s-dependent quantity is enclosed by mpmath interval arithmetic with s ranging over [s0, s1]."""
    import mpmath
    iv = mpmath.iv
    iv.prec = 80
    S = iv.mpf([s0, s1])
    T = iv.mpf([float(tau.lo), float(tau.hi)])
    A1 = 1 - T; A2 = T
    kap = lambda j, d: 1 - iv.cos(2 * iv.pi * ((j * d) % 7) / 7)
    K1 = sum(2 * iv.mpf(float(mu[d - 1])) * kap(1, d) for d in (1, 2, 3))
    K2 = sum(2 * iv.mpf(float(mu[d - 1])) * kap(2, d) for d in (1, 2, 3))
    mmin = iv.mpf(float(min(mu)))
    X1 = iv.mpf([float(x1.lo), float(x1.hi)]); X2 = iv.mpf([float(x2.lo), float(x2.hi)])
    M1 = iv.mpf(0) if inf1 else A1 * X1
    M2 = iv.mpf(0) if inf2 else A2 * X2
    Mmax = iv.mpf([max(M1.a, M2.a), max(M1.b, M2.b)])
    num = K1 * A1 + K2 * A2 - 3.5 * mmin * Mmax
    t3 = iv.mpf(3) ** (1 - S)
    c1 = 1 - t3

    def F(X, inf):
        if inf:
            return iv.mpf(1)
        a = 1 - c1 * X
        b = t3 * iv.mpf([max(1, X.a), max(1, X.b)])
        return iv.mpf([max(a.a, b.a), max(a.b, b.b)])
    y1 = F(X1, inf1) * A1; y2 = F(X2, inf2) * A2
    R = 1 / S

    def powR(y):
        """enclosure of y^R for y >= 0 (y may touch 0), R in [1/s1, 1/s0]"""
        if y.a > 0:
            return y ** R
        return iv.mpf([0, (iv.mpf(y.b) ** R).b])
    nrm = (powR(y1) + powR(y2)) ** S
    den = iv.mpf(3.5) * iv.mpf(3) ** (-S) * nrm
    # exact normalisation of the dual probability: q_delta = mu_delta / Z, Z = 2 (mu_1 + mu_2 + mu_3)
    Z = 2 * (iv.mpf(float(mu[0])) + iv.mpf(float(mu[1])) + iv.mpf(float(mu[2])))
    num = num / Z
    if num.b <= 0:
        return up(float(num.b))
    q = num / den
    return up(float(mpmath.mpf(q.b)))


def choose_mu(tau, x1, x2, inf1, inf2):
    A1 = 1 - tau; A2 = tau
    M = max(0 if inf1 else A1 * x1, 0 if inf2 else A2 * x2)
    kap = lambda j, d: 1 - np.cos(2 * np.pi * j * d / 7)
    c = np.array([2 * (kap(1, d) * A1 + kap(2, d) * A2) for d in (1, 2, 3)] + [-3.5 * M])
    Aub = np.array([[-1, 0, 0, 1], [0, -1, 0, 1], [0, 0, -1, 1]])
    res = linprog(c, A_ub=Aub, b_ub=np.zeros(3), A_eq=np.array([[2, 2, 2, 0]]), b_eq=[1], bounds=[(0, None)] * 4, method='highs')
    mu = res.x[:3]
    mu = np.maximum(mu, 0); mu = mu / (2 * mu.sum())
    return mu


def run():
    import mpmath
    nbox = 0; worst = -np.inf; fails = 0
    # the 'x_j = infinity' slab needs F_j >= 3^{1-s} XMAX >= 1 on the whole s-range
    assert (mpmath.iv.mpf(3) ** (1 - mpmath.iv.mpf(s_hi)) * XMAX).a >= 1
    ss = np.linspace(s_lo, s_hi, int(round((s_hi - s_lo) / SSTEP)) + 1)
    for k in range(len(ss) - 1):
        s0, s1 = ss[k], ss[k + 1]
        stack = []
        for inf1 in (False, True):
            for inf2 in (False, True):
                stack.append((0.0, 1.0, 0.0, XMAX, 0.0, XMAX, inf1, inf2, 0))
        while stack:
            t0, t1, a0, a1, b0, b1, inf1, inf2, dep = stack.pop()
            tau = I(t0, t1); x1 = I(a0, a1); x2 = I(b0, b1)
            mu = choose_mu(0.5 * (t0 + t1), 0.5 * (a0 + a1), 0.5 * (b0 + b1), inf1, inf2)
            r = box_ratio(mu, tau, x1, x2, inf1, inf2, s0, s1)
            nbox += 1
            if r <= TARGET:
                worst = max(worst, r); continue
            if dep > 40:
                fails += 1; print("FAIL", s0, s1, t0, t1, a0, a1, b0, b1, inf1, inf2, r); continue
            # split widest
            w = [t1 - t0, 0 if inf1 else (a1 - a0) / 2, 0 if inf2 else (b1 - b0) / 2]
            i = int(np.argmax(w))
            if i == 0:
                m = 0.5 * (t0 + t1); stack += [(t0, m, a0, a1, b0, b1, inf1, inf2, dep + 1), (m, t1, a0, a1, b0, b1, inf1, inf2, dep + 1)]
            elif i == 1:
                m = 0.5 * (a0 + a1); stack += [(t0, t1, a0, m, b0, b1, inf1, inf2, dep + 1), (t0, t1, m, a1, b0, b1, inf1, inf2, dep + 1)]
            else:
                m = 0.5 * (b0 + b1); stack += [(t0, t1, a0, a1, b0, m, inf1, inf2, dep + 1), (t0, t1, a0, a1, m, b1, inf1, inf2, dep + 1)]
    print(f"s in [{s_lo},{s_hi}] TARGET={TARGET}: boxes={nbox} fails={fails} worst={worst:.5f}")


if __name__ == '__main__':
    run()
