"""cert_pt.py -- computer-assisted proof (interval arithmetic) of the n=3 single-feature payment.

THEOREM (certified for given currency, s*, c):  for every finite atom system (beta_i in C, xi_i in T^3) and
every s in [1, s*]:
        |A| - 4^{1-s} Q^s  <=  c * Mcur,
  Mcur = max_T |beta_T|                                              (currency 'pt'),
  Mcur = max( max_T |beta_T|, |b12|+|b13|, |b12|+|b23|, |b13|+|b23| ) (currency 'cf').

Reduction (proved in the report, Lemmas C1-C4):
  bonus/L <= a - F_r(Pbar, Tbar),   r = 1/s*,  a = |A|/L,  m = Mcur/L,  L = sum|beta_i|,
  F_r(P,T) = min{ M_r(1+lam) : lam in R^4, sum lam = 0, sum lam^2 <= 2P, sum lam^3 >= T, lam >= -1 },
  Pbar = kP m^2 + 2 G^2,  Tbar = -kT m^2 G,  G = G(a,m) = cos(max(arccos m - arccos a, 0)),
  (kP,kT) = (4,24) for 'pt', (3,12) for 'cf'.
So it suffices that   a - c m <= F_r(Pbar(a,m), Tbar(a,m))   for all (a,m) in [0,1]^2.

Certificate: branch and bound over (a,m)-boxes.  On a box, monotonicity gives
   LHS <= a_hi - c m_lo,   F_r(Pbar,Tbar) >= F_r(Pbar(a_lo,m_hi), Tbar(a_lo,m_hi)),
and F_r at a point is bounded below RIGOROUSLY by the closed-form candidate list (Lemma C4):
   C1: (-1^k, x^j, y^(4-k-j)), sum=0, sum of squares = 2P (quadratic);
   C2a: (x,x,y,z), 12x^3-6Px-T=0, y,z=-x+-sqrt(P-2x^2);  C2b: (-1,x,y,z), x,y,z roots of
        t^3 - t^2 + (1-P) t - (T+3-3P)/3.
Roots are enclosed by interval bisection with exact rational coefficients; every candidate not certainly
infeasible is kept (keeping extra candidates only lowers the bound).  Global fallback F_r >= 4^{1-1/r}.
Corner (a near 1, m near 0) is handled by the analytic Lemma C5 (checked here with intervals).
usage: python cert_pt.py <pt|cf> <s*> <c> [eta0 m0]
"""
import sys
import time
from fractions import Fraction as Fr
import mpmath
from mpmath import iv, mpf

mpmath.mp.prec = 80
iv.prec = 80

cur = sys.argv[1]; S_STAR = float(sys.argv[2]); C = float(sys.argv[3])
ETA0 = float(sys.argv[4]) if len(sys.argv) > 4 else 0.02
M0 = float(sys.argv[5]) if len(sys.argv) > 5 else 0.05
kP, kT = (4, 24) if cur == 'pt' else (3, 12)
S_IV = iv.mpf(S_STAR)
R_IV = 1 / S_IV
THETA_LOW = (iv.mpf(4) ** (1 - S_IV)).a          # lower bound of 4^{1-s*} = global min of M_r


def ivF(x):  # exact Fraction -> interval
    return iv.mpf(x.numerator) / iv.mpf(x.denominator)


def mpf2fr(x):
    """exact Fraction of an mpmath mpf (or interval endpoint)"""
    x = mpmath.mpf(x)
    sign, man, exp, bc = x._mpf_
    if man == 0:
        return Fr(0)
    v = Fr(int(man)) * (Fr(2) ** exp)
    return -v if sign else v


def Mr_low(lams):
    """rigorous lower bound of M_r(1+lam) for interval components lam."""
    tot = iv.mpf(0)
    for l in lams:
        t_lo = (1 + l).a
        if t_lo <= 0:
            continue
        tot = tot + iv.mpf(t_lo) ** R_IV
    base = tot / 4
    if base.a <= 0:
        return mpf(0)
    return (iv.mpf(base.a) ** (1 / R_IV)).a


def feasible_maybe(lams, Tf):
    lo_min = min(l.b for l in lams)
    if lo_min < -1:          # certainly some component < -1
        return False
    s3 = sum(l ** 3 for l in lams)
    if s3.b < ivF(Tf).a:     # certainly violates sum lam^3 >= T
        return False
    return True


def poly_iv(coefs, X):
    acc = iv.mpf(0)
    for cf in coefs:
        acc = acc * X + cf
    return acc


def peval(coefs_frac, x):
    acc = Fr(0)
    for cf in coefs_frac:
        acc = acc * x + cf
    return acc


def root_clusters(coefs_frac, lo=Fr(-1), hi=Fr(3), width=Fr(1, 2 ** 40)):
    """Enclose ALL real roots of a cubic with exact rational coefficients.
    Fast path: the exact discriminant gives the number of distinct real roots (3 if >0, 1 if <0);
    float roots are bracketed by exact sign changes; if as many disjoint brackets as real roots are
    certified, they enclose all real roots.  Otherwise fall back to interval bisection on [lo,hi]
    (then only roots in [lo,hi] are returned, which is all that is needed: feasible lam lie in [-1,3])."""
    import numpy as np
    c3, c2, c1, c0 = coefs_frac
    disc = 18 * c3 * c2 * c1 * c0 - 4 * c2 ** 3 * c0 + c2 ** 2 * c1 ** 2 - 4 * c3 * c1 ** 3 - 27 * c3 ** 2 * c0 ** 2
    if disc != 0:
        nreal = 3 if disc > 0 else 1
        approx = np.roots([float(c) for c in coefs_frac])
        xs = sorted(float(z.real) for z in approx if abs(z.imag) <= 1e-6 * (1 + abs(z)))
        brackets = []
        for x in xs:
            for d in (1e-13, 1e-11, 1e-9, 1e-7):
                da = Fr(x) - Fr(d) * (1 + abs(Fr(x))); db = Fr(x) + Fr(d) * (1 + abs(Fr(x)))
                if peval(coefs_frac, da) * peval(coefs_frac, db) < 0:
                    brackets.append((da, db)); break
        brackets.sort()
        ok = len(brackets) == nreal and all(brackets[i][1] < brackets[i + 1][0] for i in range(len(brackets) - 1))
        if ok:
            return [iv.mpf([ivF(a).a, ivF(b).b]) for a, b in brackets]
    coefs = [ivF(c) for c in coefs_frac]
    out = []
    stack = [(lo, hi)]
    while stack:
        a, b = stack.pop()
        X = iv.mpf([ivF(a).a, ivF(b).b])
        val = poly_iv(coefs, X)
        if val.a > 0 or val.b < 0:
            continue
        if b - a <= width:
            out.append(X)
            continue
        mid = (a + b) / 2
        stack.append((a, mid)); stack.append((mid, b))
    # merge adjacent intervals
    out.sort(key=lambda I: I.a)
    merged = []
    for I in out:
        if merged and I.a <= merged[-1].b:
            merged[-1] = iv.mpf([merged[-1].a, max(merged[-1].b, I.b)])
        else:
            merged.append(I)
    return merged


def inner_low(Pf, Tf):
    """rigorous lower bound for F_r(P,T); P, T exact Fractions."""
    if Pf >= 6:
        return THETA_LOW
    best = None
    P_iv, T_iv = ivF(Pf), ivF(Tf)
    cands = []
    # C1 patterns
    for k in range(0, 4):
        for j in range(1, 4 - k + 1):
            n2 = 4 - k - j
            if n2 == 0:
                x = Fr(k, 4 - k)
                if k + (4 - k) * x * x <= 2 * Pf:
                    cands.append([iv.mpf(-1)] * k + [ivF(x)] * j)
                continue
            qa = Fr(j) + Fr(j * j, n2); qb = Fr(-2 * k * j, n2); qc = Fr(k) + Fr(k * k, n2) - 2 * Pf
            disc = qb * qb - 4 * qa * qc
            if disc < 0:
                continue
            sq = iv.sqrt(ivF(disc))
            for sgn in (1, -1):
                X = (-ivF(qb) + sgn * sq) / (2 * ivF(qa))
                Y = (k - j * X) / n2
                cands.append([iv.mpf(-1)] * k + [X] * j + [Y] * n2)
    # C2a: (x,x,y,z)
    for X in root_clusters([Fr(12), Fr(0), -6 * Pf, -Tf]):
        Sv = P_iv - 2 * X * X
        if Sv.b < 0:
            continue
        sq = iv.sqrt(iv.mpf([max(Sv.a, mpf(0)), Sv.b]))
        cands.append([X, X, -X + sq, -X - sq])
    # C2b: (-1,x,y,z)
    cb = [Fr(1), Fr(-1), 1 - Pf, -(Tf + 3 - 3 * Pf) / 3]
    db_ = 18 * cb[0] * cb[1] * cb[2] * cb[3] - 4 * cb[1] ** 3 * cb[3] + cb[1] ** 2 * cb[2] ** 2 \
        - 4 * cb[0] * cb[2] ** 3 - 27 * cb[0] ** 2 * cb[3] ** 2
    cl = root_clusters(cb) if db_ >= 0 else []   # disc<0: only one real root, no real triple exists
    for i in range(len(cl)):
        for j in range(i, len(cl)):
            for k in range(j, len(cl)):
                trip = [cl[i], cl[j], cl[k]]
                ssum = trip[0] + trip[1] + trip[2]
                if ssum.a > 1 or ssum.b < 1:
                    continue
                cands.append([iv.mpf(-1)] + trip)
    # float pre-screen: evaluate rigorously in increasing order of the float estimate; a candidate whose
    # float estimate exceeds the current rigorous best by > 1e-9 cannot lower the minimum (rigorous low >=
    # float estimate - 1e-12 for these well-conditioned evaluations), so it is skipped.
    rf = float(mpmath.mpf(R_IV.a))
    est = []
    for lam in cands:
        if not feasible_maybe(lam, Tf):
            continue
        mids = [float(mpmath.mpf(l.mid)) for l in lam]
        tot = sum(max(1 + x, 0.0) ** rf for x in mids)
        est.append(((tot / 4) ** (1 / rf), lam))
    est.sort(key=lambda t: t[0])
    for e, lam in est:
        if best is not None and e - 1e-9 > float(best):
            break
        v = Mr_low(lam)
        best = v if best is None else min(best, v)
    if best is None:
        best = THETA_LOW
    return max(best, THETA_LOW)


def G_up(a_lo, m_hi):
    """upper bound of G over the box (G increasing in m, decreasing in a)."""
    if m_hi >= a_lo:
        return iv.mpf(1)
    A, M = ivF(a_lo), ivF(m_hi)
    g = A * M + iv.sqrt((1 - A * A) * (1 - M * M))
    return iv.mpf([g.a, min(g.b, mpf(1))])


def corner_ok():
    """Lemma C5: for 1-a <= ETA0 and m <= M0:  a - F_r >= ... gives a - c m <= F_r."""
    e0, m0 = iv.mpf(ETA0), iv.mpf(M0)
    Pc = (kP + 2) * m0 * m0 + 4 * iv.sqrt(2 * e0) * m0 + 4 * e0
    t0 = 1 - iv.sqrt(iv.mpf(1.5) * Pc)
    if t0.a <= 0:
        return False, None
    K0 = t0 ** (R_IV - 2)
    A = 1 - (1 - R_IV) * K0
    if A.a <= 0:
        return False, None
    Cc = (1 - R_IV) * K0 * (kP + 2) / 4 + (1 - R_IV) ** 2 * K0 ** 2 / (2 * A)
    return (m0 * Cc).b <= C, (m0 * Cc).b


def run(N0=32, min_w=Fr(1, 2 ** 16)):
    t0 = time.time()
    cok, cval = corner_ok()
    print(f"[{cur}] s*={S_STAR} c={C}  corner lemma (eta0={ETA0}, m0={M0}): ok={cok} (m0*Cc={cval})", flush=True)
    if not cok:
        print("FAIL: corner lemma"); return
    stack = [(Fr(i, N0), Fr(i + 1, N0), Fr(j, N0), Fr(j + 1, N0)) for i in range(N0) for j in range(N0)]
    npass = ntriv = ncorner = 0
    worst_margin = None
    fails = 0
    cF = Fr(C)
    THETA_FR = mpf2fr(THETA_LOW)
    eta0F, m0F = Fr(ETA0), Fr(M0)
    it = 0
    while stack:
        it += 1
        if it % 2000 == 0:
            print(f"  ... processed {it} boxes, stack {len(stack)}, passed {npass}, time {time.time()-t0:.0f}s", flush=True)
        a_lo, a_hi, m_lo, m_hi = stack.pop()
        lhs = a_hi - cF * m_lo
        if lhs <= THETA_FR:
            ntriv += 1; continue
        if a_lo >= 1 - eta0F and m_hi <= m0F:
            ncorner += 1; continue
        G = G_up(a_lo, m_hi)
        Pst = mpf2fr((kP * ivF(m_hi) ** 2 + 2 * G * G).b)
        Tst = mpf2fr((-kT * ivF(m_hi) ** 2 * G).a)
        Fl = mpf2fr(inner_low(Pst, Tst))
        if lhs <= Fl:
            npass += 1
            marg = Fl - lhs
            worst_margin = marg if worst_margin is None else min(worst_margin, marg)
            continue
        if (a_hi - a_lo) <= min_w and (m_hi - m_lo) <= min_w:
            fails += 1
            print(f"FAIL box a=[{float(a_lo):.6f},{float(a_hi):.6f}] m=[{float(m_lo):.6f},{float(m_hi):.6f}] lhs={float(lhs):.6f} F={float(Fl):.6f}", flush=True)
            if fails > 20:
                break
            continue
        if (a_hi - a_lo) >= (m_hi - m_lo):
            mid = (a_lo + a_hi) / 2
            stack.append((a_lo, mid, m_lo, m_hi)); stack.append((mid, a_hi, m_lo, m_hi))
        else:
            mid = (m_lo + m_hi) / 2
            stack.append((a_lo, a_hi, m_lo, mid)); stack.append((a_lo, a_hi, mid, m_hi))
    print(f"[{cur}] s*={S_STAR} c={C}: boxes passed={npass} trivial={ntriv} corner={ncorner} fails={fails} "
          f"min certified margin={float(worst_margin) if worst_margin is not None else None:.3e}  time={time.time()-t0:.0f}s", flush=True)
    print("DONE fail=%d" % fails)


if __name__ == '__main__':
    run()
