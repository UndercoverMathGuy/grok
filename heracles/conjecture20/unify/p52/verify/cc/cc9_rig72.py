"""Referee: an INDEPENDENT rigorous re-certification of the computational step of R14 (my own code, mpmath intervals).

Claim to certify: for s in [S0, S1] and all tau in [0,1], x_1, x_2 in [0, 1.75] or 'infinite' (x_j > 1.75):
    max(N*, 0) / (3.5 * 3^{-s} * ((F_1 A_1)^{1/s} + (F_2 A_2)^{1/s})^s) <= TARGET,
    A_1 = 1 - tau, A_2 = tau, M = max(A_1 x_1, A_2 x_2) (a slab contributes 0 to M and F_j >= 1 there),
    N* = min( min_d [kappa_1(d) A_1 + kappa_2(d) A_2], 7/6 - (7/12) M )      (exact LP optimum over symmetric mu,
         vertex lemma; any upper bound of N* is valid, so on a box we use min_k (upper end of candidate k)).
Every quantity is an mpmath interval (outward rounded); the 'x_j infinite' slab needs 3^{1-s} * 1.75 >= 1.
Usage: python cc9_rig72.py S0 S1 TARGET SSTEP
"""
import sys
import mpmath as mp
from mpmath import iv

iv.prec = 90
S0 = mp.mpf(sys.argv[1]); S1 = mp.mpf(sys.argv[2]); TARGET = mp.mpf(sys.argv[3]); SSTEP = mp.mpf(sys.argv[4])
XM = mp.mpf('1.75')
KAP = {(j, d): 1 - iv.cos(2 * iv.pi * j * d / 7) for j in (1, 2) for d in (1, 2, 3)}


def imax(a, b):
    return iv.mpf([max(a.a, b.a), max(a.b, b.b)])


def ub(tau, x1, x2, inf1, inf2, S):
    A1 = 1 - tau; A2 = tau
    cands = [KAP[(1, d)] * A1 + KAP[(2, d)] * A2 for d in (1, 2, 3)]
    M1 = iv.mpf(0) if inf1 else A1 * x1
    M2 = iv.mpf(0) if inf2 else A2 * x2
    M = imax(M1, M2)
    cands.append(iv.mpf(7) / 6 - iv.mpf(7) / 12 * M)
    Nub = min(c.b for c in cands)
    if Nub <= 0:
        return mp.mpf(0)
    t3 = iv.mpf(3) ** (1 - S); c1 = 1 - t3

    def F(x, inf):
        if inf:
            return iv.mpf(1)
        return imax(1 - c1 * x, t3 * imax(iv.mpf(1), x))
    R = 1 / S
    y1 = F(x1, inf1) * A1; y2 = F(x2, inf2) * A2

    def pw(y):
        if y.a <= 0:
            return iv.mpf([0, (iv.mpf(max(y.b, 0)) ** R).b]) if y.b > 0 else iv.mpf(0)
        return y ** R
    tot = pw(y1) + pw(y2)
    if tot.a <= 0:
        return mp.inf
    den = iv.mpf(3.5) * iv.mpf(3) ** (-S) * tot ** S
    return (iv.mpf(Nub) / den).b


def main():
    nbox = 0; fails = 0; worst = mp.mpf(0)
    chk = (iv.mpf(3) ** (1 - iv.mpf(S1)) * XM)
    assert chk.a >= 1, "slab condition fails"
    k = 0
    s_lo = S0
    while s_lo < S1:
        s_hi = min(s_lo + SSTEP, S1)
        S = iv.mpf([s_lo, s_hi])
        stack = [(mp.mpf(0), mp.mpf(1), mp.mpf(0), XM, mp.mpf(0), XM, i1, i2, 0) for i1 in (False, True) for i2 in (False, True)]
        while stack:
            t0, t1, a0, a1, b0, b1, i1, i2, dep = stack.pop()
            v = ub(iv.mpf([t0, t1]), iv.mpf([a0, a1]), iv.mpf([b0, b1]), i1, i2, S)
            nbox += 1
            if v <= TARGET:
                worst = max(worst, v); continue
            if dep > 45:
                fails += 1; print("FAIL", s_lo, s_hi, t0, t1, a0, a1, b0, b1, i1, i2, v); continue
            w = [t1 - t0, mp.mpf(0) if i1 else (a1 - a0) / 2, mp.mpf(0) if i2 else (b1 - b0) / 2]
            i = max(range(3), key=lambda j: w[j])
            if i == 0:
                m = (t0 + t1) / 2; stack += [(t0, m, a0, a1, b0, b1, i1, i2, dep + 1), (m, t1, a0, a1, b0, b1, i1, i2, dep + 1)]
            elif i == 1:
                m = (a0 + a1) / 2; stack += [(t0, t1, a0, m, b0, b1, i1, i2, dep + 1), (t0, t1, m, a1, b0, b1, i1, i2, dep + 1)]
            else:
                m = (b0 + b1) / 2; stack += [(t0, t1, a0, a1, b0, m, i1, i2, dep + 1), (t0, t1, a0, a1, m, b1, i1, i2, dep + 1)]
        s_lo = s_hi
    print(f"[referee cc9] s in [{S0},{S1}] TARGET={TARGET}: boxes={nbox} fails={fails} worst={float(worst):.6f}")


if __name__ == '__main__':
    main()
