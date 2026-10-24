"""Evaluate the 2-parameter reduction of the n=3 pointwise payment (floating point exploration).

Cost lemma:  theta Q^s >= L * F_r(P),  F_r(P) = min{ M_r(nu) : nu>=0, sum nu = 4, sum (nu-1)^2 <= 2P }.
Chain (0,1,1,2): P <= 4 m^2 + 2 G(a,m)^2,  G = cos(max(arccos m - arccos a, 0)),  a=|A|/L, m=M/L.
Claim to certify:  a - c m <= F_r(Pmax(a,m)) for all a,m in [0,1].
K(s) = sup_{a,m} (a - F_r(Pmax))/m  is then an upper bound for c_1(3,s).
"""
import numpy as np


def Fr(P, r):
    """min of M_r over spectra (k zeros, j copies of c, rest e), sum=4, sum (nu-1)^2 = 2P' for P' <= P.
    Uses monotonicity: F_r nonincreasing, so evaluate at P (P capped at 6)."""
    P = min(P, 6.0)
    best = np.inf
    for k in range(0, 4):           # zeros
        m = 4 - k
        S2 = 2 * P - k              # sum over the m nonzero of (nu-1)^2
        if S2 < 0:
            continue
        for j in range(1, m + 1):   # j copies of c, (m-j) copies of e
            if j == m:
                # all equal: c = 4/m, need m (4/m - 1)^2 == S2
                c = 4.0 / m
                if abs(m * (c - 1) ** 2 - S2) < 1e-9:
                    best = min(best, (m * c ** r / 4) ** (1 / r))
                continue
            # j c + (m-j) e = 4 ; j (c-1)^2 + (m-j)(e-1)^2 = S2 ; with c-1 = x, e-1 = y:
            # j x + (m-j) y = 4 - m =: T ; j x^2 + (m-j) y^2 = S2
            T = 4 - m
            # y = (T - j x)/(m-j) ; j x^2 + (T - j x)^2/(m-j) = S2
            qa = j + j * j / (m - j); qb = -2 * T * j / (m - j); qc = T * T / (m - j) - S2
            disc = qb * qb - 4 * qa * qc
            if disc < 0:
                continue
            for x in [(-qb + np.sqrt(disc)) / (2 * qa), (-qb - np.sqrt(disc)) / (2 * qa)]:
                y = (T - j * x) / (m - j)
                c, e = 1 + x, 1 + y
                if c < -1e-12 or e < -1e-12:
                    continue
                c, e = max(c, 0), max(e, 0)
                val = ((j * c ** r + (m - j) * e ** r) / 4) ** (1 / r)
                best = min(best, val)
    return best


def G(a, m):
    if m >= a:
        return 1.0
    return a * m + np.sqrt((1 - a * a) * (1 - m * m))


def Pmax(a, m):
    return 4 * m * m + 2 * G(a, m) ** 2


def K(s, na=400, nm=400):
    r = 1 / s
    best = (-np.inf, None)
    for a in np.linspace(0.5, 1, na):
        for m in np.linspace(1e-3, 1, nm):
            v = (a - Fr(Pmax(a, m), r)) / m
            if v > best[0]:
                best = (v, (a, m))
    return best


if __name__ == '__main__':
    for s in [1.25, 1.5, 1.7, 1.8, 1.8552, 1.9, 1.95, 2.0]:
        v, am = K(s, 300, 300)
        x = 4 ** (1 - s)
        print(f"s={s:.4f}: K(s) ~ {v:.4f} at a={am[0]:.4f} m={am[1]:.4f};  single atom 1-x={1-x:.4f};  K/(1-x)={v/(1-x):.4f}", flush=True)
