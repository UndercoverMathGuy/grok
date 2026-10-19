"""Rigorous certificate for general b (p = 3): the margin h is computed exactly (rational weights), and
   P_b = sum_i ( n ||u_i||^2 + ||w_i||^2 )^{b/2},   gamma = h / P_b^{nu/b},
   gamma_IF(b) = nu^{1 - nu/b} * 2 (6 nu)^{-nu/2} * 3/2       (Theorem 8 with Gamma_s(3) = 3/2)
are enclosed with mpmath interval arithmetic.  Prints a verified lower bound on gamma / gamma_IF.

usage: python3 interval_certificate.py <family_params.json|.npy> n width b [digits]
"""
import sys, json
from fractions import Fraction as Fr
import numpy as np
from mpmath import iv, mp

sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from certify import build

iv.dps = 60


def exact_margin(U, W, n):
    m = len(U); h = None
    for N0 in range(n + 1):
        for N1 in range(n + 1 - N0):
            N2 = n - N0 - N1; y = (N1 + 2 * N2) % 3
            z = [U[i][0] ** N0 * U[i][1] ** N1 * U[i][2] ** N2 for i in range(m)]
            f = [sum(W[i][c] * z[i] for i in range(m)) for c in range(3)]
            g = min(f[y] - f[c] for c in range(3) if c != y)
            h = g if h is None or g < h else h
    return h


def to_iv(q):
    return iv.mpf(q.numerator) / iv.mpf(q.denominator)


def certify_b(U, W, n, b):
    nu = n + 1
    h = exact_margin(U, W, n)
    bq = Fr(b)
    norms2 = [n * sum(a * a for a in U[i]) + sum(a * a for a in W[i]) for i in range(len(U))]
    B = to_iv(bq)
    Pb = sum(iv.exp((B / 2) * iv.log(to_iv(s))) for s in norms2)
    gamma = to_iv(h) / iv.exp((iv.mpf(nu) / B) * iv.log(Pb))
    gIF = iv.exp((1 - iv.mpf(nu) / B) * iv.log(iv.mpf(nu))) * 2 * iv.exp(-(iv.mpf(nu) / 2) * iv.log(iv.mpf(6 * nu))) * iv.mpf(3) / 2
    ratio = gamma / gIF
    return h, Pb, gamma, gIF, ratio


if __name__ == "__main__":
    f, n, m, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    digits = int(sys.argv[5]) if len(sys.argv) > 5 else 7
    params = np.array(json.load(open(f))["params"]) if f.endswith(".json") else np.load(f)
    Um, Wm = build(params, n, m, 12)
    U = [[Fr("%.*f" % (digits, float(x))) for x in u] for u in Um]
    W = [[Fr("%.*f" % (digits, float(x))) for x in w] for w in Wm]
    h, Pb, gamma, gIF, ratio = certify_b(U, W, n, b)
    print("p=3 n=%d width %d b=%s  (weights rounded to %d decimals, exact rationals)" % (n, m, b, digits))
    print("  margin h (exact)        = %.12f" % float(h))
    print("  P_b in                  ", Pb)
    print("  gamma / gamma_IF in     ", ratio)
    print("  certified > 1:", ratio.a > 1)
