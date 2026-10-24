"""Rigorous certificate for the one-dimensional shared-sign family (any odd p, b = 2, single feature).

Input: parameters (lam_i, logV_i, argV_i) of the field  Phi(tau) = e^{i pi tau/p} sum_i V_i e^{lam_i tau}.
The explicit network is
    u_i(a) = eps_a exp(lam_i e_a)      (same vector in every slot),  eps_a = +1 (a < p/2), -1 (a > p/2),
                                        e_a = 2a (a < p/2), 2a - p (a > p/2)   (a permutation of 0..p-1),
    w_i(c) = Re(V_i rho^c),             rho = e^{2 pi i/p},
balanced (each of the nu factors of neuron i gets norm (||w_i|| ||u_i||^n)^{1/nu}) and rounded to D decimals.
The margin h = min_x min_{c != y(x)} f(x)[y(x)] - f(x)[c] is computed EXACTLY in rational arithmetic over all count
vectors (the network is slot-symmetric, so this is the minimum over all p^n inputs), and
    gamma = h / P_2^{nu/2},  P_2 = sum_i (n ||u_i||^2 + ||w_i||^2),   gamma_IF = nu^{1-nu/2} * 2 (2 p nu)^{-nu/2} * Gamma,
with Gamma = kappa_min(p) = 1 - cos(2 pi/p) (valid when K*(nu/2, p) = 1, e.g. p = 3 always), enclosed by interval
arithmetic.

usage: python3 certify_1d.py params.(npy|json) p n [digits]
"""
import sys, math, itertools
from fractions import Fraction as Fr
import numpy as np
from mpmath import iv

iv.dps = 60


def build(x, p, n, digits):
    m = len(x) // 3
    lam, lv, av = x[:m], x[m:2 * m], x[2 * m:]
    lv = lv - lv.max()
    eps = np.array([1.0 if a < p / 2 else -1.0 for a in range(p)])
    e = np.array([2 * a if a < p / 2 else 2 * a - p for a in range(p)], dtype=float)
    rho = np.exp(2j * np.pi * np.arange(p) / p)
    U, W = [], []
    for i in range(m):
        u = eps * np.exp(lam[i] * e)                            # unnormalised: field term V_i e^{lam_i tau}
        w = np.real(np.exp(lv[i] + 1j * av[i]) * rho)
        nu_, nw = np.linalg.norm(u), np.linalg.norm(w)
        t = (nw * nu_ ** n) ** (1.0 / (n + 1))                  # balanced factor norm
        U.append(u / nu_ * t); W.append(w / nw * t)
    Uq = [[Fr("%.*e" % (digits, v)) for v in u] for u in U]
    Wq = [[Fr("%.*e" % (digits, v)) for v in w] for w in W]
    return Uq, Wq


def counts(n, p):
    def rec(prefix, left, k):
        if k == p - 1:
            yield prefix + (left,); return
        for v in range(left + 1):
            yield from rec(prefix + (v,), left - v, k + 1)
    yield from rec((), n, 0)


def margin(U, W, p, n):
    m = len(U); h = None
    for N in counts(n, p):
        y = sum(a * N[a] for a in range(p)) % p
        z = [math.prod(U[i][a] ** N[a] for a in range(p)) for i in range(m)]
        f = [sum(W[i][c] * z[i] for i in range(m)) for c in range(p)]
        g = min(f[y] - f[c] for c in range(p) if c != y)
        h = g if h is None or g < h else h
    return h


def to_iv(q):
    return iv.mpf(q.numerator) / iv.mpf(q.denominator)


def certify(U, W, p, n):
    nu = n + 1
    h = margin(U, W, p, n)
    P2 = sum(n * sum(a * a for a in U[i]) + sum(a * a for a in W[i]) for i in range(len(U)))
    gamma = to_iv(h) / iv.exp((iv.mpf(nu) / 2) * iv.log(to_iv(P2)))
    kmin = 1 - iv.cos(2 * iv.pi / p)
    gIF = iv.exp((1 - iv.mpf(nu) / 2) * iv.log(iv.mpf(nu))) * 2 * iv.exp(-(iv.mpf(nu) / 2) * iv.log(iv.mpf(2 * p * nu))) * kmin
    return h, P2, gamma / gIF


if __name__ == "__main__":
    f, p, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    digits = int(sys.argv[4]) if len(sys.argv) > 4 else 9
    if f.endswith('.json'):
        import json
        d = json.load(open(f)); x = np.concatenate([d['lam'], d['logV'], d['argV']])
    else:
        x = np.load(f)
    U, W = build(x, p, n, digits)
    h, P2, ratio = certify(U, W, p, n)
    print("p=%d n=%d width %d (nu=%d), weights rounded to %d significant digits, exact rational margin"
          % (p, n, len(U), n + 1, digits + 1))
    print("  margin h > 0:", h > 0, "  h =", float(h))
    print("  gamma/gamma_IF in", ratio)
    print("  CERTIFIED > 1:", ratio.a > 1)
