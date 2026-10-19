"""Sanity check of R6 (circularity criterion): a single-output network (output frequency l only) has a circular field
(E(Phi_l - E Phi_l)^2 = 0) when no input slot carries a '2l-pair' {a, a'} (a != a', a +- a' = +-2l mod p), and in general
not otherwise.  Also checks the circular-cone inequality  min_x cm(Phi_l) <= Re conj(beta)... form:
(kappa_min * min cm) = h and  h <= kappa_min (Re beta_l - k_p sigma) with sigma^2 = E|Phi - E Phi|^2 / 2 (companion Thm 14)."""
import itertools, numpy as np
from lib import *
rng = np.random.default_rng(5)


def pairs(p, l):
    F0 = [0] + Fset(p)
    red = lambda x: min(x % p, (-x) % p)
    return {frozenset((a, c)) for a in F0 for c in F0 if a != c and (red(a + c) == red(2 * l) or red(a - c) == red(2 * l))}


def pseudo(U, W, l):
    m, n, p = U.shape
    ph = field_coeffs(U, W, l)
    idx = tuple(np.indices((p,) * n).reshape(n, -1)); neg = tuple((-np.array(idx)) % p)
    return abs((ph[idx] * ph[neg]).sum() - ph[(0,) * n] ** 2), ph


for (p, n, l, freqs) in [(7, 2, 1, [1, 2]), (7, 2, 1, [0, 1]), (7, 2, 1, [1, 3]), (7, 2, 1, [0, 2, 1]), (11, 2, 2, [2, 1]),
                         (11, 2, 2, [2, 3]), (7, 3, 2, [2, 1]), (5, 2, 1, [0, 1]), (3, 2, 1, [0, 1])]:
    P2 = pairs(p, l)
    has_pair = any(frozenset((a, c)) in P2 for a in freqs for c in freqs if a != c)
    m = 4
    U = np.zeros((m, n, p)); W = np.zeros((m, p))
    for i in range(m):
        for t in range(n):
            U[i, t] = sum(proj(rng.normal(size=p), k, p) for k in freqs)
        W[i] = proj(rng.normal(size=p), l, p)
    pv, ph = pseudo(U, W, l)
    print(f"p={p} n={n} l={l} input freqs {freqs}: 2l-pair present: {has_pair}  |pseudo-variance| = {pv:.3e}  "
          f"-> {'OK' if (pv < 1e-12) == (not has_pair) else 'MISMATCH'}")
