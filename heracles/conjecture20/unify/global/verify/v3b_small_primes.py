# Interval certificate that K*(1.5,p) < F for small primes where the reweighted-LP heuristic of v3 may return the
# uniform vertex: enumerate allocations on all supports of size 2 and 3 (LP for the best weights on that support is
# replaced by exact 'equalised' vertices: solve |S| active coverage constraints), keep the best, certify by mpmath.iv.
import itertools, math, sys
import numpy as np
import mpmath as mp

s = 1.5; r = 1 / s


def best_small_vertex(p):
    F = (p - 1) // 2
    K = 1 - np.cos(2 * np.pi * np.outer(np.arange(1, F + 1), np.arange(1, F + 1)) / p)
    best = (0, None)
    for k in (1, 2, 3):
        for S in itertools.combinations(range(F), k):
            for act in itertools.combinations(range(F), k):
                M = K[np.ix_(act, S)]
                if abs(np.linalg.det(M)) < 1e-12: continue
                v = np.linalg.solve(M, np.ones(k))
                if (v <= 0).any(): continue
                cov = K[:, S] @ v
                if cov.min() < 1 - 1e-12: continue
                val = cov.min() / np.sum(v ** r) ** s
                if val > best[0]:
                    full = np.zeros(F); full[list(S)] = v; best = (val, full)
    return best


def certify(p, v):
    iv = mp.iv; iv.dps = 30; F = (p - 1) // 2
    J = [j for j in range(F) if v[j] > 0]; vv = [iv.mpf(float(v[j])) for j in J]
    lo = None
    for d in range(1, F + 1):
        c = iv.mpf(0)
        for x, j in zip(vv, J): c += x * (1 - iv.cos(2 * iv.pi * (j + 1) * d / p))
        lo = c.a if lo is None else min(lo, c.a)
    den = iv.mpf(0)
    for x in vv: den += x ** (iv.mpf(2) / 3)
    return (iv.mpf(lo) / den ** iv.mpf('1.5')).a, (iv.mpf(p) / 2 * iv.mpf(F) ** iv.mpf('-1.5')).b


for p in [11, 13, 17, 19, 23]:
    val, v = best_small_vertex(p)
    lo, un = certify(p, v)
    print(f"p={p}: support {[j + 1 for j in range(len(v)) if v[j] > 0]}: certified value >= {float(lo):.8f} vs uniform <= {float(un):.8f}: {'OK' if lo > un else 'FAIL'} (ratio {float((lo / un).a) if hasattr(lo / un, 'a') else float(lo / un):.4f})")
