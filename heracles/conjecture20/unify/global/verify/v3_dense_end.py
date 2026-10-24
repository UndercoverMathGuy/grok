# R7 add-on: "every prime p >= 11 has a K*-transition below s = 1.5" (REPORT checks 11..199 in floating point).
# Logic (referee's reconstruction): programme maximisers are vertices of the coverage polytope (strict concavity of
# sum v^r along any non-degenerate segment); the only possible full-support vertex is v = (2/p) 1 (all F coverage
# constraints active; sum_j kappa_j(delta) = p/2), whose value is (p/2) F^{-s}.  K*(1,p) = F (the uniform allocation is the
# unique LP optimum at s = 1).  So any feasible allocation with value > (p/2) F^{-s} at s = 1.5 certifies K*(1.5,p) < F,
# i.e. a transition in (1, 1.5).
#  Part A: interval-certified (mpmath.iv) allocations for primes 11 <= p <= PMAX, allocations from the referee's own
#          reweighted-LP heuristic (independent code).
#  Part B: analytic bound for large p: lacunary support {2^k : 0<=k<K}, 2^{K-1} >= p/3, gives Gamma_{1.5} >= 1.5 K^{-1.5}.
import sys, math, numpy as np
from scipy.optimize import linprog
import mpmath as mp

PMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 400
s = 1.5; r = 1 / s


def primes(a, b):
    return [q for q in range(a, b + 1) if q > 1 and all(q % d for d in range(2, int(q ** .5) + 1))]


def alloc(p):
    """cheap first: reweighted LP restricted to a lacunary support {2^k mod p} (as +-classes);
    full-support reweighted LP only if that does not beat the uniform vertex"""
    F = (p - 1) // 2
    K = 1 - np.cos(2 * np.pi * np.outer(np.arange(1, F + 1), np.arange(1, F + 1)) / p)   # rows delta, cols j
    unif = (p / 2) * F ** (-s)

    def rw(cols, iters):
        w = np.ones(len(cols)); A = K[:, cols]
        for it in range(iters):
            res = linprog(w, A_ub=-A, b_ub=-np.ones(F), bounds=(0, None), method='highs')
            v = res.x; wn = (v + 1e-9) ** (r - 1)
            if np.allclose(wn, w, rtol=1e-8): break
            w = wn
        out = np.zeros(F); out[cols] = np.where(v > 1e-9, v, 0.0)
        return out
    cols = sorted({min(pow(2, k, p), p - pow(2, k, p)) - 1 for k in range(0, int(math.log2(p)) + 2)})
    v = rw(cols, 60)
    valf = (K @ v).min() / np.sum(v[v > 0] ** r) ** s
    if valf <= unif * (1 + 1e-6):
        v = rw(list(range(F)), 40)
        valf = (K @ v).min() / np.sum(v[v > 0] ** r) ** s
    return valf, v


def certify(p, v):
    """rigorous lower bound of value(v) and rigorous upper bound of the uniform-vertex value"""
    iv = mp.iv; iv.dps = 30
    F = (p - 1) // 2
    J = [j for j in range(F) if v[j] > 0]
    vv = [iv.mpf(float(v[j])) for j in J]                     # exact binary rationals
    cov_lo = None
    for d in range(1, F + 1):
        c = iv.mpf(0)
        for x, j in zip(vv, J):
            c += x * (1 - iv.cos(2 * iv.pi * (j + 1) * d / p))
        cov_lo = c.a if cov_lo is None else min(cov_lo, c.a)
    den = iv.mpf(0)
    for x in vv:
        den += x ** (iv.mpf(2) / 3)
    lower = iv.mpf(cov_lo) / den ** iv.mpf('1.5')
    unif = iv.mpf(p) / 2 * iv.mpf(F) ** iv.mpf('-1.5')
    return lower.a, unif.b, len(J)


bad = []; worst = (1e9, None)
for p in primes(11, PMAX):
    valf, v = alloc(p)
    lo, un, k = certify(p, v)
    ok = lo > un
    ratio = float((lo / un).a)
    worst = min(worst, (ratio, p))
    if not ok: bad.append(p)
    if p <= 61 or p % 100 < 4 or not ok:
        print(f"p={p}: certified Gamma_1.5 >= {float(lo):.8f} with {k} features; uniform vertex <= {float(un):.8f}; ratio {ratio:.4f}; {'OK' if ok else 'FAIL'}", flush=True)
print(f"Part A: primes 11..{PMAX}: failures {bad}; smallest certified ratio {worst[0]:.4f} at p={worst[1]}")

# Part B: analytic lacunary bound, check from which K on it beats the uniform vertex for every p in the K-block
print("Part B: lacunary bound 1.5 K^-1.5 vs uniform (p/2)((p-1)/2)^-1.5, p in (3*2^(K-2), 3*2^(K-1)] (RHS decreasing in p, so check block start)")
for K in range(4, 40):
    p0 = 3 * 2 ** (K - 2) + 1       # smallest integer in block
    lhs = 1.5 * K ** -1.5; rhs = (p0 / 2) * ((p0 - 1) / 2) ** -1.5
    if K <= 14 or lhs <= rhs:
        print(f"  K={K}: block starts p={p0}: lacunary {lhs:.5f} vs uniform {rhs:.5f}: {'OK' if lhs > rhs else 'no'}")
