"""Check R7(b),(c): S_r(P,q) = sum (1+mu_k)^r over roots of mu^3 - P mu - q; monotone (dec in P, inc in q) on K,
and S_min(Pbar,qlow) = S(Pbar, max(qlow, P-1, -2(P/3)^{3/2})) vs brute-force min over K cap {P<=Pbar, q>=qlow}."""
import numpy as np
rng = np.random.default_rng(3)
def S(P, q, r):
    rts = np.roots([1, 0, -P, -q]).real
    return np.sum(np.maximum(1+rts, 0)**r)
def lowK(P): return max(P-1, -2*(P/3)**1.5)
def upK(P): return 2*(P/3)**1.5
viol = 0; tot = 0
for r in (2/3, 0.75, 0.9):
    for t in range(20000):
        P = rng.uniform(0, 3); q = rng.uniform(lowK(P), upK(P))
        dP = 1e-4; dq = 1e-4
        if q + dq <= upK(P):
            tot += 1; viol += S(P, q+dq, r) < S(P, q, r) - 1e-12
        if q >= lowK(P+dP) and q <= upK(P+dP) and P + dP <= 3:
            tot += 1; viol += S(P+dP, q, r) > S(P, q, r) + 1e-12
print(f"monotonicity violations: {viol}/{tot}")
# S_min formula vs brute force
worst = 0
Pg = np.linspace(0, 3, 601)
for t in range(300):
    r = 2/3; Pbar = rng.uniform(0, 3.2); qlow = rng.uniform(-2.1, 2.1)
    Pc = min(Pbar, 3)
    if qlow > upK(Pc): continue
    formula = S(Pc, max(qlow, lowK(Pc)), r)
    best = np.inf
    for P in Pg[Pg <= Pc]:
        lo = max(qlow, lowK(P)); hi = upK(P)
        if lo > hi: continue
        for q in np.linspace(lo, hi, 25):
            best = min(best, S(P, q, r))
    worst = max(worst, formula - best)
print(f"S_min formula - brute force min (should be <= ~0): max = {worst:.3e}")
