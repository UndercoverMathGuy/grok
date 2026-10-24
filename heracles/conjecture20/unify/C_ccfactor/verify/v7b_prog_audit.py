# Referee check v7b (CC15): is prog.solve(p, s, restarts=4) -- used by c5_struct (c) and c7_quad (ii) -- actually
# returning programme MAXIMISERS on the 144-case grid?  Exact vertex enumeration for p <= 19; for larger p, a certified
# lower bound on Gamma_s from K=1 (kappa_min) and an exhaustive search over all 2-supports (vertex enumeration in R^2).
import sys, itertools
import numpy as np
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/mechanism_scripts')
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/C_ccfactor/verify')
import prog, vlib as V
from sympy import primerange


def best_two_support(p, s):
    F = (p - 1) // 2; r = 1 / s; K = V.kap(p)[:F]; best = (1 - np.cos(2 * np.pi / p), None)
    for (i, j) in itertools.combinations(range(F), 2):
        Ks = K[:, [i, j]]
        A = np.vstack([Ks, np.eye(2)]); bb = np.concatenate([np.ones(F), np.zeros(2)])
        for act in itertools.combinations(range(F + 2), 2):
            Mm = A[list(act)]
            if abs(np.linalg.det(Mm)) < 1e-12: continue
            v = np.linalg.solve(Mm, bb[list(act)])
            if np.any(v < -1e-12) or np.any(Ks @ v < 1 - 1e-10): continue
            v = np.clip(v, 0, None); val = np.sum(v[v > 0] ** r) ** (-s)
            if val > best[0] + 1e-13: best = (val, (i + 1, j + 1))
    return best


sub = 0; tot = 0; rows = []
for p in primerange(5, 62):
    for s in [1.05, 1.2, 1.35, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0]:
        Gp, y, cov = prog.solve(p, s, restarts=4)
        if p <= 19:
            Gx, yx = V.programme(p, s); tag = 'exact'
        else:
            Gx, _ = best_two_support(p, s); tag = 'lower bound (K<=2)'
            Gx = max(Gx, Gp)
        tot += 1
        if Gx > Gp * (1 + 1e-9):
            sub += 1; rows.append((p, s, round(Gp, 6), round(Gx, 6), tag))
print(f'prog.solve(restarts=4) suboptimal in {sub}/{tot} grid cases (p in [5,61], 9 values of s):')
for rr in rows:
    print('  p=%d s=%.2f  prog=%.6f  better=%.6f  (%s)' % rr)
