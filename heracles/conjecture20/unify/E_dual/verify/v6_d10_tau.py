"""Referee check of the D10 tolerance table: EXACT tau(S,eps) = Gamma_s / Gamma_s^{C(S,eps)} - 1 by vertex enumeration
(concave l_r minimisation over {K y >= 1, y >= 0, y_k >= eps * sum(y) for k in S} attains its minimum at a vertex).
Compare with the agent's reweighted-LP values (floor_tolerance.out), which are only upper bounds on tau."""
import numpy as np, itertools, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlib import *

cases = [(11, 1.5, [1, 4, 2]), (11, 1.5, [1, 4, 3]), (13, 2.0, [2, 3, 1]), (13, 2.0, [2, 3, 4]), (11, 2.5, [1, 4, 2]),
         (13, 1.5, [2, 3, 4, 1])]
for (p, s, S) in cases:
    F = (p - 1) // 2
    G0, y0 = programme_vertices(p, s)
    spt = sorted(int(j) + 1 for j in np.nonzero(y0 > 1e-12)[0])
    out = []
    for eps in [1e-6, 1e-4, 1e-3, 1e-2, 0.05, 0.1]:
        rows = []
        for k in S:
            a = -eps * np.ones(F); a[k - 1] += 1.0; rows.append(a)
        Gc, yc = programme_vertices(p, s, extra_rows=rows)
        out.append('eps=%g: %.5f' % (eps, G0 / Gc - 1))
    print('p=%d s=%.2f Gamma=%.6f spt*=%s  S=%s | %s' % (p, s, G0, spt, S, '; '.join(out)), flush=True)
