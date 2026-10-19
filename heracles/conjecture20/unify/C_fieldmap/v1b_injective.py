# v1b: at programme maximisers y*, is S restricted to (binding offsets B) x (spt y*) injective?
# (then the decoupled IF optimum has no first-order cost-free pointwise compensation direction)
import numpy as np
from fmlib import *
from cjlib import gamma_exact, gamma_lp
tot = 0; inj = 0; fails = []; Bge = 0
for p in [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]:
    F = (p - 1) // 2
    for s in [1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0, 2.5, 3.0]:
        if p <= 23:
            G, ys = gamma_exact(p, s)
        else:
            G, y = gamma_lp(p, s, restarts=6, seed=0); ys = [y]
        for y in ys:
            a = y / G; sig = kappa_mat(p)[:F] @ a - 1.0
            B = np.where(sig <= 1e-7)[0]; spt = np.where(a > 1e-10)[0]
            Smat = np.sin(TAU * np.arange(1, F + 1)[:, None] * np.arange(1, F + 1)[None, :] / p)
            rk = np.linalg.matrix_rank(Smat[np.ix_(B, spt)], tol=1e-8)
            tot += 1; Bge += len(B) >= len(spt)
            if rk == len(spt): inj += 1
            else: fails.append((p, s, len(B), len(spt), rk))
msg = f'programme maximisers examined: {tot}; |B| >= K*: {Bge}; S_(B,spt) injective: {inj}; failures: {fails}'
print(msg); open('v1b_injective.out', 'w').write(msg + '\n')
