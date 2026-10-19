# At the programme optimum (r = 1/s < 1) the dual coverage K_l(mu*) is often maximal at a NON-support frequency:
# the x-uniform KKT measure of the decoupled optimum has its injective norm attained by a non-feature neuron.
import numpy as np
from klib import *
from check_lemmas import programme_dual
for p in [7, 11, 13, 17]:
    for s in [1.1, 1.3, 1.5, 1.8]:
        G, ys, zs = programme_exact(p, s)
        for y in ys[:1]:
            mu, res, _ = programme_dual(p, s, y)
            K = kappa_mat(p).T @ mu
            spt = np.where(y > 0)[0]; off = np.setdiff1d(np.arange(len(y)), spt)
            kin = K[spt].max(); kout = K[off].max() if len(off) else np.nan
            print(f'p={p:2d} s={s}: |spt y*|={len(spt)}, dual residual {res:.0e}; max_l in spt K_l/Gamma = {kin/G:.4f}; max_l off spt K_l/Gamma = {kout/G:.4f}  -> argmax off support: {bool(kout>kin)}')

# which of the proper-support optima are uniform on their support (covered by the proof of R9)?
print('uniformity of optimal allocations with proper support:')
for p in [7, 11, 13, 17]:
    for s in [1.1, 1.3, 1.5, 1.8]:
        G, ys, zs = programme_exact(p, s)
        y = ys[0]; spt = y[y > 0]
        if len(spt) < len(y):
            print(f'  p={p} s={s}: support size {len(spt)}, max/min of y on support = {spt.max()/spt.min():.6f}')
