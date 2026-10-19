import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlib import *
for (p, n) in [(5, 2), (7, 3), (3, 4)]:
    F = (p - 1) // 2; y = np.ones(F)
    U, W = if_block_net(p, n, y)
    f = table(U, W); g = gaps(f, p, n); beta = clean(f, p, n)
    print(p, n, 'beta', np.round(beta, 6), 'gap - kappa Re beta max', abs(g - (kap(p) @ beta.real)[None, :]).max(),
          'budget ratio', np.abs(atoms(U, W)).sum(1) / (Bp(p, n) * acost(U, W)))
