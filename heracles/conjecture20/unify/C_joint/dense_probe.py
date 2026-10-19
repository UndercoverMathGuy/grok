# How wide is the dense regime? Compare the uniform allocation on F with cosine-modulated allocations
# y_l = 1 - eps cos(2 pi l d0/p) (all l in F), and with the Legendre-network ratio.
import numpy as np
from cjlib import kappa_full
for p in [101, 1009, 1019, 4007]:
    F = (p-1)//2; K = kappa_full(p)[:F]; l = np.arange(1, F+1)
    for s in [1.002, 1.005, 1.01, 1.02, 1.05]:
        r = 1/s
        U = np.min(K@np.ones(F))/F**s
        best = (1.0, None)
        for d0 in [1, 2, 3, F//3, F//2]:
            for eps in [0.1, 0.3, 0.5, 0.8, 0.95]:
                y = 1 - eps*np.cos(2*np.pi*l*d0/p)
                V = np.min(K@y)/np.sum(y**r)**s
                if V/U > best[0]: best = (V/U, (d0, eps))
        tau = 1.5**(s-1); RL = tau*(1-2/np.sqrt(p))/best[0]
        print(f"p={p} s={s}: best modulated/uniform = {best[0]:.6f} at {best[1]};  Legendre-network ratio vs this reference <= {RL:.5f}")
