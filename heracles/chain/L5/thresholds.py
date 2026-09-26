# Whole-heavy-set lemma thresholds: the network loses if delta + 2 sqrt(eps) sigma_L < Gamma*/sqrt3.
import numpy as np
from l5lib import *
G = {}
for p in (5, 7, 11, 13, 17, 19, 23):
    V = vertices(p); h = V.shape[1]
    G[p] = gamma_c(V, np.full(h, 1.0))[0]            # Gamma* (c = 1 gives the plain programme)
for p, g in ((59, 0.31304888), (71, 0.30803827), (113, 0.27593576)):
    G[p] = g                                          # LADDER.md free-share values (exhaustive for small K only; lower bounds)
print("p   Gamma*      Gamma*/sqrt3   max eps with delta=0 (2 sqrt(eps) < Gamma*/sqrt3)")
for p, g in G.items():
    thr = g / np.sqrt(3)
    print(f"{p:3d} {g:.8f}  {thr:.6f}      {(thr / 2) ** 2:.5f}")
