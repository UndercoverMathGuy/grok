import time, numpy as np
from approg_lib import *
t = time.time()
G23 = gamma_star_exact(23)
print(f"Gamma*(23) exact enumeration = {G23:.8f} (L5 value 0.40293339)  [{time.time() - t:.1f}s]")
# heuristic vs exact on a few floored/discounted instances
for L, dl, eps in ((3, 1.2, 0.02), (6, 1.4, 0.02), (9, 1.6, 0.012)):
    T = list(range(1, L + 1))
    we = W_exact(23, T, dl, eps, G23)
    wh, _ = W_heur(23, T, dl, eps, G23, starts=40)
    print(f"AP{L} delta={dl} eps={eps}: W exact {we:.6f}  heur {wh:.6f}")
