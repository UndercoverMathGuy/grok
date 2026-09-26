# Rigorous-input race for progressions at any p: transfer c(A*, A_s) with A* = 0 (worst case) and
# A_s = 1/sqrt(2L) (PROVED Parseval floor), vs the a_req values measured at p = 113 (crit_vs_L.log).
import sys, numpy as np
sys.path.insert(0, '..')
from transfer_flip import depth
req = {4: (0.122, 0.103), 6: (0.076, 0.087), 8: (0.090, 0.093), 12: (0.076, 0.037), 16: (0.038, None)}
for L in (4, 6, 8, 12, 16):
    As = (2 * L) ** -0.5
    c0, _ = depth(As, 0.0, n=60)
    cL, _ = depth(As, 0.0, n=60)
    r = req[L]
    print(f"AP L={L:2d}: A_s >= {As:.3f} (proved), A* >= 0 -> transfer depth >= {c0:.3f} | a_req at p=113, eps=.02/.04: {r[0]} / {r[1]}"
          f" -> {'WIN' if c0 >= max(x for x in r if x is not None) else 'no'}")
