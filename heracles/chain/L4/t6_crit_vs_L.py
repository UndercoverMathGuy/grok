# Does the critical discount of progressions approach full discount sqrt3 as the length grows (share floor eps)?
# p = 113, Gamma* = 0.27593576 (LADDER free-share value, a lower bound -> ratios over-estimated -> conservative).
# Uses L5/floor_test.value (SLSQP multistart: value lower bounds -> delta_crit over-estimated). Also prints the depth the
# race needs, a_req = (1 - delta_crit/sqrt3)/sqrt3, next to the proved ceiling 2/sqrt(L) and the search two-sided depth.
import sys, numpy as np
sys.path.insert(0, '..'); sys.path.insert(0, '../L5')
from floor_test import value
p, Gam = 113, 0.27593576
for L in (4, 6, 8, 12, 16):
    T = list(range(1, L + 1))
    for eps in (0.02, 0.04):
        if L * eps > 0.9: continue
        full = value(p, T, np.sqrt(3), eps, Gam, starts=12)
        if full <= 1:
            print(f"p={p} AP L={L:2d} eps={eps}: full-discount ratio {full:.3f} <= 1 -> delta_crit = sqrt3, a_req = 0", flush=True); continue
        lo, hi = 1.0, np.sqrt(3)
        if value(p, T, 1.0, eps, Gam, starts=12) > 1: lo = hi = 1.0
        for _ in range(8):
            if hi - lo < 1e-3: break
            mid = (lo + hi) / 2
            if value(p, T, mid, eps, Gam, starts=12) > 1: hi = mid
            else: lo = mid
        d = (lo + hi) / 2; a = max(0.0, (1 - d / np.sqrt(3)) / np.sqrt(3))
        print(f"p={p} AP L={L:2d} eps={eps} (forced share {L*eps:.2f}): full-discount ratio {full:.3f}, delta_crit {d:.3f}, a_req {a:.3f} | ceiling 2/sqrt(L) = {2/np.sqrt(L):.3f}", flush=True)
