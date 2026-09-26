# Critical discount with a share floor (NUMERICAL; optimizer lower-bounds the value, so delta_crit here is an OVER-estimate).
import numpy as np
from floor_test import value
for p, Gam in ((23, 0.40293339), (59, 0.31304888)):
    for L in (5, 8):
        for eps in (0.03, 0.06):
            T = list(range(1, L + 1))
            if value(p, T, 1.0, eps, Gam, starts=25) >= 1:
                print(f"p={p} AP{{1..{L}}} eps={eps}: pays even with no discount?"); continue
            lo, hi = 1.0, np.sqrt(3)
            for _ in range(10):
                mid = (lo + hi) / 2
                if value(p, T, mid, eps, Gam, starts=25) > 1: hi = mid
                else: lo = mid
            d = (lo + hi) / 2
            print(f"p={p} AP{{1..{L}}} share floor {eps}: delta_crit ~ {d:.3f}  -> depth needed on T >= {(1 - d / np.sqrt(3)) / np.sqrt(3):.3f} (lone-atom curve)", flush=True)
