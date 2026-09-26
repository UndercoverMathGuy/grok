import numpy as np
from tools import *
rng = np.random.default_rng(2)
for p in (11, 13, 17):
    H = list(range(1, (p - 1) // 2 + 1))
    a1, _ = depth_polar(p, H, False, rng, starts=20, iters=400)
    a2, th = depth_polar(p, H, True, rng, starts=20, iters=400)
    print(f"p={p} dense: A* ~ {a1:.4f}  A_s ~ {a2:.4f}  (enlarged depth <= A_s)  transfer lower {transfer(a2, a1):.4f}", flush=True)
