import sys, numpy as np
sys.path.insert(0, '..')
from transfer_flip import depth
from l4lib import two_sided_depth
rng = np.random.default_rng(5)
req = {8: 0.093, 12: 0.076, 16: 0.038}
for L in (8, 12, 16):
    As = two_sided_depth(113, list(range(1, L + 1)), rng, starts=8)
    c0, _ = depth(As, 0.0); c1, _ = depth(As, 1 / L)
    print(f"p=113 AP L={L}: A_s (search, upper est.) {As:.3f}: transfer with A*=0: {c0:.3f}, with A*=1/L: {c1:.3f} | max a_req {req[L]}")
