import time, numpy as np
from tools import *
rng = np.random.default_rng(1)
p = 59
for H in ([1, 6, 14, 16, 19], [1, 4, 12, 18, 25, 27]):
    t = time.time()
    print(H, 'A* polar', round(depth_polar(p, H, False, rng)[0], 4), 'A_s polar', round(depth_polar(p, H, True, rng)[0], 4), f'{time.time() - t:.1f}s', flush=True)
print('grid: LADDER A* in [0.401, 0.463], A_s in [0.429, 0.496]; old A* in [0.336, 0.388] (L=6; certified >= 0.358 at L=8), A_s in [0.352, 0.406]')
