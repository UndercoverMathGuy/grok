import time, numpy as np
from tools import *
rng = np.random.default_rng(1)
p = 59
for H in ([1, 6, 14, 16, 19], [1, 4, 12, 18, 25, 27], list(range(1, 30))[:5]):
    t = time.time()
    a1 = depth_est(p, H, False, rng); a2 = depth_est(p, H, True, rng)
    print(H, 'd', dissociated_dimension(p, H), 'j', shortest_relation(p, H), 'E', round(energy(p, H), 3),
          'A*', round(a1, 3), 'A_s', round(a2, 3), 'transfer', round(transfer(a2, a1), 3),
          'Gamma_H', round(gamma_H(p, H, rng)[0], 4), f'{time.time() - t:.1f}s', flush=True)
t = time.time(); print('enlarged est', round(enlarged_depth_est(p, [1, 6, 14, 16, 19], rng, extra=4), 3), f'{time.time() - t:.0f}s')
