import sys, time, numpy as np
from tools import *
rng = np.random.default_rng(0)
seeds = {59: [[1, 6, 14, 16, 19], [1, 7, 11, 23, 27]], 113: [[1, 6, 16, 31, 35, 44, 55], [1, 5, 12, 15, 38, 46]]}
for p in [int(a) for a in sys.argv[1:]]:
    t = time.time(); G, S = gamma_star(p, rng, restarts=300 if p < 500 else 120, seeds=seeds.get(p, ()))
    print(p, round(G, 6), 'K', len(S), S, f'{time.time() - t:.0f}s', flush=True)
