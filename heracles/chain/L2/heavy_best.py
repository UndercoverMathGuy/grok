# Best heavy-only spectra (lower bounds on Gamma*), seeded with known supports (opus55 p=59, LADDER p=113) + multistart.
import json, sys, numpy as np
from progopt import solve
rng = np.random.default_rng(1)
known = {59: [1, 7, 11, 23, 27], 113: [1, 6, 16, 31, 35, 44, 55]}
out = {}
for p in (59, 113, 211):
    h = (p - 1) // 2; seeds = []
    for S in ([known[p]] if p in known else []):
        for _ in range(4):
            x = np.zeros(h); x[np.array(S) - 1] = rng.dirichlet(np.ones(len(S)) * 20); seeds.append(np.r_[x, np.zeros(h)])
    v, z = solve(p, 0.0, 1.0, 120 if p == 211 else 60, rng, x0s=seeds)
    out[p] = dict(Gamma_lb=float(v * np.sqrt(3)), x=z[:h].tolist(), support=[int(k) + 1 for k in np.nonzero(z[:h] > 1e-7)[0]])
    print(p, out[p]['Gamma_lb'], out[p]['support'], flush=True)
json.dump(out, open('heavy_best.json', 'w'))
