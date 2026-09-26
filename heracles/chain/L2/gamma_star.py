# Heavy-only programme: lower bounds on Gamma*(p) by multistart (u = 0), p = 59, 113, 211.
import sys, json, numpy as np
from progopt import solve, cover
rng = np.random.default_rng(0)
out = {}
for p in [int(a) for a in sys.argv[1:]]:
    v, z = solve(p, 0.0, 1.0, 60, rng)
    G = v * np.sqrt(3); h = (p - 1) // 2
    S = [int(k) + 1 for k in np.nonzero(z[:h] > 1e-6)[0]]
    out[p] = dict(Gamma_lb=G, support=S)
    print(p, json.dumps(out[p]), flush=True)
json.dump(out, open(f'gamma_star_{"_".join(sys.argv[1:])}.json', 'w'), indent=1)
