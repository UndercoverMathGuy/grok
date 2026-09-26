import json, numpy as np
from floor_opt import floored_value, rate_matrix
from tools import gamma_star
rng = np.random.default_rng(22)
p = 59; Gam, S = gamma_star(p, rng); T = list(range(1, 30))
for d in (1.053, 1.13, 1.2247):
    for eps in (0.005, 0.01):
        v = floored_value(p, T, rate_matrix(p, T, d), eps, Gam, rng, seeds=[S])
        print(json.dumps(dict(case='dense p=59', delta=d, eps=eps, ratio=round(v, 4))), flush=True)
