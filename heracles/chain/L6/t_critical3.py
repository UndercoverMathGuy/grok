import json, numpy as np
from floor_opt import floored_value, rate_matrix
from tools import gamma_star
rng = np.random.default_rng(23); p = 59; Gam, S = gamma_star(p, rng)
T = [1, 2, 5, 8, 13, 15, 16, 18, 19, 28]; d = max(1.0, np.sqrt(3) * (1 - np.sqrt(3) * 0.1859))
for eps in (0.01, 0.03):
    print(json.dumps(dict(case='adversary K=10 p=59, delta(c_lo=.186)', delta=round(d, 4), eps=eps,
                          ratio=round(floored_value(p, T, rate_matrix(p, T, d), eps, Gam, rng, seeds=[S]), 4))), flush=True)
