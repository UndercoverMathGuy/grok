# Targeted strong-optimiser checks on the critical sets (floor_opt.py).
import json, numpy as np
from floor_opt import floored_value, rate_matrix
from tools import gamma_star, cov_matrix
rng = np.random.default_rng(21)
delta = lambda a: max(1.0, np.sqrt(3) * (1 - np.sqrt(3) * a))
p = 59; Gam, S = gamma_star(p, rng)
cases = [('dense p=59, delta(A_s=0.140)', list(range(1, 30)), delta(0.1402)),
         ('dense p=59, delta(c_lo=0.055)', list(range(1, 30)), delta(0.0547)),
         ('AP8 p=59, delta(c_lo=0.161)', list(range(1, 9)), delta(0.1614)),
         ('AP5 p=59, delta(c_lo=0.230)', list(range(1, 6)), delta(0.2301))]
for name, T, d in cases:
    for eps in (0.005, 0.01, 0.02, 0.03):
        if eps * len(T) > 0.95: continue
        v = floored_value(p, T, rate_matrix(p, T, d), eps, Gam, rng, seeds=[S])
        print(json.dumps(dict(case=name, delta=round(d, 4), eps=eps, ratio=round(v, 4))), flush=True)
