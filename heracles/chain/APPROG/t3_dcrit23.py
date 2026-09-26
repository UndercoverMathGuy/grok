# Exact (complete vertex enumeration, fp64) delta_crit(AP_L, eps) at p = 23.
import json, numpy as np
from approg_lib import *
p = 23; G = gamma_star_exact(p)
eps_list = [G ** 2 / 12, G ** 2 / (12 * (25 / 16) ** 2), 0.03]
for eps in eps_list:
    for L in range(1, 12):
        T = list(range(1, L + 1))
        if L * eps > 1:
            continue
        d, safe = delta_crit(lambda dl: W_exact(p, T, dl, eps, G), steps=26)
        print(json.dumps(dict(p=p, L=L, eps=round(eps, 6), forced_share=round(L * eps, 4), delta_crit=round(d, 6),
                              safe_at_sqrt3=bool(safe),
                              depth_needed=round(max(0.0, (1 - d / np.sqrt(3)) / np.sqrt(3)), 5))), flush=True)
