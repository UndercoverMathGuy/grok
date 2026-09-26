import numpy as np, time
from approg_lib import *
rng = np.random.default_rng(1)
G23 = gamma_star_exact(23)
worst = 0; n = 0; t = time.time()
for _ in range(24):
    L = int(rng.integers(1, 9)); dl = float(rng.uniform(1, 1.73)); eps = float(rng.choice([0.005, 0.012, 0.03]))
    T = list(range(1, L + 1))
    we = W_exact(23, T, dl, eps, G23)
    wh, _ = W_heur2(23, T, dl, eps, G23, rng=rng)
    worst = max(worst, (wh - we) / we); n += 1
print(f"heur2 vs exact over {n} instances: max relative overestimate {worst:.2e}  [{time.time() - t:.0f}s]")
