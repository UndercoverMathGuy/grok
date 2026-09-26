import time, numpy as np
from approg_lib import *
rng = np.random.default_rng(2)
G23 = gamma_star_exact(23)
bad = 0; t = time.time()
for _ in range(16):
    L = int(rng.integers(1, 9)); dl = float(rng.uniform(1, 1.73)); eps = float(rng.choice([0.005, 0.012, 0.03]))
    T = list(range(1, L + 1))
    we = W_exact(23, T, dl, eps, G23)
    st, _ = decide(23, T, dl, eps, G23)
    truth = 'ge' if we >= 1 else 'lt'
    bad += (st != truth)
    print(f"L={L} d={dl:.3f} eps={eps}: exact W={we:.6f} -> {truth}; B&B -> {st}", flush=True)
print(f"mismatches {bad}/16  [{time.time() - t:.0f}s]")
