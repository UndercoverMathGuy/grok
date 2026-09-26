import numpy as np
from tools import depth_osc, depth_polar
rng = np.random.default_rng(8)
for p, T in [(59, list(range(1, 9))), (59, list(range(1, 6))), (59, [1, 7, 11, 23, 27]), (17, list(range(1, 9))), (13, list(range(1, 7)))]:
    Ao = depth_osc(p, T, rng)
    print(f"p={p} T={T}: A_osc ~ {Ao:.4f}   (floor sqrt(2/K) = {np.sqrt(2 / len(T)):.4f})", flush=True)
